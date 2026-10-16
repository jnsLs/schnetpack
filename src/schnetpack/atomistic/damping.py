"""Damping-factor heads that read the softest local mode instead of summing over atoms.

:class:`~schnetpack.atomistic.DampingFactor` sums per-atom contributions. The damping it should
provide, though, is set by the smallest Hessian eigenvalue, and in a molecule with several soft
local modes (methyl rotors, amide groups, ...) that is a *minimum* over sites, which a sum cannot
form. The heads here replace the sum with a soft maximum,

.. math::
    \\mathrm{smax}_\\tau(x) = \\tau \\log \\sum_k w_k \\exp(x_k / \\tau),

over atoms (:class:`SoftMaxDampingFactor`), over atom pairs (:class:`PairDampingFactor`), or over
the negated eigenvalues of a predicted surrogate Hessian (:class:`SurrogateHessianDamping`). All
of them write one non-negative damping factor per molecule under ``output_key`` and so drop in for
:class:`~schnetpack.atomistic.DampingFactor`.
"""

import math
from typing import Callable, Dict, Optional, Sequence, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

import schnetpack.nn as snn
import schnetpack.properties as properties

__all__ = [
    "SoftMaxDampingFactor",
    "PairDampingFactor",
    "SurrogateHessianDamping",
    "segment_soft_max",
    "dense_from_pair_blocks",
    "surrogate_hessian",
]


def segment_soft_max(
    x: torch.Tensor,
    segment: torch.Tensor,
    n_segments: int,
    tau: torch.Tensor,
    weights: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Per-segment soft maximum ``tau * log(sum_k w_k exp(x_k / tau))``.

    It lies within ``tau * log(n)`` above the hard maximum (with unit weights), and is smooth in
    ``x``. Weights in ``[0, 1]`` let entries fade out smoothly, e.g. pairs approaching the cutoff.

    Args:
        x: Values, shape ``(n,)``.
        segment: Segment index of every value, shape ``(n,)``.
        n_segments: Number of segments.
        tau: Temperature, a positive scalar tensor.
        weights: Optional non-negative weights, shape ``(n,)``.

    Returns:
        One soft maximum per segment, shape ``(n_segments,)``.
    """
    z = x / tau
    # the shift only stabilises the exponentials; logsumexp's gradient does not depend on it
    shift = torch.zeros(n_segments, dtype=z.dtype, device=z.device).scatter_reduce(
        0, segment, z.detach(), reduce="amax", include_self=False
    )
    terms = torch.exp(z - shift[segment])
    if weights is not None:
        terms = terms * weights
    total = snn.scatter_add(terms, segment, dim_size=n_segments)
    # a segment whose weights are all zero (e.g. a lone atom without pairs) has no soft maximum;
    # the clamp gives it a very negative one, i.e. no damping, without a NaN gradient
    return tau * (torch.log(total.clamp_min(1e-30)) + shift)


class _SoftMax(nn.Module):
    """Learnable temperature and offset: ``softplus(offset + scale * smax_tau(x))``."""

    def __init__(self, output_scale: float, init_tau: float, init_offset: float):
        super().__init__()
        self.output_scale = output_scale
        self.log_tau = nn.Parameter(torch.tensor(math.log(init_tau)))
        self.offset = nn.Parameter(torch.tensor(float(init_offset)))

    @property
    def tau(self) -> torch.Tensor:
        return self.log_tau.exp()

    def damping(self, soft_max: torch.Tensor) -> torch.Tensor:
        return F.softplus(self.offset + self.output_scale * soft_max)


class SoftMaxDampingFactor(_SoftMax):
    """Per-atom scores, combined by a soft maximum over the atoms of each molecule.

    The per-atom MLP is the one :class:`~schnetpack.atomistic.Atomwise` builds; only the sum over
    atoms is replaced.
    """

    def __init__(
        self,
        n_in: int,
        n_hidden: Optional[Union[int, Sequence[int]]] = None,
        n_layers: int = 2,
        activation: Callable = F.silu,
        output_key: str = properties.damping_factor,
        output_scale: float = 10.0,
        init_tau: float = 0.5,
        init_offset: float = 0.0,
    ):
        """
        Args:
            n_in: Size of the scalar representation.
            n_hidden, n_layers, activation: Per-atom MLP, as in ``Atomwise``.
            output_key: Key of the damping factor.
            output_scale: Damping per unit of the (dimensionless) MLP score, in the model's
                energy/distance² unit. Sets the scale the head starts out at.
            init_tau: Initial temperature, in units of the MLP score.
            init_offset: Initial offset added before the softplus, in damping units. Start near
                the expected damping, so the head does not have to climb there by the MLP's
                last bias.
        """
        super().__init__(output_scale, init_tau, init_offset)
        self.output_key = output_key
        self.model_outputs = [output_key]
        self.outnet = snn.build_mlp(
            n_in=n_in,
            n_out=1,
            n_hidden=n_hidden,
            n_layers=n_layers,
            activation=activation,
        )

    def forward(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        scores = self.outnet(inputs["scalar_representation"]).squeeze(-1)
        n_mols = inputs[properties.n_atoms].shape[0]
        soft_max = segment_soft_max(scores, inputs[properties.idx_m], n_mols, self.tau)
        inputs[self.output_key] = self.damping(soft_max)
        return inputs


class _PairFeatures(nn.Module):
    """An i<->j symmetric MLP on ``[q_i + q_j, q_i * q_j, rbf(r_ij)]``, and the cutoff weights."""

    def __init__(
        self,
        n_in: int,
        n_out: int,
        n_rbf: int,
        cutoff: float,
        n_hidden: Optional[Union[int, Sequence[int]]],
        n_layers: int,
        activation: Callable,
    ):
        super().__init__()
        self.radial_basis = snn.GaussianRBF(n_rbf=n_rbf, cutoff=cutoff)
        self.cutoff_fn = snn.CosineCutoff(cutoff)
        self.net = snn.build_mlp(
            n_in=2 * n_in + n_rbf,
            n_out=n_out,
            n_hidden=n_hidden,
            n_layers=n_layers,
            activation=activation,
        )

    def forward(self, inputs: Dict[str, torch.Tensor]):
        q = inputs["scalar_representation"]
        idx_i, idx_j = inputs[properties.idx_i], inputs[properties.idx_j]
        r_ij = inputs[properties.Rij]
        d_ij = torch.norm(r_ij, dim=-1)
        q_i, q_j = q[idx_i], q[idx_j]
        features = torch.cat([q_i + q_j, q_i * q_j, self.radial_basis(d_ij)], dim=-1)
        return self.net(features), self.cutoff_fn(d_ij), r_ij, d_ij


class PairDampingFactor(_SoftMax):
    """Pair scores over the neighbor list, combined by a soft maximum over each molecule's pairs.

    Pairs are weighted by a cosine cutoff, so a pair leaving ``cutoff`` fades out of the soft
    maximum smoothly rather than dropping out of it.
    """

    def __init__(
        self,
        n_in: int,
        n_rbf: int = 20,
        cutoff: float = 5.0,
        n_hidden: Optional[Union[int, Sequence[int]]] = None,
        n_layers: int = 2,
        activation: Callable = F.silu,
        output_key: str = properties.damping_factor,
        output_scale: float = 10.0,
        init_tau: float = 0.5,
        init_offset: float = 0.0,
    ):
        """
        Args:
            n_in: Size of the scalar representation.
            n_rbf: Number of Gaussians expanding the pair distance.
            cutoff: Pair cutoff; at most the neighbor-list cutoff.
            n_hidden, n_layers, activation: Pair MLP.
            output_key, output_scale, init_tau, init_offset: As in :class:`SoftMaxDampingFactor`.
        """
        super().__init__(output_scale, init_tau, init_offset)
        self.output_key = output_key
        self.model_outputs = [output_key]
        self.pairs = _PairFeatures(
            n_in, 1, n_rbf, cutoff, n_hidden, n_layers, activation
        )

    def forward(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        scores, weights, _, _ = self.pairs(inputs)
        n_mols = inputs[properties.n_atoms].shape[0]
        pair_mol = inputs[properties.idx_m][inputs[properties.idx_i]]
        soft_max = segment_soft_max(
            scores.squeeze(-1), pair_mol, n_mols, self.tau, weights=weights
        )
        inputs[self.output_key] = self.damping(soft_max)
        return inputs


def _local_index(n_atoms: torch.Tensor, idx_m: torch.Tensor) -> torch.Tensor:
    """Index of every atom within its own molecule."""
    first = torch.cumsum(n_atoms, 0) - n_atoms
    return torch.arange(idx_m.shape[0], device=idx_m.device) - first[idx_m]


def dense_from_pair_blocks(
    blocks: torch.Tensor,
    idx_i: torch.Tensor,
    idx_j: torch.Tensor,
    idx_m: torch.Tensor,
    n_atoms: torch.Tensor,
) -> torch.Tensor:
    """Assemble per-molecule dense ``(3N, 3N)`` matrices from 3x3 pair blocks B_ij.

    Every directed pair (i, j) adds ``-B_ij`` to block (i, j) and ``+B_ij`` to block (i, i), the
    acoustic sum rule: each block row sums to zero, so a rigid translation is an exact zero mode,
    as for a true Hessian. With a symmetric B_ij = B_ji (both directions present in the neighbor
    list) the result is symmetric.

    Molecules smaller than the largest one are zero-padded; the caller decides what the padding
    should look like to an eigensolver.

    Returns:
        Shape ``(n_molecules, 3 N_max, 3 N_max)``.
    """
    n_mols = n_atoms.shape[0]
    n3 = 3 * int(n_atoms.max())
    local = _local_index(n_atoms, idx_m)
    mol = idx_m[idx_i]
    comp = torch.arange(3, device=blocks.device)
    row = 3 * local[idx_i][:, None, None] + comp[None, :, None]  # (n_pairs, 3, 1)
    col_j = 3 * local[idx_j][:, None, None] + comp[None, None, :]  # (n_pairs, 1, 3)
    col_i = 3 * local[idx_i][:, None, None] + comp[None, None, :]
    base = (mol * n3 * n3)[:, None, None]
    flat = torch.cat(
        [(base + row * n3 + col_j).reshape(-1), (base + row * n3 + col_i).reshape(-1)]
    )
    values = torch.cat([-blocks.reshape(-1), blocks.reshape(-1)])
    dense = torch.zeros(n_mols * n3 * n3, dtype=blocks.dtype, device=blocks.device)
    return dense.index_add(0, flat, values).reshape(n_mols, n3, n3)


def surrogate_hessian(
    alpha: torch.Tensor,
    beta: torch.Tensor,
    gamma: torch.Tensor,
    w: torch.Tensor,
    r_ij: torch.Tensor,
    d_ij: torch.Tensor,
    idx_i: torch.Tensor,
    idx_j: torch.Tensor,
    idx_m: torch.Tensor,
    n_atoms: torch.Tensor,
) -> torch.Tensor:
    """Dense surrogate Hessian from pair blocks ``alpha I + beta r̂r̂ᵀ + Σ_k gamma_k sym(w_i^k w_j^kᵀ)``.

    Args:
        alpha, beta: Pair scalars, shape ``(n_pairs,)``.
        gamma: Pair scalars per vector channel, shape ``(n_pairs, K)``.
        w: Equivariant per-atom vectors, shape ``(n_atoms, 3, K)``.
        r_ij, d_ij: Pair vectors and distances.
        idx_i, idx_j, idx_m, n_atoms: Neighbor list and molecule bookkeeping.
    """
    r_hat = r_ij / d_ij[:, None]
    eye = torch.eye(3, dtype=alpha.dtype, device=alpha.device)
    w_i, w_j = w[idx_i], w[idx_j]  # (n_pairs, 3, K)
    outer = torch.einsum("pak,pbk,pk->pab", w_i, w_j, gamma)
    blocks = (
        alpha[:, None, None] * eye
        + beta[:, None, None] * r_hat[:, :, None] * r_hat[:, None, :]
        + 0.5 * (outer + outer.transpose(1, 2))
    )
    return dense_from_pair_blocks(blocks, idx_i, idx_j, idx_m, n_atoms)


class SurrogateHessianDamping(nn.Module):
    """Predicts a surrogate Hessian Ĥ and damps by a soft maximum of its negated eigenvalues.

    Ĥ is built from 3x3 pair blocks (see :func:`surrogate_hessian`): pair scalars from an
    i<->j symmetric MLP, weighted by a cosine cutoff, and K equivariant vector channels mixed out
    of the representation's vector features. It is symmetric, rotation-equivariant (so its
    eigenvalues are invariant) and has rigid translations as exact zero modes. The damping is

    .. math::
        \\lambda = \\mathrm{softplus}(m + \\tau \\log \\sum_k \\exp(-e_k / \\tau)),

    i.e. about ``m - e_min(Ĥ)`` once Ĥ has a clearly softest mode.

    Without further supervision Ĥ is a latent that only learns through λ. Set
    ``surrogate_hessian_key`` on :class:`~schnetpack.train.NewtonSurrogateTask` to also anchor it
    to the reference potential's Hessian through Hessian-vector products.
    """

    def __init__(
        self,
        n_in: int,
        n_vector_channels: int = 8,
        n_rbf: int = 20,
        cutoff: float = 5.0,
        n_hidden: Optional[Union[int, Sequence[int]]] = None,
        n_layers: int = 2,
        activation: Callable = F.silu,
        output_key: str = properties.damping_factor,
        hessian_key: Optional[str] = "surrogate_hessian",
        output_scale: float = 10.0,
        init_tau: float = 5.0,
        init_offset: float = 0.0,
    ):
        """
        Args:
            n_in: Size of the scalar and vector representation.
            n_vector_channels: Number K of equivariant vector channels in the pair blocks.
            n_rbf, cutoff: Pair distance expansion and cutoff.
            n_hidden, n_layers, activation: Pair MLP.
            output_key: Key of the damping factor.
            hessian_key: If not None, Ĥ is also stored under this key, shape
                ``(n_molecules, 3 N_max, 3 N_max)``.
            output_scale: Scale of the pair blocks, in the model's energy/distance² unit.
            init_tau: Initial temperature, in energy/distance² (Ĥ's unit).
            init_offset: Initial offset m, in energy/distance².
        """
        super().__init__()
        self.output_key = output_key
        self.hessian_key = hessian_key
        self.model_outputs = [output_key] + ([hessian_key] if hessian_key else [])
        self.output_scale = output_scale
        self.n_vector_channels = n_vector_channels
        self.pairs = _PairFeatures(
            n_in, 2 + n_vector_channels, n_rbf, cutoff, n_hidden, n_layers, activation
        )
        self.mix_vectors = nn.Linear(n_in, n_vector_channels, bias=False)
        self.log_tau = nn.Parameter(torch.tensor(math.log(init_tau)))
        self.offset = nn.Parameter(torch.tensor(float(init_offset)))

    @property
    def tau(self) -> torch.Tensor:
        return self.log_tau.exp()

    def hessian(self, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """The zero-padded surrogate Hessian, shape ``(n_molecules, 3 N_max, 3 N_max)``."""
        scalars, weights, r_ij, d_ij = self.pairs(inputs)
        scalars = self.output_scale * scalars * weights[:, None]
        w = self.mix_vectors(inputs["vector_representation"])  # (n_atoms, 3, K)
        return surrogate_hessian(
            scalars[:, 0],
            scalars[:, 1],
            scalars[:, 2:],
            w,
            r_ij,
            d_ij,
            inputs[properties.idx_i],
            inputs[properties.idx_j],
            inputs[properties.idx_m],
            inputs[properties.n_atoms],
        )

    def forward(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        H = self.hessian(inputs)
        n_atoms = inputs[properties.n_atoms]
        n3 = H.shape[-1]
        padded = torch.arange(n3, device=H.device)[None, :] >= 3 * n_atoms[:, None]
        # padding gets eigenvalues above all real ones (Gershgorin bound), so they are the last
        # ones of the ascending spectrum and can be masked out of the soft maximum exactly
        bound = H.detach().abs().sum(-1).amax(-1, keepdim=True) + 1.0
        e = torch.linalg.eigvalsh(H + torch.diag_embed(padded * bound))
        neg_e = torch.where(padded, torch.full_like(e, -torch.inf), -e)
        soft_max = self.tau * torch.logsumexp(neg_e / self.tau, dim=-1)
        inputs[self.output_key] = F.softplus(self.offset + soft_max)
        if self.hessian_key:
            inputs[self.hessian_key] = H
        return inputs
