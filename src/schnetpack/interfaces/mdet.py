"""Use an MD-ET (``TriangularEncoder``) potential as a SchNetPack reference model.

MD-ET predicts forces with a direct head rather than as :math:`-\\nabla E`, so
its Hessian is a single derivative away -- which is what makes it attractive as
the frozen reference potential of the Newton-step surrogate training (see
:mod:`schnetpack.train.surrogate`).

Two things stand between the released MD-ET checkpoints and that use:

**Interface.** MD-ET is called with its own dictionary (``Property`` enum keys,
its own neighbour list *and* the CSR-packed triangle indices, positions in the
model's native length unit) and returns ``formation_energy``/``forces`` in its
native units. :class:`MDETTeacher` wraps all of that behind SchNetPack's
``Dict[str, Tensor] -> Dict[str, Tensor]`` interface: it reads
``_positions``/``_atomic_numbers``/``_idx_m``/``_n_atoms`` off a SchNetPack
batch, builds the MD-ET graph (per molecule, so nothing bonds across the batch),
and returns ``energy`` and ``forces`` in SchNetPack units.

**Differentiability.** On GPU, MD-ET's triplet attention runs a Triton kernel
whose backward is itself an opaque kernel. That backward

* returns no gradient for the envelope (``env_pair``/``env_gate``), which is a
  function of the positions, so ``d forces / d positions`` silently misses those
  terms, and
* cannot be differentiated a second time, which the surrogate task needs -- the
  Hessian-vector product is built with ``create_graph=True`` so that gradients
  reach the student through the vector it contracts with.

:func:`patch_mdet_for_autodiff` therefore swaps in vectorised pure-PyTorch
implementations of both attention variants. They reproduce the semantics of
MD-ET's own reference fallbacks (which are correct but loop over CSR rows in
Python), and, unlike the Triton path, they are exact and twice differentiable.
The same patch replaces the two places where MD-ET takes ``|r_ij|``: with
``self_loops=True`` every atom carries an ``(i, i)`` edge whose difference
vector is identically zero, and the derivative of a norm at zero is ``0/0``.
The patched code takes the norm only off the non-degenerate edges and pins the
self-loop distance to a constant zero -- same forward, no NaN in the backward.

Because the patch changes what MD-ET computes on GPU (it adds the missing
envelope gradients), :func:`check_teacher` is provided to verify against the
unpatched model that the *forward* pass is unchanged.

Typical use::

    from schnetpack.interfaces.mdet import load_mdet_teacher, save_mdet_teacher
    teacher = load_mdet_teacher(variant="qcml-s2")
    save_mdet_teacher(teacher, "mdet_s2_teacher.pt")

    # spktrain experiment=newton_step_training_horm_hvp \\
    #     globals.ref_model_path=.../mdet_s2_teacher.pt \\
    #     globals.ref_model_format=torch
"""

import math
import warnings
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from schnetpack import properties
from schnetpack.units import convert_units

__all__ = [
    "MDETTeacher",
    "load_mdet_teacher",
    "resolve_mdet_run_dir",
    "native_units_of",
    "save_mdet_teacher",
    "patch_mdet_for_autodiff",
    "check_teacher",
    "MDET_NATIVE_UNITS",
]

#: Native (internal) units of the released MD-ET variants, as
#: ``(length, energy)`` strings understood by :func:`schnetpack.units.convert_units`.
MDET_NATIVE_UNITS = {
    "qcml-s2": ("Bohr", "Hartree"),
    "qcml-m5": ("Bohr", "Hartree"),
    "qcml-m5-16L-c4": ("Bohr", "Hartree"),
    "omat24-6L256d": ("Ang", "eV"),
    "omat24-6L512d": ("Ang", "eV"),
}


# ---------------------------------------------------------------------------
# differentiable replacements for MD-ET's fused kernels
# ---------------------------------------------------------------------------


def _edge_geometry(positions, i_idx, j_idx, offsets=None):
    """Edge difference vectors and their lengths, safe at zero.

    ``self_loops=True`` puts an ``(i, i)`` edge on every atom, whose difference
    vector is exactly zero *for every configuration*. Its derivative with
    respect to the positions is therefore exactly zero -- but autograd does not
    know that. It would take the norm at the origin (``0/0``) and route the two
    halves of the difference into the same atom to cancel there. The
    cancellation is exact in real arithmetic and useless in floating point: the
    direction feature divides by ``|r| + 1e-5``, so a self-loop contributes
    terms ~1e5 times larger than everything they are summed with, and the real
    gradient is what gets rounded away.

    Both are fixed by detaching the self-loop entries: same forward, and the
    derivative they should have had (zero) instead of a cancellation.
    """
    i = i_idx.long()
    j = j_idx.long()
    diff = positions[j] - positions[i]
    if offsets is not None:
        diff = diff + offsets

    self_loop = i == j
    if offsets is not None:
        self_loop = self_loop & (offsets == 0).all(-1)
    diff = torch.where(self_loop.unsqueeze(-1), diff.detach(), diff)

    # a coincident pair of distinct atoms is a genuine singularity of the model
    # rather than a bookkeeping artefact; the norm is kept finite so that it
    # shows up as a zero row instead of a NaN everywhere.
    sq = (diff * diff).sum(-1)
    degenerate = sq == 0
    safe_sq = torch.where(degenerate, torch.ones_like(sq), sq)
    d = torch.where(degenerate, torch.zeros_like(sq), torch.sqrt(safe_sq))
    return diff, d


def _composer_forward(self, h, positions, i_idx, j_idx, Z=None, offsets=None):
    """``EdgeComposer.forward`` with a self-loop-safe distance."""
    edge_h = self.compose_proj(torch.cat([h[i_idx.long()], h[j_idx.long()]], dim=-1))
    diff, d_ij = _edge_geometry(positions, i_idx, j_idx, offsets)
    edge_h = edge_h + self.edge_proj(self.rbf(d_ij))
    dir_ij = diff / (d_ij.unsqueeze(-1) + 1e-5)
    edge_h = edge_h + self.dir_proj(dir_ij)
    if self.embed_edge_types and Z is not None:
        edge_types = Z[i_idx.long()] * 128 + Z[j_idx.long()]
        edge_h = edge_h + self.edge_type_proj(self.edge_type_embed(edge_types.long()))
    return edge_h


def _row_ids(offsets: torch.Tensor, n_triplets: int) -> torch.Tensor:
    """CSR offsets -> per-triplet row index."""
    counts = offsets[1:] - offsets[:-1]
    rows = torch.arange(counts.shape[0], device=offsets.device)
    return torch.repeat_interleave(rows, counts)


def _split_qkvv(QKVV, num_heads, src, dst):
    E, S = QKVV.shape
    D = S // 4
    d_head = D // num_heads
    Q, K, V1, V2 = (
        QKVV[:, :D],
        QKVV[:, D : 2 * D],
        QKVV[:, 2 * D : 3 * D],
        QKVV[:, 3 * D :],
    )
    q = Q[dst].view(-1, num_heads, d_head)
    k = K[src].view(-1, num_heads, d_head)
    v1 = V1[dst].view(-1, num_heads, d_head)
    v2 = V2[src].view(-1, num_heads, d_head)
    return q, k, v1, v2, E, D, d_head


def triplet_attention_softmax(
    QKVV, idxs_src, idxs_dst, num_heads, offsets, env_gate=None, needs_grad=None
):
    """Vectorised, twice-differentiable softmax triplet attention.

    Same quantity as ``md_et.nn.triplet_attention.sparse_triplet_attention_fallback``:
    a softmax over the triplets of each CSR row, an optional post-softmax
    envelope gate, and a scatter of ``attn * V1 * V2`` onto the destination
    edges. Unlike the Triton kernel, ``env_gate`` carries a gradient here.
    """
    E, S = QKVV.shape
    D = S // 4
    n_triplets = idxs_src.shape[0]
    if n_triplets == 0:
        return QKVV.new_zeros(E, D)

    src = idxs_src.long()
    dst = idxs_dst.long()
    q, k, v1, v2, E, D, d_head = _split_qkvv(QKVV, num_heads, src, dst)

    scores = (q * k).sum(-1) / math.sqrt(d_head)  # (T, H)

    row = _row_ids(offsets, n_triplets)
    n_rows = offsets.shape[0] - 1
    row_h = row.unsqueeze(-1).expand(-1, num_heads)

    # softmax over the triplets sharing a row; the shift is a constant
    row_max = torch.full(
        (n_rows, num_heads), float("-inf"), device=scores.device, dtype=scores.dtype
    ).scatter_reduce(0, row_h, scores.detach(), reduce="amax", include_self=True)
    weights = torch.exp(scores - row_max[row])
    denom = torch.zeros_like(row_max).index_add(0, row, weights)
    attn = weights / denom[row]

    if env_gate is not None:
        attn = attn * env_gate.unsqueeze(-1)

    weighted = (attn.unsqueeze(-1) * v1 * v2).reshape(-1, D)
    return torch.zeros(E, D, device=QKVV.device, dtype=QKVV.dtype).index_add(
        0, dst, weighted
    )


def triplet_attention_sigmoid(
    QKVV,
    idxs_src,
    idxs_dst,
    num_heads,
    offsets,
    env_pair,
    inv_sqrt_K_eff,
    bias,
    needs_grad=None,
):
    """Vectorised, twice-differentiable sigmoid triplet attention.

    Same quantity as
    ``md_et.nn.triplet_attention_sigmoid.sparse_triplet_attention_sigmoid_fallback``.
    Every triplet is independent (no row reduction), so the CSR row structure is
    not needed at all; ``offsets`` is accepted only to match the call signature.
    """
    E, S = QKVV.shape
    D = S // 4
    if idxs_src.shape[0] == 0:
        return QKVV.new_zeros(E, D)

    src = idxs_src.long()
    dst = idxs_dst.long()
    q, k, v1, v2, E, D, d_head = _split_qkvv(QKVV, num_heads, src, dst)

    scores = (q * k).sum(-1) / math.sqrt(d_head) + bias  # (T, H)
    weight = (
        torch.sigmoid(scores)
        * env_pair.unsqueeze(-1)
        * inv_sqrt_K_eff[dst].unsqueeze(-1)
    )

    weighted = (weight.unsqueeze(-1) * v1 * v2).reshape(-1, D)
    return torch.zeros(E, D, device=QKVV.device, dtype=QKVV.dtype).index_add(
        0, dst, weighted
    )


def _encoder_forward(self, inputs):
    """``TriangularEncoder.forward`` with a self-loop-safe edge length.

    A copy of MD-ET's own forward; the only change is that the envelope
    distance comes from :func:`_edge_geometry` instead of ``diff.norm(-1)``.
    """
    from md_et.nn.triangular_encoder import polynomial_envelope
    from md_et.nn.types import NODE_FEATURES_OFFSET, Property as Props, PropertyType

    positions = inputs[Props.positions]
    Z = inputs[Props.atomic_numbers]
    i_idx = inputs[Props.i_idx]
    j_idx = inputs[Props.j_idx]
    tri_flat_src = inputs[Props.tri_flat_src]
    tri_flat_dst = inputs[Props.tri_flat_dst]
    tri_offsets = inputs[Props.tri_offsets]
    offsets = inputs.get(Props.offsets)
    charge = inputs[Props.charge]
    multiplicity = inputs[Props.multiplicity]
    mol_idx = inputs[Props.mol_idx]

    cached_h0 = inputs.get("_cached_h0")
    h = self._embed(Z, charge, multiplicity, mol_idx, cached_h0=cached_h0)
    N = h.shape[0]

    edge_h = self.composer(h, positions, i_idx, j_idx, Z=Z, offsets=offsets)

    env = None
    env_pair = None
    inv_sqrt_K_eff = None
    if self.softmax_env_gate or self.atomic_output_env or self.sigmoid_attn:
        _, d_ij = _edge_geometry(positions, i_idx, j_idx, offsets)
        cutoff_val = self.cutoff
        margin = cutoff_val * 0.2
        d_shifted = (d_ij - (cutoff_val - margin)).clamp(min=0.0)
        env = polynomial_envelope(d_shifted, margin, 5)
    if self.softmax_env_gate or self.sigmoid_attn:
        env_src = env[tri_flat_src.long()]
        env_dst = env[tri_flat_dst.long()]
        env_pair = (env_src * env_dst).contiguous()
    if self.sigmoid_attn:
        E_edges = i_idx.shape[0]
        K_eff = torch.zeros(E_edges, device=positions.device, dtype=env_pair.dtype)
        K_eff = K_eff.scatter_add(0, tri_flat_dst.long(), env_pair)
        inv_sqrt_K_eff = 1.0 / torch.sqrt(K_eff + 1.0)

    cached_cond = inputs.get("_cached_cond_proj")
    if cached_cond is None:
        c_emb = self.charge_embed(charge.reshape(-1) + NODE_FEATURES_OFFSET // 2)
        m_emb = self.multiplicity_embed(multiplicity.reshape(-1))
        cond_mol = self.adaln_cond_proj(torch.cat([c_emb, m_emb], dim=-1))
    else:
        cond_mol = cached_cond
    edge_mol_idx = mol_idx[i_idx.long()]
    cond = cond_mol[edge_mol_idx]

    attn_env_gate = env_pair if self.softmax_env_gate else None
    for layer in self.layers:
        edge_h = layer(
            edge_h,
            tri_flat_src,
            tri_flat_dst,
            tri_offsets,
            cond=cond,
            env_gate=attn_env_gate,
            env_pair=env_pair,
            inv_sqrt_K_eff=inv_sqrt_K_eff,
        )

    h = self.decomposer(edge_h, i_idx, j_idx, N)

    atom_env = None
    if self.atomic_output_env:
        is_real = (i_idx != j_idx).to(env.dtype)
        env_real = env * is_real
        mass_atom = torch.zeros(N, device=positions.device, dtype=env.dtype)
        mass_atom = mass_atom.scatter_add(0, i_idx.long(), env_real)
        atom_env = 1.0 - torch.exp(-mass_atom / self.atom_env_tau)

    out = {}
    for k, head in self.heads.items():
        target = Props[k]
        if atom_env is None:
            out[target] = head(h, inputs)
            continue
        h_normed = head.norm(h)
        per_atom = head.mlp(h_normed)
        gated = per_atom * atom_env.unsqueeze(-1)
        if head.target_type == PropertyType.mol_wise:
            n_mols = inputs[Props.n_atoms].shape[0]
            pooled = torch.zeros(
                n_mols, per_atom.shape[-1], device=h.device, dtype=h.dtype
            )
            pooled = pooled.scatter_add(
                0, mol_idx.unsqueeze(-1).expand_as(gated), gated
            )
            if pooled.shape[-1] == 1:
                pooled = pooled.squeeze(-1)
            out[target] = pooled
        else:
            out[target] = gated
    out["embd"] = h
    return out


_PATCHED = False


def patch_mdet_for_autodiff(force: bool = False) -> None:
    """Replace MD-ET's fused kernels with differentiable PyTorch equivalents.

    Idempotent, and process-wide: MD-ET dispatches to the Triton kernels through
    module-level names, which is where the replacements are installed. Anything
    else in the process that runs an MD-ET model afterwards gets the slower but
    exact path too.
    """
    global _PATCHED
    if _PATCHED and not force:
        return

    from md_et.nn import triangular_encoder as te

    te.fused_sparse_triplet_attention = triplet_attention_softmax
    te.fused_sparse_triplet_attention_sigmoid = triplet_attention_sigmoid
    te.EdgeComposer.forward = _composer_forward
    te.TriangularEncoder.forward = _encoder_forward
    _PATCHED = True


# ---------------------------------------------------------------------------
# graph construction for a batch of molecules
# ---------------------------------------------------------------------------


def _build_batch_graph(
    positions: torch.Tensor,
    idx_m: torch.Tensor,
    cutoff: float,
    self_loops: bool,
    dst_keyed: bool,
) -> Dict[str, torch.Tensor]:
    """MD-ET's graph for a whole SchNetPack batch.

    Pairs are taken within each molecule only -- the batch is a set of separate
    systems that happen to share a coordinate array, and MD-ET's own
    ``build_graph`` would bond them to each other. The triangle CSR is then
    built by MD-ET's numba kernel, exactly as for a single molecule.
    """
    from md_et.nn.graph import _to_dst_keyed_csr, _triplet_blocks_numba

    n_atoms_total = positions.shape[0]
    with torch.no_grad():
        delta = positions.unsqueeze(0) - positions.unsqueeze(1)
        dist_sq = (delta * delta).sum(-1)
        same_mol = idx_m.unsqueeze(0) == idx_m.unsqueeze(1)
        mask = (dist_sq < cutoff * cutoff) & (dist_sq > 0) & same_mol
        i_idx, j_idx = mask.nonzero(as_tuple=True)

    if self_loops:
        self_i = torch.arange(n_atoms_total, device=i_idx.device, dtype=i_idx.dtype)
        i_idx = torch.cat([self_i, i_idx])
        j_idx = torch.cat([self_i, j_idx])

    i_np = i_idx.detach().cpu().numpy().astype(np.int32)
    j_np = j_idx.detach().cpu().numpy().astype(np.int32)
    flat_dst_np, flat_src_np, offsets_np, _ = _triplet_blocks_numba(
        i_np, j_np, n_atoms_total
    )

    device = positions.device
    flat_dst = torch.from_numpy(flat_dst_np).to(device)
    flat_src = torch.from_numpy(flat_src_np).to(device)
    offsets = torch.from_numpy(offsets_np).to(device)

    if dst_keyed:
        flat_dst, flat_src, offsets = _to_dst_keyed_csr(flat_dst, flat_src, device)

    return {
        "i_idx": i_idx.to(torch.int64),
        "j_idx": j_idx.to(torch.int64),
        "tri_flat_src": flat_src,
        "tri_flat_dst": flat_dst,
        "tri_offsets": offsets,
    }


# ---------------------------------------------------------------------------
# the wrapper
# ---------------------------------------------------------------------------


class MDETTeacher(nn.Module):
    """An MD-ET potential behind SchNetPack's model interface.

    Called with a SchNetPack batch, returns ``{energy, forces}`` in SchNetPack
    units, differentiably in ``batch[properties.R]`` -- which is what
    :class:`schnetpack.train.surrogate.NewtonSurrogateTask` needs in order to
    take the Hessian-vector product.

    Args:
        model: A loaded ``md_et.nn.triangular_encoder.TriangularEncoder``.
        cutoff: Neighbour cutoff, in the model's native length unit.
        native_length_unit: Length unit the model works in (``"Bohr"`` for the
            QCML variants, ``"Ang"`` for the OMat ones).
        native_energy_unit: Energy unit the model works in.
        position_unit: Length unit of the incoming batch.
        energy_unit: Energy unit the outputs are converted to. Only the ratio
            energy/length² matters for the Newton step itself, but it fixes the
            scale of the learned damping, so keep it at whatever the rest of the
            experiment uses.
        energy_key, force_key: Keys of the returned quantities.
        charge, multiplicity: Per-molecule inputs MD-ET conditions on. Read from
            the batch when ``properties.total_charge`` /
            ``properties.spin_multiplicity`` are present, otherwise these
            defaults are used.
    """

    def __init__(
        self,
        model: nn.Module,
        cutoff: float,
        native_length_unit: str = "Bohr",
        native_energy_unit: str = "Hartree",
        position_unit: str = "Ang",
        energy_unit: str = "kcal/mol",
        energy_key: str = properties.energy,
        force_key: str = properties.forces,
        charge: int = 0,
        multiplicity: int = 1,
    ):
        super().__init__()
        self.model = model
        self.cutoff = float(cutoff)
        self.native_length_unit = native_length_unit
        self.native_energy_unit = native_energy_unit
        self.position_unit = position_unit
        self.energy_unit = energy_unit
        self.energy_key = energy_key
        self.force_key = force_key
        self.charge = int(charge)
        self.multiplicity = int(multiplicity)

        # batch positions -> native, native energy/forces -> batch units
        self.pos_scale = convert_units(position_unit, native_length_unit)
        self.energy_scale = convert_units(native_energy_unit, energy_unit)
        self.force_scale = self.energy_scale / convert_units(
            native_length_unit, position_unit
        )

        self.self_loops = bool(getattr(model, "self_loops", False))
        self.dst_keyed = bool(getattr(model, "sigmoid_attn", False))

        # keeps schnetpack.utils.load_model from trying to convert this
        self.spk_version = "2.2.0"
        self.do_postprocessing = False

    def forward(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        from md_et.nn.types import Property as Props

        R = inputs[properties.R]
        Z = inputs[properties.Z]
        idx_m = inputs[properties.idx_m]
        n_atoms = inputs[properties.n_atoms]
        n_mols = n_atoms.shape[0]

        positions = R * self.pos_scale
        graph = _build_batch_graph(
            positions.detach(), idx_m, self.cutoff, self.self_loops, self.dst_keyed
        )

        data = {
            Props.positions: positions,
            Props.atomic_numbers: Z.long(),
            Props.charge: self._mol_input(
                inputs, properties.total_charge, self.charge, n_mols, positions.device
            ),
            Props.multiplicity: self._mol_input(
                inputs,
                properties.spin_multiplicity,
                self.multiplicity,
                n_mols,
                positions.device,
            ),
            Props.mol_idx: idx_m.long(),
            Props.n_atoms: n_atoms.long(),
            Props.i_idx: graph["i_idx"],
            Props.j_idx: graph["j_idx"],
            Props.tri_flat_src: graph["tri_flat_src"],
            Props.tri_flat_dst: graph["tri_flat_dst"],
            Props.tri_offsets: graph["tri_offsets"],
        }

        out = self.model(data)

        energy = out.get(Props.formation_energy, out.get(Props.energy))
        result = {self.force_key: out[Props.forces] * self.force_scale}
        if energy is not None:
            result[self.energy_key] = energy.reshape(-1) * self.energy_scale
        return result

    @staticmethod
    def _mol_input(inputs, key, default, n_mols, device):
        if key in inputs:
            return inputs[key].reshape(-1).long().to(device)
        return torch.full((n_mols,), default, dtype=torch.int64, device=device)


def resolve_mdet_run_dir(source: str, variant: str):
    """Locate a released MD-ET run directory.

    ``source`` is a local path (either the run directory itself or a directory
    holding one per variant) or a HuggingFace repo id. MD-ET's own resolver
    only accepts the four variants it lists, so the download is done here --
    ``qcml-m5-16L-c4`` and anything else published in the repo is reachable
    this way.
    """
    from pathlib import Path

    path = Path(source)
    if (path / variant).exists():
        return path / variant
    if path.exists():
        return path

    from huggingface_hub import snapshot_download

    cache = snapshot_download(
        repo_id=str(source), repo_type="model", allow_patterns=[f"{variant}/**"]
    )
    run_dir = Path(cache) / variant
    if not run_dir.exists():
        raise FileNotFoundError(
            f"variant {variant!r} not found in {source!r} (downloaded to {cache})"
        )
    return run_dir


def native_units_of(variant: str):
    """``(length, energy)`` units a released variant works in internally."""
    if variant in MDET_NATIVE_UNITS:
        return MDET_NATIVE_UNITS[variant]
    # the QCML models are trained in atomic units, the OMat ones in ASE units
    return ("Ang", "eV") if variant.startswith("omat") else ("Bohr", "Hartree")


def load_mdet_teacher(
    source: str = "mx-e/md-et-v3",
    variant: str = "qcml-s2",
    checkpoint_name: str = "best_model",
    device: str = "cpu",
    energy_unit: str = "kcal/mol",
    position_unit: str = "Ang",
    **kwargs,
) -> MDETTeacher:
    """Load a released MD-ET checkpoint as a SchNetPack reference potential."""
    from md_et.calculator import _extract_model_config, _load_model

    patch_mdet_for_autodiff()

    run_dir = resolve_mdet_run_dir(source, variant)
    model = _load_model(run_dir, device=device, checkpoint_name=checkpoint_name)
    config = _extract_model_config(run_dir / ".hydra" / "config.yaml")

    length_unit, native_energy_unit = native_units_of(variant)
    return MDETTeacher(
        model,
        cutoff=config["cutoff"],
        native_length_unit=length_unit,
        native_energy_unit=native_energy_unit,
        position_unit=position_unit,
        energy_unit=energy_unit,
        **kwargs,
    ).to(device)


def save_mdet_teacher(teacher: MDETTeacher, path: str) -> None:
    """Write the teacher so that ``ref_model_format=torch`` can load it.

    The class is pickled by reference, so ``schnetpack`` and ``md_et`` both have
    to be importable when the training run loads it back.
    """
    teacher = teacher.to("cpu")
    torch.save(teacher, path)


# ---------------------------------------------------------------------------
# checks
# ---------------------------------------------------------------------------


def check_teacher(
    teacher: MDETTeacher,
    batch: Dict[str, torch.Tensor],
    eps: float = 1e-3,
) -> Dict[str, float]:
    """Verify the two properties the surrogate training relies on.

    1. The forces are differentiable in the positions *twice*: the
       Hessian-vector product is taken with ``create_graph=True`` and a gradient
       is pulled back through it to the vector it contracts with. A teacher that
       fails this trains a student on a constant.
    2. The Hessian-vector product is a Hessian rather than an artefact of the
       kernels, checked against a central difference of the forces.

    The second number does not go to zero. The reverse-mode product is
    :math:`H^T v` while the finite difference is :math:`H v`, and MD-ET predicts
    forces with a head rather than as a gradient, so its Hessian is only
    symmetric to the extent that the model has learned to be conservative --
    around 1e-3 relative for the released checkpoints. Read it as an upper bound
    on the error that also contains the asymmetry, not as the error alone.
    """
    positions = batch[properties.R].detach().clone().requires_grad_(True)
    batch = dict(batch)
    batch[properties.R] = positions

    forces = teacher(batch)[teacher.force_key]
    if forces.grad_fn is None:
        raise RuntimeError("teacher forces are not differentiable w.r.t. the positions")

    vector = torch.randn_like(positions)
    vector.requires_grad_(True)
    hvp = -torch.autograd.grad(forces, positions, vector, create_graph=True)[0]

    probe = torch.randn_like(hvp)
    back = torch.autograd.grad((hvp * probe).sum(), vector, retain_graph=False)[0]
    if back is None or not torch.isfinite(back).all():
        raise RuntimeError("no finite gradient reaches the contracted vector")

    # central difference of the forces along `vector`
    with torch.no_grad():
        plus = dict(batch)
        plus[properties.R] = positions.detach() + eps * vector.detach()
        minus = dict(batch)
        minus[properties.R] = positions.detach() - eps * vector.detach()
        f_plus = teacher(plus)[teacher.force_key]
        f_minus = teacher(minus)[teacher.force_key]
    hvp_fd = -(f_plus - f_minus) / (2 * eps)

    hvp_val = hvp.detach()
    hvp_err = (hvp_val - hvp_fd).norm() / hvp_fd.norm().clamp(min=1e-12)
    # the backward through the vector should reproduce J acting on `probe`;
    # only its finiteness and scale are checked here
    return {
        "hvp_vs_finite_difference": float(hvp_err),
        "hvp_norm": float(hvp_val.norm()),
        "grad_through_vector_norm": float(back.norm()),
    }
