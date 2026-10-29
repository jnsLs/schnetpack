"""Smallest Hessian eigenvalue per molecule from Hessian-vector products.

Eigenvalue damping, :math:`\\lambda = -e_{\\min} + \\varepsilon`, needs the
smallest eigenvalue of every molecule's Hessian. This module gets it without
forming the Hessian, by running the Lanczos method on a function that returns
Hessian-vector products. The function covers a whole SchNetPack batch at once,
which works because the batch Hessian is block-diagonal: molecules do not
interact, so one product with a stacked vector of shape ``(n_atoms_total, 3)``
gives every molecule's product at the price of one backward pass. Every inner
product and norm is taken per molecule (an ``index_add`` over ``idx_m``), so the
molecules run as independent Lanczos processes that share their matvecs.

Implementation choices
----------------------
Full reorthogonalization
    Every new Lanczos vector is orthogonalized against all previous ones (two
    Gram-Schmidt passes, in float64). For the step counts that make sense here
    (k <= ~50) this is cheap next to a single matvec, and it removes the ghost
    eigenvalues of plain three-term Lanczos.

Rayleigh-Ritz on the stored products, symmetrized
    Instead of the three-term coefficients the Ritz matrix is built as
    :math:`T = Q^T (A Q)` from the stored products and then symmetrized,
    :math:`\\tfrac12 (T + T^T) = Q^T \\tfrac12 (A + A^T) Q`. For a symmetric
    operator this is the usual tridiagonal Lanczos matrix (to rounding). For a
    slightly asymmetric one -- see the next point -- it gives Ritz values of the
    *symmetric part* of the operator, on the Krylov space of the operator
    itself, at no extra cost.

Reverse mode gives :math:`H^T v`
    A Hessian-vector product taken as ``-grad(forces, positions, v)`` is a
    vector-Jacobian product, i.e. :math:`-J^T v` with :math:`J = \\partial F /
    \\partial R`. For a conservative potential (forces = -grad E) :math:`J` is
    symmetric and this is exactly :math:`H v`. For direct-force models such as
    MD-ET it is :math:`H^T v`, with :math:`\\lVert H - H^T \\rVert / (2\\lVert H
    \\rVert)` about 0.4-0.7%. Because of the symmetrized Rayleigh-Ritz step, the
    result is a Ritz value of :math:`\\tfrac12 (H + H^T)` either way, which is the
    matrix the offline pipeline (``make_newton_dataset.py``) took ``eigvalsh`` of.
    A forward-mode :math:`H v` would allow an exactly symmetrized operator,
    :math:`\\tfrac12 (H v + H^T v)`, but costs a second kind of derivative per step.

Projecting out translations
    With ``project_translations=True`` the three rigid translations are removed
    from the start vector and from every product, so the operator is
    :math:`P H P` with :math:`P = I - \\sum_t t t^T` restricted to the
    :math:`3N - 3` dimensional complement. A translation-invariant potential has
    :math:`H t = 0` exactly, so the raw Hessian's spectrum is the projected
    spectrum plus three zeros, :math:`e_{\\min}^{raw} = \\min(e_{\\min}^{proj}, 0)`,
    and the paper's rule is reproduced by
    :math:`\\lambda = \\mathrm{relu}(-e_{\\min}^{proj}) + \\varepsilon`. Projecting
    also keeps the Krylov space from spending steps on three directions whose
    eigenvalue is known. Rotations are *not* projected: away from a stationary
    point they are not zero modes.

Molecules smaller than the step count
    A molecule's Krylov space cannot exceed its dimension :math:`d` (``3N`` or
    ``3N - 3``). After ``min(k, d)`` steps, or once the new residual vanishes
    relative to the product it came from (``breakdown_tol``), the molecule stops:
    its later Lanczos vectors are zero and the corresponding rows and columns of
    its Ritz matrix are masked out. Other molecules in the batch continue.

Ritz values are upper bounds
    Every Ritz value is at least the true smallest eigenvalue (Courant-Fischer),
    and the gap closes as k grows. ``lambda = -theta + eps`` computed from a Ritz
    value :math:`\\theta` is therefore never larger than the exact rule, and with
    small :math:`\\varepsilon` it can come out *below* :math:`-e_{\\min}`, leaving
    :math:`H + \\lambda I` indefinite. Pick k so that the error is well below
    :math:`\\varepsilon`, or add a margin.
"""

from typing import Callable, NamedTuple, Optional

import torch
from torch import Tensor

__all__ = [
    "LanczosResult",
    "lanczos_min_eigenvalue",
    "molecule_dot",
    "remove_molecule_mean",
]


class LanczosResult(NamedTuple):
    """Output of :func:`lanczos_min_eigenvalue`.

    Attributes:
        e_min: Smallest Ritz value per molecule, shape ``(n_mol,)``, in the units
            of the operator. 0 for molecules with no degrees of freedom left.
        ritz_vector: Matching unit-norm Ritz vector per molecule, stacked as
            ``(n_atoms_total, 3)``, or ``None`` unless requested.
        residual_norm: :math:`\\lVert A x - \\theta x \\rVert` per molecule,
            computed from the stored products (no extra matvec). A posteriori
            check: for a symmetric operator some eigenvalue lies within this
            distance of :math:`\\theta`.
        n_steps: Lanczos steps actually used per molecule, shape ``(n_mol,)``.
    """

    e_min: Tensor
    ritz_vector: Optional[Tensor]
    residual_norm: Tensor
    n_steps: Tensor


def molecule_dot(a: Tensor, b: Tensor, idx_m: Tensor, n_mol: int) -> Tensor:
    """Per-molecule inner product of two stacked ``(n_atoms_total, 3)`` vectors."""
    out = torch.zeros(n_mol, dtype=a.dtype, device=a.device)
    return out.index_add_(0, idx_m, (a * b).sum(-1))


def remove_molecule_mean(v: Tensor, idx_m: Tensor, n_atoms: Tensor) -> Tensor:
    """Subtract each molecule's mean over atoms: projects out the translations.

    For a stacked vector of shape ``(n_atoms_total, 3)`` this is :math:`P v` with
    :math:`P = I - \\sum_t t t^T` over the three unit translations of each
    molecule. Applied to forces, it removes the net force.
    """
    n_mol = n_atoms.shape[0]
    sums = torch.zeros(n_mol, v.shape[-1], dtype=v.dtype, device=v.device)
    sums = sums.index_add(0, idx_m, v)
    mean = sums / n_atoms.to(v.dtype).unsqueeze(-1)
    return v - mean[idx_m]


@torch.no_grad()
def lanczos_min_eigenvalue(
    hvp_fn: Callable[[Tensor], Tensor],
    idx_m: Tensor,
    n_atoms: Tensor,
    k: int = 20,
    project_translations: bool = True,
    seed: Optional[int] = 0,
    start: Optional[Tensor] = None,
    return_ritz_vector: bool = False,
    breakdown_tol: float = 1e-5,
    dtype: torch.dtype = torch.float64,
) -> LanczosResult:
    """Smallest eigenvalue of each molecule's Hessian by Lanczos on batched HVPs.

    Args:
        hvp_fn: Maps a stacked vector ``V`` of shape ``(n_atoms_total, 3)`` to the
            stacked products ``H V`` (or ``H^T V``, see the module docstring).
            It receives ``V`` in ``dtype`` and must cast it to whatever its model
            needs; its output is cast back to ``dtype``. It must act on each
            molecule independently (block-diagonal batch Hessian).
        idx_m: Molecule index of every atom, shape ``(n_atoms_total,)``.
        n_atoms: Atoms per molecule, shape ``(n_mol,)``.
        k: Maximum number of Lanczos steps, i.e. matvecs. Capped per molecule at
            its dimension.
        project_translations: Remove the three translations from the start
            vector and every product (operator :math:`P H P`, dimension
            ``3N - 3``). See the module docstring for why this reproduces the
            paper's damping rule as ``relu(-e_min) + eps``.
        seed: Seed of the random start vector (ignored if ``start`` is given).
            The vector is drawn on the CPU, so it does not depend on the device.
            ``None`` draws from the global generator.
        start: Optional explicit start vector, shape ``(n_atoms_total, 3)``.
        return_ritz_vector: Also return the Ritz vector of ``e_min``.
        breakdown_tol: A molecule stops once the norm of its reorthogonalized
            residual falls below this fraction of the norm of the product it came
            from: its Krylov space is then (numerically) invariant.
        dtype: Working precision of the Lanczos vectors and the Ritz problem.

    Returns:
        A :class:`LanczosResult`.
    """
    if k < 1:
        raise ValueError(f"need at least one Lanczos step, got k={k}")

    device = idx_m.device
    n_mol = int(n_atoms.shape[0])
    n_total = int(idx_m.shape[0])
    idx_m = idx_m.long()
    n_atoms = n_atoms.long().to(device)

    dim = 3 * n_atoms - (3 if project_translations else 0)
    max_steps = torch.clamp(dim, min=0).clamp(max=k)

    def project(v: Tensor) -> Tensor:
        return remove_molecule_mean(v, idx_m, n_atoms) if project_translations else v

    def per_atom(c: Tensor) -> Tensor:
        return c[idx_m].unsqueeze(-1)

    # ---- start vector ----------------------------------------------------
    if start is None:
        generator = None
        if seed is not None:
            generator = torch.Generator().manual_seed(int(seed))
        start = torch.randn(n_total, 3, generator=generator, dtype=torch.float64)
    q = project(start.to(device=device, dtype=dtype))
    norm = molecule_dot(q, q, idx_m, n_mol).sqrt()
    active = (max_steps > 0) & (norm > 0)
    q = torch.where(per_atom(active), q / per_atom(norm.clamp_min(1e-300)), 0.0)

    Q = torch.zeros(k, n_total, 3, dtype=dtype, device=device)
    W = torch.zeros_like(Q)
    used = torch.zeros(k, n_mol, dtype=torch.bool, device=device)

    # ---- Lanczos with full reorthogonalization -----------------------------
    for j in range(k):
        if not bool(active.any()):
            break
        Q[j] = q
        used[j] = active
        w = hvp_fn(q).to(dtype)
        w = project(w)
        w = torch.where(per_atom(active), w, 0.0)
        W[j] = w

        if j == k - 1:
            break

        r = w
        for _ in range(2):
            coeffs = torch.stack(
                [molecule_dot(Q[i], r, idx_m, n_mol) for i in range(j + 1)]
            )  # (j+1, n_mol)
            r = r - (coeffs[:, idx_m].unsqueeze(-1) * Q[: j + 1]).sum(0)
        r = project(r)

        beta = molecule_dot(r, r, idx_m, n_mol).sqrt()
        w_norm = molecule_dot(w, w, idx_m, n_mol).sqrt()
        active = (
            active & (beta > breakdown_tol * w_norm) & (beta > 0) & (max_steps > j + 1)
        )
        q = torch.where(per_atom(active), r / per_atom(beta.clamp_min(1e-300)), 0.0)

    # ---- Rayleigh-Ritz -----------------------------------------------------
    # T[m, i, j] = <q_i, A q_j> summed over the atoms of molecule m.
    per_atom_T = torch.einsum("iac,jac->aij", Q, W)  # (n_total, k, k)
    T = torch.zeros(n_mol, k, k, dtype=dtype, device=device)
    T = T.index_add(0, idx_m, per_atom_T)
    T = 0.5 * (T + T.transpose(1, 2))

    used_t = used.T  # (n_mol, k)
    mask = used_t.unsqueeze(-1) & used_t.unsqueeze(-2)
    T = torch.where(mask, T, 0.0)
    # Unused directions get a diagonal above every real Ritz value, so they
    # cannot become the minimum; with zero coupling they do not perturb the rest.
    big = 2.0 * k * T.abs().amax(dim=(1, 2)) + 1.0
    eye = torch.eye(k, dtype=torch.bool, device=device)
    T = torch.where((~used_t).unsqueeze(-1) & eye, big.view(-1, 1, 1).expand_as(T), T)

    evals, evecs = torch.linalg.eigh(T)
    n_steps = used_t.sum(-1)
    e_min = torch.where(n_steps > 0, evals[:, 0], 0.0)
    s = torch.where(used_t, evecs[:, :, 0], 0.0)  # (n_mol, k)

    coeff = s.T[:, idx_m].unsqueeze(-1)  # (k, n_total, 1)
    x = (coeff * Q).sum(0)
    ax = (coeff * W).sum(0)
    res = ax - per_atom(e_min) * x
    residual_norm = molecule_dot(res, res, idx_m, n_mol).sqrt()

    return LanczosResult(
        e_min=e_min,
        ritz_vector=x if return_ritz_vector else None,
        residual_norm=residual_norm,
        n_steps=n_steps,
    )
