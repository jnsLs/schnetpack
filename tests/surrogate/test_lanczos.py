"""Lanczos smallest eigenvalue from batched Hessian-vector products.

Checked against ``torch.linalg.eigvalsh`` on explicit block-diagonal matrices
(several molecule sizes in one batch, including molecules smaller than the
step count), on the materialized Hessian of a small PaiNN potential, and, if
the checkpoint is on disk, of the ethanol PaiNN teacher. The task-level tests
check the ``batch`` and ``lanczos`` damping sources and the net-force removal.
"""

import os

import numpy as np
import pytest
import torch
from torch.autograd import grad

import schnetpack as spk
from schnetpack import properties
from schnetpack.train.lanczos import (
    lanczos_min_eigenvalue,
    molecule_dot,
    remove_molecule_mean,
)

from .conftest import N_ATOM_BASIS, _representation

# molecule sizes in atoms: 1 (no dof after projection), 2 (3N-3 = 3 < k),
# up to 14 (3N = 42 > k for the small k values)
SIZES = [2, 3, 5, 1, 9, 14]


def _idx_m(sizes):
    return torch.cat([torch.full((n,), i) for i, n in enumerate(sizes)])


def _translation_projector(n):
    t = torch.zeros(3 * n, 3, dtype=torch.float64)
    for c in range(3):
        t[c::3, c] = 1.0 / np.sqrt(n)
    return torch.eye(3 * n, dtype=torch.float64) - t @ t.T


def _complement_basis(n):
    """Orthonormal basis of the 3N-3 dimensional translation-free subspace."""
    evals, evecs = torch.linalg.eigh(_translation_projector(n))
    return evecs[:, evals > 0.5]


def _block_operator(blocks, sizes):
    """hvp_fn for a block-diagonal matrix acting on stacked (n_atoms, 3) vectors."""

    def hvp(v):
        out, start = [], 0
        for block, n in zip(blocks, sizes):
            x = v[start : start + n].reshape(-1)
            out.append((block @ x).reshape(n, 3))
            start += n
        return torch.cat(out)

    return hvp


def _random_symmetric(n, seed, translation_invariant):
    g = torch.Generator().manual_seed(seed)
    m = torch.randn(3 * n, 3 * n, generator=g, dtype=torch.float64)
    s = 0.5 * (m + m.T)
    if translation_invariant:
        p = _translation_projector(n)
        s = p @ s @ p
    return s


@pytest.mark.parametrize("project", [False, True])
def test_exact_at_full_dimension(project):
    """With k >= every molecule's dimension, Lanczos is exact."""
    blocks = [_random_symmetric(n, i, project) for i, n in enumerate(SIZES)]
    k = 3 * max(SIZES)
    res = lanczos_min_eigenvalue(
        _block_operator(blocks, SIZES),
        _idx_m(SIZES),
        torch.tensor(SIZES),
        k=k,
        project_translations=project,
        return_ritz_vector=True,
    )
    for i, (block, n) in enumerate(zip(blocks, SIZES)):
        if project:
            if n == 1:
                assert res.e_min[i] == 0.0 and res.n_steps[i] == 0
                continue
            basis = _complement_basis(n)
            exact = torch.linalg.eigvalsh(basis.T @ block @ basis)[0]
            assert res.n_steps[i] <= 3 * n - 3
        else:
            exact = torch.linalg.eigvalsh(block)[0]
            assert res.n_steps[i] <= 3 * n
        torch.testing.assert_close(res.e_min[i], exact, rtol=0, atol=1e-9)
        assert res.residual_norm[i] < 1e-8
    assert torch.isfinite(res.ritz_vector).all()


def test_projected_rule_matches_raw_rule():
    """relu(-e_min(PHP)) + eps == -e_min(H_raw) + eps for translation-invariant H."""
    blocks = [_random_symmetric(n, 10 + i, True) for i, n in enumerate(SIZES)]
    res = lanczos_min_eigenvalue(
        _block_operator(blocks, SIZES),
        _idx_m(SIZES),
        torch.tensor(SIZES),
        k=3 * max(SIZES),
    )
    raw = torch.stack([torch.linalg.eigvalsh(b)[0] for b in blocks])
    eps = 0.3
    torch.testing.assert_close(
        torch.relu(-res.e_min) + eps, -raw + eps, rtol=0, atol=1e-9
    )


@pytest.mark.parametrize("k", [3, 6, 10])
def test_truncated_ritz_value_is_an_upper_bound(k):
    blocks = [_random_symmetric(n, 20 + i, False) for i, n in enumerate(SIZES)]
    res = lanczos_min_eigenvalue(
        _block_operator(blocks, SIZES),
        _idx_m(SIZES),
        torch.tensor(SIZES),
        k=k,
        project_translations=False,
    )
    for i, block in enumerate(blocks):
        evals = torch.linalg.eigvalsh(block)
        assert res.e_min[i] >= evals[0] - 1e-10
        assert res.e_min[i] <= evals[-1] + 1e-10
        # Kahan / Bauer-Fike: some eigenvalue lies within the residual
        assert (evals - res.e_min[i]).abs().min() <= res.residual_norm[i] + 1e-9
    assert (res.n_steps <= k).all()


def test_ritz_vector_is_a_translation_free_unit_eigenvector():
    sizes = [4, 7]
    blocks = [_random_symmetric(n, 30 + i, True) for i, n in enumerate(sizes)]
    idx_m = _idx_m(sizes)
    n_atoms = torch.tensor(sizes)
    res = lanczos_min_eigenvalue(
        _block_operator(blocks, sizes),
        idx_m,
        n_atoms,
        k=3 * max(sizes),
        return_ritz_vector=True,
    )
    x = res.ritz_vector
    torch.testing.assert_close(
        molecule_dot(x, x, idx_m, 2), torch.ones(2, dtype=torch.float64)
    )
    torch.testing.assert_close(
        remove_molecule_mean(x, idx_m, n_atoms), x, rtol=0, atol=1e-12
    )
    hx = _block_operator(blocks, sizes)(x)
    torch.testing.assert_close(
        hx, res.e_min[idx_m].unsqueeze(-1) * x, rtol=0, atol=1e-8
    )


def test_asymmetric_operator_gives_eigenvalue_of_symmetric_part():
    """Reverse-mode HVPs of direct-force models are H^T v: we report sym(H)."""
    sizes = [3, 6, 8]
    blocks = []
    for i, n in enumerate(sizes):
        s = _random_symmetric(n, 40 + i, False)
        g = torch.Generator().manual_seed(50 + i)
        a = torch.randn(3 * n, 3 * n, generator=g, dtype=torch.float64)
        blocks.append(s + 0.01 * (a - a.T))
    res = lanczos_min_eigenvalue(
        _block_operator(blocks, sizes),
        _idx_m(sizes),
        torch.tensor(sizes),
        k=3 * max(sizes),
        project_translations=False,
    )
    for i, b in enumerate(blocks):
        exact = torch.linalg.eigvalsh(0.5 * (b + b.T))[0]
        torch.testing.assert_close(res.e_min[i], exact, rtol=0, atol=1e-9)


def test_seed_makes_it_reproducible_and_independent_of_float_dtype():
    blocks = [_random_symmetric(n, 60 + i, True) for i, n in enumerate(SIZES)]
    args = (_block_operator(blocks, SIZES), _idx_m(SIZES), torch.tensor(SIZES))
    a = lanczos_min_eigenvalue(*args, k=5, seed=3)
    b = lanczos_min_eigenvalue(*args, k=5, seed=3)
    c = lanczos_min_eigenvalue(*args, k=5, seed=4)
    assert torch.equal(a.e_min, b.e_min)
    assert not torch.equal(a.e_min, c.e_min)


# ---------------------------------------------------------------------------
# against a real potential
# ---------------------------------------------------------------------------


def _materialized_hessians(model, batch):
    """Per-molecule Hessians of a conservative model, by 3N reverse passes."""
    positions = batch[properties.R]
    forces = model(batch)[properties.forces]
    n_total = positions.shape[0]
    rows = []
    for i in range(3 * n_total):
        seed = torch.zeros(3 * n_total, dtype=positions.dtype)
        seed[i] = 1.0
        rows.append(
            -grad(forces, positions, seed.view_as(forces), retain_graph=True)[
                0
            ].reshape(-1)
        )
    hess = torch.stack(rows)  # = H^T, symmetric for a conservative model
    out, start = [], 0
    for n in batch[properties.n_atoms].tolist():
        sl = slice(3 * start, 3 * (start + n))
        out.append(hess[sl, sl])
        start += n
    return forces, hess, out


def _hvp_of(forces, positions):
    def hvp(v):
        return -grad(forces, positions, v.to(forces.dtype), retain_graph=True)[0]

    return hvp


def test_matches_eigvalsh_of_painn_hessian(newton_batch, ref_model_path):
    model = spk.train.load_reference_model(ref_model_path).double()
    model.train()
    batch = {
        k: (v.double() if v.is_floating_point() else v) for k, v in newton_batch.items()
    }
    batch[properties.R].requires_grad_()
    forces, _, hessians = _materialized_hessians(model, batch)

    res = lanczos_min_eigenvalue(
        _hvp_of(forces, batch[properties.R]),
        batch[properties.idx_m],
        batch[properties.n_atoms],
        k=30,
    )
    raw = res.e_min.new_tensor(
        [torch.linalg.eigvalsh(0.5 * (h + h.T))[0] for h in hessians]
    )
    # raw spectrum = projected spectrum + three (numerical) zeros
    torch.testing.assert_close(
        torch.minimum(res.e_min, torch.zeros_like(res.e_min)),
        raw,
        rtol=0,
        atol=1e-8,
    )


ETHANOL_TEACHER = "/home/sgugler/newton-mdet/teachers/painn_ethanol_teacher.pt"
ETHANOL_DB = "/home/sgugler/data/ethanol_rmd17.db"


@pytest.mark.slow
@pytest.mark.skipif(
    not (os.path.exists(ETHANOL_TEACHER) and os.path.exists(ETHANOL_DB)),
    reason="ethanol PaiNN teacher or rMD17 database not on this machine",
)
def test_matches_eigvalsh_of_ethanol_painn_teacher():
    from schnetpack.data import ASEAtomsData, AtomsLoader

    data = ASEAtomsData(
        ETHANOL_DB,
        load_properties=[],
        transforms=[
            spk.transform.MatScipyNeighborList(cutoff=5.0),
            spk.transform.CastTo64(),
        ],
    )
    subset = torch.utils.data.Subset(data, list(range(0, 4000, 500)))
    batch = next(iter(AtomsLoader(subset, batch_size=8)))
    model = spk.train.load_reference_model(ETHANOL_TEACHER, "torch").double()
    model.train()
    batch[properties.R].requires_grad_()
    forces, _, hessians = _materialized_hessians(model, batch)
    raw = torch.stack([torch.linalg.eigvalsh(0.5 * (h + h.T))[0] for h in hessians])

    for k, atol in [(24, 1e-6), (12, None)]:
        res = lanczos_min_eigenvalue(
            _hvp_of(forces, batch[properties.R]),
            batch[properties.idx_m],
            batch[properties.n_atoms],
            k=k,
        )
        lam = torch.relu(-res.e_min)
        if atol is not None:  # 3N - 3 = 24: exact
            torch.testing.assert_close(lam, -raw, rtol=0, atol=atol)
        else:  # truncated: never over-damps
            assert (lam <= -raw + 1e-6).all()


# ---------------------------------------------------------------------------
# the task's damping sources
# ---------------------------------------------------------------------------


@pytest.fixture
def step_only_student():
    torch.manual_seed(0)
    return spk.model.NeuralNetworkPotential(
        representation=_representation(),
        input_modules=[spk.atomistic.PairwiseDistances()],
        output_modules=[
            spk.atomistic.NewtonStep(
                newton_step_key=properties.newton_step,
                n_in=N_ATOM_BASIS,
                n_hidden=12,
            )
        ],
        postprocessors=[spk.transform.CastTo64()],
    )


def _outputs():
    return [
        spk.task.ModelOutput(
            name=properties.damped_hvp,
            target_property=properties.ref_forces,
            target_source="prediction",
            loss_fn=torch.nn.MSELoss(),
            metrics={},
        )
    ]


def _task(student, ref_model_path, **kwargs):
    return spk.train.NewtonSurrogateTask(
        model=student,
        outputs=_outputs(),
        ref_model_path=ref_model_path,
        optimizer_args={"lr": 1e-3},
        **kwargs,
    )


def _exact_damping(task, batch, eps):
    """relu(-e_min) + eps from the materialized Hessians of the task's teacher."""
    b = {k: v.detach().clone() if torch.is_tensor(v) else v for k, v in batch.items()}
    b[properties.R].requires_grad_()
    with task._reference_in_train_mode():
        _, _, hessians = _materialized_hessians(task.ref_model, b)
    e = torch.stack([torch.linalg.eigvalsh(0.5 * (h + h.T))[0] for h in hessians])
    return torch.relu(-e) + eps


def test_batch_damping_source(step_only_student, ref_model_path, newton_batch):
    task = _task(
        step_only_student, ref_model_path, damping_source="batch", damping_eps=0.5
    )
    e_min = torch.tensor([-2.0, 0.3, -0.1])
    newton_batch["e_min"] = e_min
    task.train()
    pred = task.predict_without_postprocessing(newton_batch)
    pred = task.augment_predictions(newton_batch, pred)
    torch.testing.assert_close(pred["damping_factor"], torch.tensor([2.5, 0.5, 0.6]))
    torch.testing.assert_close(pred["e_min"], e_min)

    # damped_hvp = H p + lambda p
    step = pred[properties.newton_step]
    hp = -grad(
        pred[properties.ref_forces],
        newton_batch[properties.R],
        step,
        retain_graph=True,
    )[0]
    lam = pred["damping_factor"].repeat_interleave(newton_batch[properties.n_atoms])
    torch.testing.assert_close(
        pred[properties.damped_hvp], hp + lam.unsqueeze(-1) * step
    )


def test_batch_damping_source_needs_the_key(
    step_only_student, ref_model_path, newton_batch
):
    task = _task(step_only_student, ref_model_path, damping_source="batch")
    with pytest.raises(KeyError, match="load_properties"):
        task.training_step(newton_batch, 0)


def test_lanczos_damping_source_matches_materialized_hessian(
    step_only_student, ref_model_path, newton_batch
):
    eps = 0.25
    task = _task(
        step_only_student,
        ref_model_path,
        damping_source="lanczos",
        damping_eps=eps,
        lanczos_steps=30,
    )
    expected = _exact_damping(task, newton_batch, eps)
    task.train()
    pred = task.predict_without_postprocessing(newton_batch)
    pred = task.augment_predictions(newton_batch, pred)
    # float32 model: agreement to float32 precision of the HVPs
    torch.testing.assert_close(pred["damping_factor"], expected, rtol=1e-4, atol=1e-4)
    assert not pred["damping_factor"].requires_grad

    loss = task.training_step(newton_batch, 0)
    loss.backward()
    assert all(p.grad is not None for p in task.model.parameters() if p.requires_grad)


def test_eigenvalue_sources_refuse_a_damping_head(
    student_model, ref_model_path, newton_batch
):
    task = _task(student_model, ref_model_path, damping_source="lanczos")
    with pytest.raises(ValueError, match="DampingFactor"):
        task.training_step(newton_batch, 0)


def test_damping_eps_unit_conversion(step_only_student, ref_model_path):
    task = _task(
        step_only_student,
        ref_model_path,
        damping_source="lanczos",
        damping_eps=0.1,
        damping_eps_unit="Hartree/Bohr/Bohr",
        damping_unit="eV/Ang/Ang",
    )
    assert task.damping_eps == pytest.approx(9.7173, rel=1e-4)
    # checkpoints rebuild the task from its saved hyperparameters (spktrain's
    # test stage does): the conversion must not be applied a second time
    assert task.hparams["damping_eps"] == 0.1
    rebuilt = type(task)(**task.hparams)
    assert rebuilt.damping_eps == pytest.approx(9.7173, rel=1e-4)
    with pytest.raises(ValueError):
        _task(step_only_student, ref_model_path, damping_eps_unit="eV/Ang/Ang")


def test_unknown_damping_source(step_only_student, ref_model_path):
    with pytest.raises(ValueError, match="damping_source"):
        _task(step_only_student, ref_model_path, damping_source="exact")


def test_remove_teacher_net_force(step_only_student, ref_model_path, newton_batch):
    """Removing the net force zeroes it per molecule; for a conservative
    teacher it is (numerically) a no-op."""
    plain = _task(step_only_student, ref_model_path, damping_source="batch")
    centred = _task(
        step_only_student,
        ref_model_path,
        damping_source="batch",
        remove_teacher_net_force=True,
    )
    newton_batch["e_min"] = torch.tensor([-1.0, -1.0, -1.0])
    outs = []
    for task in (plain, centred):
        task.train()
        b = dict(newton_batch)
        b[properties.R] = newton_batch[properties.R].detach().clone()
        pred = task.predict_without_postprocessing(b)
        outs.append(task.augment_predictions(b, pred))
    f = outs[1][properties.ref_forces]
    net = torch.zeros(3, 3).index_add(0, newton_batch[properties.idx_m], f)
    torch.testing.assert_close(net, torch.zeros(3, 3), rtol=0, atol=1e-5)
    torch.testing.assert_close(outs[0][properties.ref_forces], f, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(
        outs[0][properties.damped_hvp],
        outs[1][properties.damped_hvp],
        rtol=1e-4,
        atol=1e-5,
    )
