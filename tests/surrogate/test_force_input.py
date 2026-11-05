"""The reference forces as an input to the student (ForceInput + PaiNN inputs).

Checks: the input module's scale and direction; a force-input student starts
out identical to the geometry-only one it is initialized from (all new weights
are zero); the step is still equivariant once those weights are nonzero; the
task runs the reference potential once per batch and hands its forces to the
student; models pickled before the inputs existed still run; scripting works.
"""

import pytest
import torch

import schnetpack as spk
from schnetpack import properties
from tests.surrogate.conftest import (
    CUTOFF,
    N_ATOM_BASIS,
    N_INTERACTIONS,
    N_RBF,
)

FORCE_KEY = "input_forces"


def _representation(with_inputs):
    kwargs = {}
    if with_inputs:
        kwargs = dict(
            vector_input_key="force_input_dir",
            scalar_input_key="force_input_scalars",
            n_scalar_inputs=2,
        )
    return spk.representation.PaiNN(
        n_atom_basis=N_ATOM_BASIS,
        n_interactions=N_INTERACTIONS,
        radial_basis=spk.nn.GaussianRBF(n_rbf=N_RBF, cutoff=CUTOFF),
        cutoff_fn=spk.nn.CosineCutoff(CUTOFF),
        **kwargs,
    )


def _student(with_inputs, seed=0):
    torch.manual_seed(seed)
    input_modules = [spk.atomistic.PairwiseDistances()]
    if with_inputs:
        input_modules.append(spk.atomistic.ForceInput(force_key=FORCE_KEY))
    return spk.model.NeuralNetworkPotential(
        representation=_representation(with_inputs),
        input_modules=input_modules,
        output_modules=[
            spk.atomistic.NewtonStep(
                newton_step_key=properties.newton_step,
                n_in=N_ATOM_BASIS,
                n_hidden=12,
            )
        ],
    )


def _with_forces(batch, seed=3, scale=1.0):
    g = torch.Generator().manual_seed(seed)
    batch = dict(batch)
    batch[FORCE_KEY] = scale * torch.randn(
        batch[properties.R].shape, generator=g, dtype=batch[properties.R].dtype
    )
    return batch


def test_force_input_scale_and_direction(newton_batch):
    batch = _with_forces(newton_batch, scale=3.0)
    out = spk.atomistic.ForceInput(force_key=FORCE_KEY, floor=0.5)(dict(batch))

    forces = batch[FORCE_KEY]
    idx_m = batch[properties.idx_m]
    for m, n in enumerate(batch[properties.n_atoms].tolist()):
        f = forces[idx_m == m]
        s = torch.sqrt((f**2).mean() + 0.5**2)
        assert out["force_input_scale"][m] == pytest.approx(s.item(), rel=1e-6)
        assert torch.allclose(out["force_input_dir"][idx_m == m], f / s)
        assert torch.allclose(
            out["force_input_scalars"][idx_m == m, 0], torch.log(s).expand(n)
        )
    assert torch.allclose(
        out["force_input_scalars"][:, 1],
        torch.log(torch.sqrt((out["force_input_dir"] ** 2).sum(-1) + 1e-6)),
    )


def test_force_input_floor_bounds_the_scale(newton_batch):
    batch = _with_forces(newton_batch, scale=1e-6)
    out = spk.atomistic.ForceInput(force_key=FORCE_KEY, floor=0.02306)(dict(batch))
    assert torch.allclose(
        out["force_input_scale"], torch.full_like(out["force_input_scale"], 0.02306)
    )


def test_force_input_student_starts_as_the_geometry_only_one(newton_batch):
    plain = _student(with_inputs=False)
    fin = _student(with_inputs=True)
    missing, unexpected = fin.load_state_dict(plain.state_dict(), strict=False)
    assert not unexpected
    assert sorted(missing) == [
        "representation.scalar_input_net.bias",
        "representation.scalar_input_net.weight",
        "representation.vector_input_weight",
    ]
    for key in missing:
        assert torch.all(fin.state_dict()[key] == 0)

    batch = _with_forces(newton_batch)
    step_plain = plain(dict(batch))[properties.newton_step]
    step_fin = fin(dict(batch))[properties.newton_step]
    assert torch.equal(step_plain, step_fin)


def test_force_input_changes_the_step_once_trained(newton_batch):
    fin = _student(with_inputs=True)
    batch = _with_forces(newton_batch)
    step0 = fin(dict(batch))[properties.newton_step]
    with torch.no_grad():
        fin.representation.vector_input_weight.fill_(0.5)
    step1 = fin(dict(batch))[properties.newton_step]
    assert not torch.allclose(step0, step1)
    # and it depends on the forces, not just on the geometry
    step2 = fin(dict(_with_forces(newton_batch, seed=4)))[properties.newton_step]
    assert not torch.allclose(step1, step2)


def test_force_input_step_is_equivariant(newton_batch):
    fin = _student(with_inputs=True)
    with torch.no_grad():
        fin.representation.vector_input_weight.normal_()
        fin.representation.scalar_input_net.weight.normal_()
    batch = _with_forces(newton_batch)

    # a rotation (QR of a fixed random matrix, sign-fixed to det +1)
    g = torch.Generator().manual_seed(7)
    q, _ = torch.linalg.qr(torch.randn(3, 3, generator=g))
    if torch.det(q) < 0:
        q[:, 0] = -q[:, 0]
    q = q.to(batch[properties.R].dtype)

    rotated = dict(batch)
    rotated[properties.R] = batch[properties.R] @ q.T
    rotated[FORCE_KEY] = batch[FORCE_KEY] @ q.T

    step = fin(dict(batch))[properties.newton_step]
    step_rot = fin(rotated)[properties.newton_step]
    assert torch.allclose(step @ q.T, step_rot, atol=1e-5)


def test_old_pickles_without_the_inputs_still_run(newton_batch):
    plain = _student(with_inputs=False)
    rep = plain.representation
    del rep.__dict__["vector_input_key"]
    del rep.__dict__["scalar_input_key"]
    assert not hasattr(rep, "vector_input_key")
    step = plain(dict(newton_batch))[properties.newton_step]
    assert torch.isfinite(step).all()


def test_force_input_representation_scripts(newton_batch):
    # The NewtonStep head itself does not script (gated MLP, not touched
    # here), so the inputs and the representation are scripted on their own,
    # behind an Atomwise head.
    torch.manual_seed(0)
    model = spk.model.NeuralNetworkPotential(
        representation=_representation(with_inputs=True),
        input_modules=[
            spk.atomistic.PairwiseDistances(),
            spk.atomistic.ForceInput(force_key=FORCE_KEY),
        ],
        # Forces, so that required_derivatives is not an empty list, which
        # TorchScript cannot type (as for the reference models in conftest)
        output_modules=[
            spk.atomistic.Atomwise(n_in=N_ATOM_BASIS, output_key="energy"),
            spk.atomistic.Forces(),
        ],
    )
    with torch.no_grad():
        model.representation.vector_input_weight.normal_()
    scripted = torch.jit.script(model)
    batch = _with_forces(newton_batch)
    assert torch.allclose(
        scripted(dict(batch))["energy"], model(dict(batch))["energy"]
    )


def test_task_hands_the_reference_forces_to_the_student(
    newton_batch, surrogate_outputs, ref_model_path, monkeypatch
):
    fin = _student(with_inputs=True)
    with torch.no_grad():
        fin.representation.vector_input_weight.fill_(0.3)
    task = spk.train.NewtonSurrogateTask(
        model=fin,
        outputs=[surrogate_outputs[0], surrogate_outputs[2]],
        ref_model_path=ref_model_path,
        damping_source="lanczos",
        lanczos_steps=4,
        damping_eps=1.0,
        remove_teacher_net_force=True,
        force_input_key=FORCE_KEY,
        optimizer_cls=torch.optim.AdamW,
        optimizer_args={"lr": 1e-3, "weight_decay": 0.01},
    )

    calls = []
    ref_forward = task.ref_model.forward

    def counted(inputs):
        calls.append(1)
        return ref_forward(inputs)

    monkeypatch.setattr(task.ref_model, "forward", counted)

    batch = dict(newton_batch)
    loss = task.training_step(batch, 0)
    assert torch.isfinite(loss)
    assert len(calls) == 1, "the reference potential should run once per batch"

    # the student saw the (net-force-free, detached) reference forces
    assert FORCE_KEY in batch
    assert batch[FORCE_KEY].grad_fn is None
    assert torch.allclose(
        batch[FORCE_KEY],
        spk.train.lanczos.remove_molecule_mean(
            task.ref_model(dict(newton_batch))[properties.forces].detach(),
            batch[properties.idx_m],
            batch[properties.n_atoms],
        ),
        atol=1e-6,
    )
    assert task._ref_forces_graph_key not in batch

    loss.backward()
    assert fin.representation.vector_input_weight.grad is not None
    assert fin.representation.vector_input_weight.grad.abs().sum() > 0
    assert fin.representation.scalar_input_net.weight.grad.abs().sum() > 0


def test_task_without_force_input_is_unchanged(
    newton_batch, surrogate_task, monkeypatch
):
    batch = dict(newton_batch)
    loss = surrogate_task.training_step(batch, 0)
    assert torch.isfinite(loss)
    assert FORCE_KEY not in batch
    assert surrogate_task._ref_forces_graph_key not in batch
