"""Newton-step training on stored Hessians with eigenvalue damping."""

import numpy as np
import pytest
import torch
import torchmetrics.aggregation
from hydra import compose, initialize_config_module
from hydra.utils import instantiate
from omegaconf import open_dict

import schnetpack as spk
from schnetpack.train.dense_hessian import pad_hessians

from .conftest import N_ATOM_BASIS, _representation

DAMPING_OFFSET = 0.1


def _random_hessian(n3, rng, positive_definite):
    m = rng.normal(size=(n3, n3))
    if positive_definite:
        return m @ m.T / n3 + 0.5 * np.eye(n3)
    return 0.5 * (m + m.T)


@pytest.fixture
def dense_newton_batch(newton_batch):
    """``newton_batch`` with forces and flat Hessians; molecule 1 is positive definite."""
    rng = np.random.default_rng(0)
    n_atoms = newton_batch[spk.properties.n_atoms].tolist()
    hessians = [
        _random_hessian(3 * n, rng, positive_definite=(m == 1))
        for m, n in enumerate(n_atoms)
    ]
    batch = dict(newton_batch)
    batch[spk.properties.hessian] = torch.tensor(
        np.concatenate([h.reshape(-1) for h in hessians]), dtype=torch.float32
    )
    batch[spk.properties.forces] = torch.tensor(
        rng.normal(size=(sum(n_atoms), 3)), dtype=torch.float32
    )
    batch["_hessians_np"] = hessians
    return batch


@pytest.fixture
def step_model():
    torch.manual_seed(0)
    return spk.model.NeuralNetworkPotential(
        representation=_representation(),
        input_modules=[spk.atomistic.PairwiseDistances()],
        output_modules=[
            spk.atomistic.NewtonStep(
                newton_step_key=spk.properties.newton_step,
                n_in=N_ATOM_BASIS,
                n_hidden=12,
            )
        ],
    )


@pytest.fixture
def dense_task(step_model):
    outputs = [
        spk.task.ModelOutput(
            name=spk.properties.damped_hvp,
            target_property=spk.properties.forces,
            loss_fn=torch.nn.MSELoss(),
            metrics={},
        ),
        spk.task.UnsupervisedModelOutput(
            name=spk.properties.damping_factor,
            metrics={"mean": torchmetrics.aggregation.MeanMetric()},
            loss_weight=0.0,
        ),
    ]
    return spk.train.DenseHessianNewtonTask(
        model=step_model,
        outputs=outputs,
        hessian_unit="Hartree/Bohr/Bohr",
        damping_offset=DAMPING_OFFSET,
        optimizer_cls=torch.optim.AdamW,
        optimizer_args={"lr": 1e-3},
    )


def _reference(hessians, step, n_atoms):
    """Per-molecule numpy ``(H + lambda I) p`` and ``lambda``."""
    products, dampings, start = [], [], 0
    for h, n in zip(hessians, n_atoms):
        e_min = np.linalg.eigvalsh(h)[0]
        damping = 0.0 if e_min > 0 else -e_min + DAMPING_OFFSET
        p = step[start : start + n].reshape(-1)
        products.append(((h + damping * np.eye(len(h))) @ p).reshape(-1, 3))
        dampings.append(damping)
        start += n
    return np.concatenate(products), np.array(dampings)


def test_damping_and_product_match_numpy(dense_task, dense_newton_batch):
    batch = dense_newton_batch
    n_atoms = batch[spk.properties.n_atoms].tolist()
    pred = dense_task.augment_predictions(batch, dense_task(dict(batch)))

    step = pred[spk.properties.newton_step].detach().double().numpy()
    hessians = [h.astype(np.float32).astype(np.float64) for h in batch["_hessians_np"]]
    product, damping = _reference(hessians, step, n_atoms)

    predicted_damping = pred[spk.properties.damping_factor].numpy()
    assert predicted_damping[1] == 0.0, "a positive definite Hessian is not damped"
    assert (predicted_damping[[0, 2]] > DAMPING_OFFSET).all()
    np.testing.assert_allclose(predicted_damping, damping, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(
        pred[spk.properties.damped_hvp].detach().numpy(), product, rtol=1e-4, atol=1e-5
    )


def test_stacked_and_flat_layouts_agree():
    rng = np.random.default_rng(1)
    n_atoms = torch.tensor([3, 3])
    hessians = [_random_hessian(9, rng, False) for _ in range(2)]
    stacked = torch.tensor(np.concatenate(hessians, axis=0))
    flat = torch.tensor(np.concatenate([h.reshape(-1) for h in hessians]))

    assert stacked.shape == (18, 9)
    torch.testing.assert_close(
        pad_hessians(stacked, n_atoms), pad_hessians(flat, n_atoms)
    )
    torch.testing.assert_close(
        pad_hessians(flat, n_atoms), torch.tensor(np.stack(hessians))
    )


def test_mismatched_hessian_size_is_rejected():
    with pytest.raises(ValueError, match="Hessian entries"):
        pad_hessians(torch.zeros(80), torch.tensor([3]))


def test_damped_newton_step_minimises_the_loss(dense_task, dense_newton_batch):
    """At ``p = (H + lambda I)^{-1} F`` the residual and its gradient vanish."""
    batch = dense_newton_batch
    n_atoms = batch[spk.properties.n_atoms].tolist()
    forces = batch[spk.properties.forces].double().numpy()
    hessians = [h.astype(np.float32).astype(np.float64) for h in batch["_hessians_np"]]
    _, damping = _reference(hessians, np.zeros_like(forces), n_atoms)
    steps, start = [], 0
    for h, lam, n in zip(hessians, damping, n_atoms):
        f = forces[start : start + n].reshape(-1)
        steps.append(np.linalg.solve(h + lam * np.eye(len(h)), f).reshape(-1, 3))
        start += n

    step = torch.tensor(np.concatenate(steps), dtype=torch.float32, requires_grad=True)
    pred = dense_task.augment_predictions(batch, {spk.properties.newton_step: step})
    loss = torch.nn.functional.mse_loss(
        pred[spk.properties.damped_hvp], batch[spk.properties.forces]
    )
    loss.backward()

    assert loss.item() < 1e-8
    assert step.grad.abs().max() < 1e-4


def test_training_and_validation_steps_run(dense_task, dense_newton_batch):
    loss = dense_task.training_step(dict(dense_newton_batch), 0)
    assert torch.isfinite(loss)
    loss.backward()
    grads = [p.grad for p in dense_task.model.parameters() if p.grad is not None]
    assert grads and any(g.abs().sum() > 0 for g in grads)

    with torch.no_grad():
        val_loss = dense_task.validation_step(dict(dense_newton_batch), 0)["val_loss"]
    assert torch.isfinite(val_loss)


@pytest.fixture
def experiment_config():
    with initialize_config_module(
        config_module="schnetpack.configs", version_base="1.2"
    ):
        config = compose(
            config_name="train",
            overrides=[
                "experiment=newton_step_training_dense_hessian",
                "model.representation.n_atom_basis=16",
                "model.representation.n_interactions=2",
                "run.work_dir=.",
                "run.path=.",
            ],
            return_hydra_config=True,
        )
    with open_dict(config):
        del config["hydra"]
    return config


@pytest.mark.integration
def test_config_keys_and_units(experiment_config):
    import schnetpack.properties as properties

    globals_ = experiment_config.globals
    assert globals_.newton_step_key == properties.newton_step
    assert globals_.damping_factor_key == properties.damping_factor
    assert globals_.damped_hvp_key == properties.damped_hvp
    assert globals_.hessian_key == properties.hessian
    units = experiment_config.data.dataset.property_units
    assert set(units) == {globals_.forces_key, globals_.hessian_key}
    assert units.hessian == experiment_config.task.hessian_unit == "kcal/mol/Ang/Ang"


@pytest.mark.integration
def test_configured_task_runs_a_step(experiment_config, dense_newton_batch):
    model = instantiate(experiment_config.model)
    task = instantiate(
        experiment_config.task, model=model, _convert_="partial", _recursive_=True
    )
    assert task.damping_offset == pytest.approx(
        0.1 * spk.units.convert_units("Hartree/Bohr/Bohr", "kcal/mol/Ang/Ang")
    )

    loss = task.training_step(dict(dense_newton_batch), 0)

    assert torch.isfinite(loss), loss
    loss.backward()
    assert any(
        p.grad is not None and p.grad.abs().sum() > 0 for p in task.model.parameters()
    )
