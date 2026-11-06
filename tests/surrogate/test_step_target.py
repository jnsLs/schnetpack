"""The relative step loss on structures with a precomputed Newton step.

Checks: the task writes the step and the target divided by the target's scale,
zero for molecules without a target; the floor; an MSE on the two is the
relative step error on the masked molecules; the loss reaches the model; a
task without a step target writes nothing.
"""

import pytest
import torch

import schnetpack as spk
from schnetpack import properties

STEP_KEY = "newton_step_ref"
MASK_KEY = "has_newton_step_ref"


def _with_target(batch, mask=(1.0, 0.0, 1.0), seed=5, scale=0.1):
    g = torch.Generator().manual_seed(seed)
    batch = dict(batch)
    batch[STEP_KEY] = scale * torch.randn(
        batch[properties.R].shape, generator=g, dtype=torch.float64
    )
    batch[MASK_KEY] = torch.tensor(mask, dtype=torch.float64)
    return batch


def _task(student_model, surrogate_outputs, ref_model_path, **kwargs):
    outputs = list(surrogate_outputs) + [
        spk.task.ModelOutput(
            name="newton_step_rel",
            target_property="newton_step_ref_rel",
            target_source="prediction",
            loss_fn=torch.nn.MSELoss(),
            loss_weight=1.0,
            metrics={},
        )
    ]
    return spk.train.NewtonSurrogateTask(
        model=student_model,
        outputs=outputs,
        ref_model_path=ref_model_path,
        step_target_key=STEP_KEY,
        step_target_mask_key=MASK_KEY,
        optimizer_cls=torch.optim.AdamW,
        optimizer_args={"lr": 1e-3},
        **kwargs,
    )


def _per_molecule_norm(x, batch):
    return torch.sqrt(
        torch.zeros(batch[properties.n_atoms].shape[0], dtype=x.dtype).index_add(
            0, batch[properties.idx_m], (x**2).sum(-1)
        )
    )


def test_scaled_step_and_target(newton_batch, student_model, surrogate_outputs, ref_model_path):
    task = _task(student_model, surrogate_outputs, ref_model_path, step_floor=0.0)
    batch = _with_target(newton_batch)
    pred = task.predict_without_postprocessing(dict(batch))
    pred = task.augment_predictions(batch, pred)

    step = pred[properties.newton_step]
    target = batch[STEP_KEY].to(step.dtype)
    idx_m, n_atoms = batch[properties.idx_m], batch[properties.n_atoms]
    for m, n in enumerate(n_atoms.tolist()):
        sel = idx_m == m
        s = torch.sqrt((target[sel] ** 2).sum() / (3 * n))
        if batch[MASK_KEY][m] == 0:
            assert torch.all(pred["newton_step_rel"][sel] == 0)
            assert torch.all(pred["newton_step_ref_rel"][sel] == 0)
        else:
            assert torch.allclose(pred["newton_step_rel"][sel], step[sel] / s)
            assert torch.allclose(pred["newton_step_ref_rel"][sel], target[sel] / s)
    # the target is not differentiated through, the step is
    assert pred["newton_step_ref_rel"].grad_fn is None
    assert pred["newton_step_rel"].grad_fn is not None


def test_floor_bounds_the_scale(newton_batch, student_model, surrogate_outputs, ref_model_path):
    task = _task(student_model, surrogate_outputs, ref_model_path, step_floor=0.05)
    batch = _with_target(newton_batch, mask=(1.0, 1.0, 1.0), scale=1e-6)
    pred = task.predict_without_postprocessing(dict(batch))
    pred = task.augment_predictions(batch, pred)
    assert torch.allclose(
        pred["newton_step_rel"], pred[properties.newton_step] / 0.05, rtol=1e-5
    )


def test_mse_is_the_relative_step_error(newton_batch, student_model, surrogate_outputs, ref_model_path):
    task = _task(student_model, surrogate_outputs, ref_model_path, step_floor=0.0)
    batch = _with_target(newton_batch)
    pred = task.predict_without_postprocessing(dict(batch))
    pred = task.augment_predictions(batch, pred)
    mse = torch.nn.functional.mse_loss(pred["newton_step_rel"], pred["newton_step_ref_rel"])

    step, target = pred[properties.newton_step].detach(), batch[STEP_KEY].to(torch.float32)
    n_atoms = batch[properties.n_atoms]
    err = _per_molecule_norm(step - target, batch) ** 2
    norm = _per_molecule_norm(target, batch) ** 2
    mask = batch[MASK_KEY].to(torch.float32)
    # sum over molecules of mask * ||p - p*||^2 / <p*^2>, divided by the number of
    # entries the MSE averages over
    expected = (mask * err / (norm / (3 * n_atoms))).sum() / (3 * n_atoms.sum())
    assert mse.item() == pytest.approx(expected.item(), rel=1e-4)


def test_step_loss_reaches_the_model(newton_batch, student_model, surrogate_outputs, ref_model_path):
    task = _task(student_model, surrogate_outputs, ref_model_path)
    batch = _with_target(newton_batch)
    loss = task.training_step(batch, 0)
    assert torch.isfinite(loss)
    loss.backward()
    assert any(
        p.grad is not None and p.grad.abs().sum() > 0 for p in task.model.parameters()
    )


def test_task_without_step_target_writes_nothing(newton_batch, surrogate_task):
    batch = _with_target(newton_batch)
    pred = surrogate_task.predict_without_postprocessing(dict(batch))
    pred = surrogate_task.augment_predictions(batch, pred)
    assert "newton_step_rel" not in pred and "newton_step_ref_rel" not in pred
