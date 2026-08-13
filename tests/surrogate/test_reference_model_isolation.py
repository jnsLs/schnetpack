"""The reference model must not leak into the optimiser, the checkpoints or the export.

It belongs to the training procedure, not to the model, so it sits on the task.
Keeping it there as an *unregistered* attribute is what stops its frozen
parameters reaching the optimiser and a full second potential being written into
every checkpoint, while ``_apply`` forwarding keeps it on the task's device.
"""

import pickle

import pytest
import torch


def _ref_parameter_ids(task):
    return {id(p) for p in task.ref_model.parameters()}


def test_reference_parameters_are_frozen(reference_model):
    assert not any(p.requires_grad for p in reference_model.parameters())


def test_reference_parameters_stay_out_of_the_optimizer(surrogate_task):
    optimizer = surrogate_task.configure_optimizers()
    optimised = {id(p) for group in optimizer.param_groups for p in group["params"]}

    assert not (optimised & _ref_parameter_ids(surrogate_task))
    assert optimised, "the optimiser was left with no parameters at all"
    assert {id(p) for p in surrogate_task.model.parameters()} <= optimised


def test_checkpoint_excludes_the_reference_model(surrogate_task):
    """No hook needed: an unregistered attribute is never in the state_dict."""
    state_dict = surrogate_task.state_dict()

    assert not [k for k in state_dict if "ref_model" in k]
    assert [k for k in state_dict if k.startswith("model.")]


def test_hyperparameters_carry_the_path_not_the_model(surrogate_task):
    """The path is a plain string, so hparams stay small and picklable."""
    assert surrogate_task.hparams["ref_model_path"] == surrogate_task.ref_model_path
    assert isinstance(surrogate_task.hparams["ref_model_path"], str)


def test_pickling_stores_the_reference_model_by_path(surrogate_task):
    assert "ref_model" not in surrogate_task.__getstate__()

    restored = pickle.loads(pickle.dumps(surrogate_task))
    assert restored.ref_model is not surrogate_task.ref_model
    for original, reloaded in zip(
        surrogate_task.ref_model.parameters(), restored.ref_model.parameters()
    ):
        assert torch.equal(original, reloaded)


def test_reference_model_still_follows_the_task_device(surrogate_task):
    """Unregistered means .to() would skip it, so _apply forwards by hand."""
    surrogate_task.to(torch.float64)
    assert all(p.dtype == torch.float64 for p in surrogate_task.ref_model.parameters())


def test_checkpoint_round_trips_through_load_from_checkpoint(surrogate_task, tmp_path):
    """cli.py reloads the best task this way, so it has to keep working."""
    import schnetpack as spk

    checkpoint_path = tmp_path / "best.ckpt"
    checkpoint = {
        "state_dict": dict(surrogate_task.state_dict()),
        "hyper_parameters": dict(surrogate_task.hparams),
        "pytorch-lightning_version": "2.0.0",
        "global_step": 0,
        "epoch": 0,
        "loops": {},
        "callbacks": {},
        "optimizer_states": [],
        "lr_schedulers": [],
    }
    surrogate_task.on_save_checkpoint(checkpoint)
    torch.save(checkpoint, checkpoint_path)

    reloaded = spk.train.NewtonSurrogateTask.load_from_checkpoint(
        checkpoint_path, weights_only=False
    )

    for original, restored in zip(
        surrogate_task.model.parameters(), reloaded.model.parameters()
    ):
        assert torch.equal(original, restored)
    for original, restored in zip(
        surrogate_task.ref_model.parameters(), reloaded.ref_model.parameters()
    ):
        assert torch.equal(original, restored)


def test_export_contains_only_the_trained_modules(surrogate_task, tmp_path):
    """The exported model never held the reference potential in the first place."""
    import schnetpack as spk
    from schnetpack import properties

    path = tmp_path / "best_model"
    surrogate_task.save_model(str(path), do_postprocessing=True)
    exported = torch.load(path, weights_only=False)

    assert [type(m).__name__ for m in exported.output_modules] == [
        "NewtonStep",
        "DampingFactor",
    ]
    assert properties.damped_hvp not in exported.model_outputs
    assert properties.newton_step in exported.model_outputs
    assert properties.damping_factor in exported.model_outputs

    # the live task is left exactly as it was
    assert surrogate_task.model.do_postprocessing is True


def test_missing_reference_model_fails_loudly(
    student_model, surrogate_outputs, tmp_path
):
    """A path that does not resolve must say so, not crash inside torch.load."""
    import schnetpack as spk

    with pytest.raises(FileNotFoundError, match="no reference model"):
        spk.train.NewtonSurrogateTask(
            model=student_model,
            outputs=surrogate_outputs,
            ref_model_path=str(tmp_path / "does_not_exist"),
            optimizer_args={"lr": 1e-3},
        )


def test_unset_reference_model_fails_loudly(student_model, surrogate_outputs):
    import schnetpack as spk

    with pytest.raises(ValueError, match="ref_model_path"):
        spk.train.NewtonSurrogateTask(
            model=student_model,
            outputs=surrogate_outputs,
            ref_model_path=None,
            optimizer_args={"lr": 1e-3},
        )
