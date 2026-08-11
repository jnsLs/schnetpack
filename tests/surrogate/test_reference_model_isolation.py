"""The reference model must not leak into the optimiser or the checkpoints.

It is a submodule of the task, so that Lightning moves it to the right device
along with everything else. That convenience used to come at a price: its
frozen parameters were handed to the optimiser and a full copy of it was
written into every checkpoint.
"""

import pytest
import torch


def _ref_parameter_ids(task):
    return {id(p) for p in task.ref_model.parameters()}


def test_reference_parameters_are_frozen(surrogate_task):
    assert not any(p.requires_grad for p in surrogate_task.ref_model.parameters())


def test_reference_parameters_stay_out_of_the_optimizer(surrogate_task):
    optimizer = surrogate_task.configure_optimizers()
    optimised = {id(p) for group in optimizer.param_groups for p in group["params"]}

    assert not (optimised & _ref_parameter_ids(surrogate_task))
    assert optimised, "the optimiser was left with no parameters at all"
    assert {id(p) for p in surrogate_task.model.parameters()} <= optimised


def test_checkpoint_excludes_the_reference_model(surrogate_task):
    checkpoint = {"state_dict": dict(surrogate_task.state_dict())}
    surrogate_task.on_save_checkpoint(checkpoint)

    assert not [k for k in checkpoint["state_dict"] if "ref_model" in k]
    assert [k for k in checkpoint["state_dict"] if k.startswith("model.")]


def test_checkpoint_can_be_loaded_back(surrogate_task, ref_model_path, tmp_path):
    """A checkpoint without the reference model must still load."""
    checkpoint = {"state_dict": dict(surrogate_task.state_dict())}
    surrogate_task.on_save_checkpoint(checkpoint)

    missing, unexpected = surrogate_task.load_state_dict(
        checkpoint["state_dict"], strict=False
    )
    assert not unexpected
    assert all("ref_model" in key for key in missing)


def test_reference_model_still_follows_the_task_device(surrogate_task):
    """Keeping it a submodule is the point: .to() must still reach it."""
    surrogate_task.to(torch.float64)
    assert all(p.dtype == torch.float64 for p in surrogate_task.ref_model.parameters())


def test_checkpoint_round_trips_through_load_from_checkpoint(
    surrogate_task, newton_batch, tmp_path
):
    """cli.py reloads the best task this way, so it has to keep working.

    Dropping the reference model from the checkpoint would otherwise make the
    strict load fail on the missing keys.
    """
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

    reloaded = spk.task.AtomisticTaskSurrogate.load_from_checkpoint(
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


def test_missing_reference_model_fails_loudly(
    tmp_path, student_model, surrogate_outputs
):
    """A path that does not resolve must say so, not crash inside torch.load."""
    import schnetpack as spk

    with pytest.raises(FileNotFoundError, match="no reference model"):
        spk.task.AtomisticTaskSurrogate(
            model=student_model,
            outputs=surrogate_outputs,
            optimizer_args={"lr": 1e-3},
            ref_model_path=str(tmp_path / "does_not_exist"),
        )


def test_unset_reference_model_fails_loudly(student_model, surrogate_outputs):
    import schnetpack as spk

    with pytest.raises(ValueError, match="ref_model_path"):
        spk.task.AtomisticTaskSurrogate(
            model=student_model,
            outputs=surrogate_outputs,
            optimizer_args={"lr": 1e-3},
            ref_model_path=None,
        )
