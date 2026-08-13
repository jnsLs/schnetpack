"""Gradients must reach the student through the reference model's Hessian.

The whole setup hinges on a detail that is easy to break silently: ``HVP``
passes ``create_graph=self.training`` to the second ``grad`` call. Get that
wrong and training still runs, losses still look plausible, and the part of the
gradient that comes through the Hessian is simply missing.
"""

import pytest
import torch


def _backward(task, batch):
    task.train()
    torch.set_grad_enabled(True)
    task.zero_grad(set_to_none=True)
    task.training_step(batch, 0).backward()


def test_gradients_reach_every_student_parameter(surrogate_task, newton_batch):
    _backward(surrogate_task, newton_batch)

    without_grad = [
        name
        for name, p in surrogate_task.model.named_parameters()
        if p.requires_grad and (p.grad is None or p.grad.abs().sum() == 0)
    ]
    assert not without_grad, f"no gradient reached: {without_grad}"


def test_gradients_flow_through_the_hessian_not_only_the_damping(
    surrogate_task, newton_batch
):
    """Zeroing the damping must not zero the gradient.

    If it did, the model would only be learning through the lambda*p term and
    the Hessian-vector product would be contributing nothing.
    """
    _backward(surrogate_task, newton_batch)
    with_damping = torch.cat(
        [
            p.grad.flatten()
            for p in surrogate_task.model.parameters()
            if p.grad is not None
        ]
    ).clone()

    # force lambda to zero, leaving (H + 0*I) p = H p
    damping = surrogate_task.model.output_modules[1]
    with torch.no_grad():
        damping.outnet[-1].weight.zero_()
        damping.outnet[-1].bias.zero_()

    _backward(surrogate_task, newton_batch)
    without_damping = torch.cat(
        [
            p.grad.flatten()
            for p in surrogate_task.model.parameters()
            if p.grad is not None
        ]
    )

    assert without_damping.abs().sum() > 0, "no gradient survives without damping"
    assert not torch.allclose(with_damping, without_damping)


def _augmented(task, batch):
    pred = task.predict_without_postprocessing(batch)
    return task.augment_predictions(batch, pred)


def test_hvp_builds_a_graph_in_training(surrogate_task, newton_batch):
    """create_graph=self.training is what lets gradients reach the student."""
    surrogate_task.train()
    torch.set_grad_enabled(True)
    assert _augmented(surrogate_task, newton_batch)["damped_hvp"].grad_fn is not None


def test_eval_gives_the_same_numbers_without_the_second_order_graph(
    surrogate_task, newton_batch
):
    """Eval must produce the same Hessian-vector product, not merely a finite one.

    Not bit for bit: ``create_graph=True`` makes autograd take differentiable
    variants of some backward formulas, which round differently. The point here
    is that eval computes the same quantity, not a degenerate one.
    """
    torch.set_grad_enabled(True)

    surrogate_task.train()
    training = _augmented(surrogate_task, newton_batch)["damped_hvp"].detach().clone()

    surrogate_task.eval()
    evaluating = _augmented(surrogate_task, newton_batch)["damped_hvp"]

    torch.testing.assert_close(evaluating.detach(), training, rtol=1e-5, atol=1e-7)
    assert training.abs().sum() > 0


def test_reference_model_runs_in_train_mode_whatever_the_task_does(
    surrogate_task, reference_model, newton_batch
):
    """Forces gates its own graph on self.training, so the reference must be in
    train mode while it runs -- and must be handed back afterwards."""
    surrogate_task.eval()
    reference_model.eval()

    seen = {}
    forces_module = reference_model.output_modules[1]
    original_forward = forces_module.forward

    def spy(inputs):
        seen["training"] = forces_module.training
        return original_forward(inputs)

    forces_module.forward = spy
    torch.set_grad_enabled(True)
    _augmented(surrogate_task, newton_batch)

    assert seen["training"] is True, "reference forces would come back detached"
    assert reference_model.training is False, "the reference model's mode was not restored"


def test_validation_step_runs_with_grad_globally_disabled(surrogate_task, newton_batch):
    """Lightning disables grad for validation; the second derivative needs it."""
    surrogate_task.eval()
    with torch.no_grad():
        loss = surrogate_task.validation_step(newton_batch, 0)["val_loss"]
    assert torch.isfinite(loss)


def test_augment_needs_the_predictions_first(surrogate_task, newton_batch):
    """The documented ordering constraint, made explicit."""
    torch.set_grad_enabled(True)
    with pytest.raises(KeyError):
        surrogate_task.augment_predictions(newton_batch, {})
