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


def test_hvp_builds_a_graph_in_training_but_not_in_eval(surrogate_task, newton_batch):
    """create_graph=self.training is deliberate; pin both halves of it."""
    hvp = surrogate_task.ref_model.output_modules[1]

    surrogate_task.train()
    assert hvp.training
    torch.set_grad_enabled(True)
    pred = surrogate_task.predict_without_postprocessing(newton_batch)
    pred, _ = surrogate_task.targets(newton_batch, pred)
    assert pred["damped_hvp"].grad_fn is not None

    surrogate_task.eval()
    assert not hvp.training, "eval() must reach the reference model"


def test_validation_step_runs_with_grad_globally_disabled(surrogate_task, newton_batch):
    """Lightning disables grad for validation; the second derivative needs it."""
    surrogate_task.eval()
    with torch.no_grad():
        loss = surrogate_task.validation_step(newton_batch, 0)["val_loss"]
    assert torch.isfinite(loss)


def test_reference_model_forward_needs_the_student_first(surrogate_task, newton_batch):
    """The documented ordering constraint, made explicit."""
    with pytest.raises(KeyError):
        surrogate_task.ref_model(newton_batch)
