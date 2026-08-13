"""Shared driver for the Newton-step surrogate golden regression test.

``collect_golden`` runs the full surrogate pipeline once and returns every
quantity worth pinning, keyed by *semantic* name. The key mapping at the top is
the only place that knows the concrete tensor keys, so renaming those in the
source only requires editing :data:`KEYS` -- the recorded golden file stays
valid across the rename.
"""

from typing import Dict

import torch

from schnetpack import properties

#: semantic name -> the key the pipeline actually uses.
KEYS = {
    # predictions of the trained modules
    "newton_step": properties.newton_step,
    "damping_factor": properties.damping_factor,
    # outputs of the frozen reference potential
    "ref_forces": properties.ref_forces,  # F = -grad E
    "damped_hvp": properties.damped_hvp,  # (H + lambda I) p
}

#: concrete ModelOutput name -> stable label used in the golden file, so that
#: renaming an output does not invalidate the recorded values.
OUTPUT_LABELS = {
    KEYS["damped_hvp"]: "damped_hvp",
    KEYS["damping_factor"]: "damping_factor",
    KEYS["newton_step"]: "newton_step",
}

N_OPTIM_STEPS = 3


def collect_golden(task, batch) -> Dict[str, torch.Tensor]:
    """Run one train step, one val step and a short optimisation, recording all of it."""
    golden: Dict[str, torch.Tensor] = {}

    task.train()
    torch.set_grad_enabled(True)

    # ---- forward pass ---------------------------------------------------
    # One pass now covers both models: the reference potential is an output
    # module of the model under training.
    targets = task._collect_targets(batch)
    pred = task.predict_without_postprocessing(batch)
    pred = task.augment_predictions(batch, pred)
    for name, key in KEYS.items():
        golden[name] = pred[key].detach().clone()

    targets.update(task._collect_predicted_targets(pred))

    # ---- per-output losses and the composite loss -----------------------
    for output in task.outputs:
        contribution = output.calculate_loss(pred, targets)
        golden[f"loss/{OUTPUT_LABELS[output.name]}"] = (
            torch.as_tensor(contribution).detach().clone()
        )
    loss = task.loss_fn(pred, targets)
    golden["loss/total"] = loss.detach().clone()

    # ---- metrics --------------------------------------------------------
    for output in task.outputs:
        for metric in output.metrics["train"].values():
            metric.reset()
        output.update_metrics(pred, targets, "train")
        for name, metric in output.metrics["train"].items():
            golden[f"metric/{OUTPUT_LABELS[output.name]}/{name}"] = (
                metric.compute().detach().clone()
            )
            metric.reset()

    # ---- gradients ------------------------------------------------------
    # The strongest assertion in the suite: it exercises the double backward
    # through HVP and therefore the create_graph=self.training semantics.
    task.zero_grad(set_to_none=True)
    loss.backward()
    for name, param in task.model.named_parameters():
        if param.grad is not None:
            golden[f"grad/{name}"] = param.grad.detach().clone()

    # ---- a few optimisation steps ---------------------------------------
    golden.update(_optimise(task, batch))
    return golden


def _optimise(task, batch) -> Dict[str, torch.Tensor]:
    """Run ``N_OPTIM_STEPS`` AdamW updates and record the resulting parameters."""
    out: Dict[str, torch.Tensor] = {}
    torch.manual_seed(1234)
    optimizer = task.configure_optimizers()

    for step in range(N_OPTIM_STEPS):
        optimizer.zero_grad(set_to_none=True)
        targets = task._collect_targets(batch)
        pred = task.predict_without_postprocessing(batch)
        pred = task.augment_predictions(batch, pred)
        targets.update(task._collect_predicted_targets(pred))
        loss = task.loss_fn(pred, targets)
        loss.backward()
        optimizer.step()
        out[f"optim/step{step}/loss"] = loss.detach().clone()

    for name, param in task.model.named_parameters():
        out[f"optim/param/{name}"] = param.detach().clone()
    return out
