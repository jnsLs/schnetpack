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
    # student predictions
    "newton_step": properties.newton_step,
    "damping_factor": properties.damping_factor,
    # reference-model outputs
    "ref_forces": properties.ref_forces,  # F = -grad E
    "damped_hvp": properties.damped_hvp,  # (H + lambda I) p
    # regression targets built by the task
    "target_ref_forces": properties.ref_forces,
    "target_damping_factor": "target_damping_factor",
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

    # ---- forward passes -------------------------------------------------
    pred = task.predict_without_postprocessing(batch)
    golden["newton_step"] = pred[KEYS["newton_step"]].detach().clone()
    golden["damping_factor"] = pred[KEYS["damping_factor"]].detach().clone()

    _ = task.ref_model(batch)
    golden["ref_forces"] = batch[KEYS["ref_forces"]].detach().clone()
    golden["damped_hvp"] = batch[KEYS["damped_hvp"]].detach().clone()

    targets = {
        KEYS["target_ref_forces"]: batch[KEYS["ref_forces"]].detach(),
        KEYS["target_damping_factor"]: torch.zeros_like(batch[KEYS["damping_factor"]]),
    }

    # ---- per-output losses and the composite loss -----------------------
    for output in task.outputs:
        contribution = output.calculate_loss(batch, targets)
        golden[f"loss/{OUTPUT_LABELS[output.name]}"] = (
            torch.as_tensor(contribution).detach().clone()
        )
    loss = task.loss_fn(batch, targets)
    golden["loss/total"] = loss.detach().clone()

    # ---- metrics --------------------------------------------------------
    for output in task.outputs:
        for name, metric in output.metrics["train"].items():
            metric.reset()
            metric(batch[output.name], targets[output.target_property])
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
    optimizer = task.optimizer_cls(params=task.parameters(), **task.optimizer_kwargs)

    for step in range(N_OPTIM_STEPS):
        optimizer.zero_grad(set_to_none=True)
        pred = task.predict_without_postprocessing(batch)
        _ = task.ref_model(batch)
        targets = {
            KEYS["target_ref_forces"]: batch[KEYS["ref_forces"]].detach(),
            KEYS["target_damping_factor"]: torch.zeros_like(
                batch[KEYS["damping_factor"]]
            ),
        }
        loss = task.loss_fn(batch, targets)
        loss.backward()
        optimizer.step()
        out[f"optim/step{step}/loss"] = loss.detach().clone()

    for name, param in task.model.named_parameters():
        out[f"optim/param/{name}"] = param.detach().clone()
    return out
