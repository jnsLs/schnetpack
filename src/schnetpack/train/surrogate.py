"""On-the-fly Newton-step targets from a frozen reference model.

The surrogate setup trains a model to predict the Newton step

.. math::
    p = -(H + \\lambda I)^{-1} \\nabla E

without ever forming or inverting the Hessian. Instead of regressing against
precomputed steps, targets are built during training from a frozen reference
potential: it supplies the forces :math:`F = -\\nabla E` and, through a second
backward pass, the damped Hessian-vector product :math:`(H + \\lambda I) p`.
Driving the two together is equivalent to solving the damped Newton system,
and the damping :math:`\\lambda` is learned alongside the step.
"""

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from schnetpack import properties
from schnetpack.atomistic import Forces, HVP
from schnetpack.model.base import AtomisticModel
from schnetpack.utils import load_model

__all__ = ["NewtonStepTargets", "load_reference_model"]

#: Key of the (identically zero) regression target that pulls the damping
#: factor towards zero. It is expressed as a target rather than as a penalty
#: term so that it can reuse the regular ``ModelOutput`` machinery.
target_damping_factor: str = "target_damping_factor"


def load_reference_model(
    ref_model_path: str,
    newton_step_key: str = properties.newton_step,
    damping_factor_key: str = properties.damping_factor,
    ref_force_key: str = properties.ref_forces,
    damped_hvp_key: str = properties.damped_hvp,
) -> AtomisticModel:
    """Load a trained potential and convert it into a Newton-step target generator.

    The checkpoint is an ordinary energy/forces model. Its :class:`Forces`
    module is replaced by an :class:`~schnetpack.atomistic.HVP`, which keeps
    the forces but additionally contracts the Hessian with the trial step.

    The model is frozen but deliberately left to follow the training mode of
    its parent: ``HVP`` builds a graph for the second derivative only while
    training, which is what lets gradients reach the student.

    Args:
        ref_model_path: Path to the saved reference model.
        newton_step_key: Key of the trial step in the batch.
        damping_factor_key: Key of the damping factor in the batch.
        ref_force_key: Key under which to store the reference forces.
        damped_hvp_key: Key under which to store the damped Hessian-vector product.

    Raises:
        FileNotFoundError: If no file exists at ``ref_model_path``.
        ValueError: If the checkpoint does not contain exactly one ``Forces``
            module to replace.
    """
    if ref_model_path is None:
        raise ValueError(
            "a reference model is required to build Newton-step targets; "
            "set `ref_model_path` (e.g. `task.ref_model_path=/path/to/best_model`)"
        )

    import os

    if not os.path.exists(ref_model_path):
        raise FileNotFoundError(f"no reference model at {ref_model_path!r}")

    ref_model = load_model(ref_model_path, device="cpu")

    force_indices = [
        i for i, m in enumerate(ref_model.output_modules) if isinstance(m, Forces)
    ]
    if len(force_indices) != 1:
        raise ValueError(
            f"expected exactly one Forces module in the reference model to replace "
            f"with an HVP, found {len(force_indices)} in "
            f"{[type(m).__name__ for m in ref_model.output_modules]}"
        )

    ref_model.output_modules[force_indices[0]] = HVP(
        ref_force_key=ref_force_key,
        damped_hvp_key=damped_hvp_key,
        newton_step_key=newton_step_key,
        damping_factor_key=damping_factor_key,
    )
    ref_model.collect_derivatives()
    ref_model.collect_outputs()
    ref_model.do_postprocessing = False

    for parameter in ref_model.parameters():
        parameter.requires_grad = False

    return ref_model


class NewtonStepTargets(nn.Module):
    """Builds damped-Newton-step targets from a frozen reference model.

    Note:
        The student model must run *before* this module: the reference model
        contracts the Hessian with the student's predicted step and damping
        factor, both of which it reads out of the batch.
    """

    def __init__(
        self,
        ref_model_path: str,
        newton_step_key: str = properties.newton_step,
        damping_factor_key: str = properties.damping_factor,
        ref_force_key: str = properties.ref_forces,
        damped_hvp_key: str = properties.damped_hvp,
    ):
        """
        Args:
            ref_model_path: Path to the saved reference model.
            newton_step_key: Key of the step predicted by the student.
            damping_factor_key: Key of the damping factor predicted by the student.
            ref_force_key: Key under which the reference forces are stored.
            damped_hvp_key: Key under which the damped Hessian-vector product
                is stored.
        """
        super().__init__()
        self.newton_step_key = newton_step_key
        self.damping_factor_key = damping_factor_key
        self.ref_force_key = ref_force_key
        self.damped_hvp_key = damped_hvp_key
        self.ref_model = load_reference_model(
            ref_model_path,
            newton_step_key=newton_step_key,
            damping_factor_key=damping_factor_key,
            ref_force_key=ref_force_key,
            damped_hvp_key=damped_hvp_key,
        )

    def forward(
        self, batch: Dict[str, torch.Tensor], pred: Dict[str, torch.Tensor]
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """Run the reference model and return the completed predictions and targets.

        Args:
            batch: The training batch, already passed through the student model.
            pred: The student's predictions.

        Returns:
            ``(pred, targets)``, where ``pred`` additionally carries the damped
            Hessian-vector product and ``targets`` carries the reference forces
            it should match, plus the zero target for the damping factor.
        """
        ref_out = self.ref_model(batch)

        pred = {**pred, self.damped_hvp_key: ref_out[self.damped_hvp_key]}
        targets = {
            # (H + lambda I) p should equal F = -grad E, i.e. p should solve
            # the damped Newton system
            self.ref_force_key: ref_out[self.ref_force_key].detach(),
            # ... using as little damping as it can get away with
            target_damping_factor: torch.zeros_like(pred[self.damping_factor_key]),
        }
        return pred, targets
