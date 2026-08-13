"""Training a Newton-step surrogate against a frozen reference potential.

The surrogate setup trains a model to predict the damped Newton step

.. math::
    p = -(H + \\lambda I)^{-1} \\nabla E

without ever forming or inverting the Hessian. Instead of regressing against
precomputed steps, both sides of the damped Newton system are built during
training by a frozen reference potential: it supplies the forces
:math:`F = -\\nabla E`, and a second backward pass through those forces gives
the damped Hessian-vector product :math:`(H + \\lambda I) p`. Driving the two
together is equivalent to solving the system, and the damping :math:`\\lambda`
is learned alongside the step.

The reference potential belongs to the training procedure rather than to the
model, so it lives on :class:`NewtonSurrogateTask` and not in the student's
``output_modules``. The exported model is therefore exactly what it should be:
a network that predicts a step and a damping factor.

Nothing here modifies the reference potential. It is called as-is and only its
``forces`` output is read, so any model following SchNetPack's
``Dict[str, Tensor]`` interface can be used, whether saved with
:func:`torch.save` or :func:`torch.jit.save`.
"""

import contextlib
import os
import warnings
from typing import Any, Dict, List, Optional, Type

import torch
import torch.nn as nn
from torch.autograd import grad

from schnetpack import properties
from schnetpack.atomistic.response import expand_per_atom
from schnetpack.model.base import AtomisticModel
from schnetpack.task import AtomisticTask, ModelOutput
from schnetpack.utils import load_model

__all__ = ["NewtonSurrogateTask", "load_reference_model", "REF_MODEL_FORMATS"]

#: Accepted values of ``ref_model_format``. ``"auto"`` tries TorchScript first
#: and falls back to a pickled module.
REF_MODEL_FORMATS = ("auto", "torch", "torchscript")

#: Class-name fragments of modules that behave differently in train and eval
#: mode. The reference potential has to run in train mode (see
#: :meth:`NewtonSurrogateTask._reference_in_train_mode`), so any of these
#: changes its predictions and is worth a warning.
_TRAIN_MODE_SENSITIVE = ("Dropout", "BatchNorm", "InstanceNorm")


def _module_class_names(model: nn.Module) -> List[str]:
    """Class names of all submodules, seeing through TorchScript wrappers."""
    return [
        getattr(module, "original_name", type(module).__name__)
        for module in model.modules()
    ]


def load_reference_model(
    ref_model_path: str,
    ref_model_format: str = "auto",
) -> nn.Module:
    """Load a frozen reference potential, without modifying it.

    The model is used exactly as saved: it is called on a batch and its
    ``forces`` output is read. No output module is inspected or replaced, which
    is what allows potentials whose internals are not known to be used here.

    Args:
        ref_model_path: Path to the saved reference model.
        ref_model_format: ``"torchscript"`` to load with :func:`torch.jit.load`,
            ``"torch"`` to load a pickled module, or ``"auto"`` (default) to try
            TorchScript first and fall back to a pickled module.

    Returns:
        The frozen model, with all parameters set to ``requires_grad=False``.

    Raises:
        ValueError: If ``ref_model_path`` is not set, or ``ref_model_format`` is
            not one of :data:`REF_MODEL_FORMATS`.
        FileNotFoundError: If no file exists at ``ref_model_path``.
    """
    if ref_model_format not in REF_MODEL_FORMATS:
        raise ValueError(
            f"unknown ref_model_format {ref_model_format!r}, expected one of "
            f"{REF_MODEL_FORMATS}"
        )

    if ref_model_path is None:
        raise ValueError(
            "a reference model is required to build Newton-step targets; "
            "set `ref_model_path` (e.g. `globals.ref_model_path=/path/to/best_model`)"
        )

    if not os.path.exists(ref_model_path):
        raise FileNotFoundError(f"no reference model at {ref_model_path!r}")

    if ref_model_format == "torchscript":
        ref_model = torch.jit.load(ref_model_path, map_location="cpu")
    elif ref_model_format == "torch":
        ref_model = load_model(ref_model_path, device="cpu")
    else:
        try:
            ref_model = torch.jit.load(ref_model_path, map_location="cpu")
        except Exception:
            ref_model = load_model(ref_model_path, device="cpu")

    for parameter in ref_model.parameters():
        parameter.requires_grad_(False)

    # Postprocessing is for inference, and CastTo64 in particular would return
    # float64 forces that no longer combine with the float32 step.
    try:
        ref_model.do_postprocessing = False
    except (AttributeError, RuntimeError) as err:
        warnings.warn(
            f"could not disable postprocessing on the reference model ({err}). "
            "If it casts or shifts its outputs, the Newton system will be built "
            "from postprocessed forces."
        )

    sensitive = [
        name
        for name in _module_class_names(ref_model)
        if any(fragment in name for fragment in _TRAIN_MODE_SENSITIVE)
    ]
    if sensitive:
        warnings.warn(
            f"the reference model contains train/eval-sensitive modules {sorted(set(sensitive))}. "
            "It has to run in train mode so that its forces stay differentiable, "
            "so these will be active while the Newton-step targets are built."
        )

    return ref_model


class NewtonSurrogateTask(AtomisticTask):
    """Trains a model to solve the damped Newton system of a reference potential.

    The student predicts the step :math:`p` and the damping :math:`\\lambda`.
    This task then runs the frozen reference potential over the same structures
    and adds two more entries to the predictions:

    * ``ref_force_key``: :math:`F = -\\nabla E`, the right-hand side, and
    * ``damped_hvp_key``: :math:`(H + \\lambda I) p`, the left-hand side.

    Both are ordinary predictions from there on, so the usual
    :class:`~schnetpack.task.ModelOutput` machinery pairs them in the loss --
    with ``target_source="prediction"`` on the output whose target is the
    reference forces.

    The reference potential is held as a plain attribute rather than as a
    registered submodule. It is frozen and reconstructible from
    ``ref_model_path``, so keeping it unregistered leaves it out of
    ``state_dict()`` and out of ``parameters()`` (and therefore out of the
    optimizer and out of every checkpoint), at the price of forwarding
    ``_apply`` by hand.
    """

    def __init__(
        self,
        model: AtomisticModel,
        outputs: List[ModelOutput],
        ref_model_path: str,
        ref_model_format: str = "auto",
        ref_model_force_key: str = properties.forces,
        newton_step_key: str = properties.newton_step,
        damping_factor_key: str = properties.damping_factor,
        ref_force_key: str = properties.ref_forces,
        damped_hvp_key: str = properties.damped_hvp,
        optimizer_cls: Type[torch.optim.Optimizer] = torch.optim.Adam,
        optimizer_args: Optional[Dict[str, Any]] = None,
        scheduler_cls: Optional[Type] = None,
        scheduler_args: Optional[Dict[str, Any]] = None,
        scheduler_monitor: Optional[str] = None,
        warmup_steps: int = 0,
    ):
        """
        Args:
            model: The model under training, predicting the step and damping.
            outputs: List of outputs and their loss functions.
            ref_model_path: Path to the frozen reference potential.
            ref_model_format: See :func:`load_reference_model`.
            ref_model_force_key: Key of the forces in the reference model's
                output. Only this one output is read.
            newton_step_key: Key of the step predicted by the model.
            damping_factor_key: Key of the damping factor predicted by the model.
            ref_force_key: Key under which the reference forces are added to the
                predictions.
            damped_hvp_key: Key under which the damped Hessian-vector product is
                added to the predictions.
            optimizer_cls: Type of torch optimizer, e.g. torch.optim.Adam.
            optimizer_args: Dict of optimizer keyword arguments.
            scheduler_cls: Type of learning rate scheduler.
            scheduler_args: Dict of scheduler keyword arguments.
            scheduler_monitor: Name of metric to be observed for ReduceLROnPlateau.
            warmup_steps: Number of steps to linearly increase the learning rate.
        """
        super().__init__(
            model=model,
            outputs=outputs,
            optimizer_cls=optimizer_cls,
            optimizer_args=optimizer_args,
            scheduler_cls=scheduler_cls,
            scheduler_args=scheduler_args,
            scheduler_monitor=scheduler_monitor,
            warmup_steps=warmup_steps,
        )
        self.ref_model_path = ref_model_path
        self.ref_model_format = ref_model_format
        self.ref_model_force_key = ref_model_force_key
        self.newton_step_key = newton_step_key
        self.damping_factor_key = damping_factor_key
        self.ref_force_key = ref_force_key
        self.damped_hvp_key = damped_hvp_key

        self._load_ref_model()

        # The second derivative is taken here rather than by an output module,
        # so the model itself declares no required derivatives and the base
        # class would leave gradients off during validation and testing.
        self.grad_enabled = True

    def _load_ref_model(self) -> None:
        # object.__setattr__ bypasses nn.Module.__setattr__, which would
        # otherwise register the reference potential as a submodule.
        object.__setattr__(
            self,
            "ref_model",
            load_reference_model(self.ref_model_path, self.ref_model_format),
        )

    @contextlib.contextmanager
    def _reference_in_train_mode(self):
        """Run the reference potential in train mode, then restore its mode.

        ``Forces`` builds the graph of its own gradient with
        ``create_graph=self.training``. In eval mode the forces therefore come
        back detached from the positions and the Hessian-vector product cannot
        be taken at all -- so the reference potential runs in train mode even
        while the task is validating. Anything whose forward differs between
        the two modes is reported by :func:`load_reference_model`.
        """
        was_training = self.ref_model.training
        self.ref_model.train(True)
        try:
            yield
        finally:
            self.ref_model.train(was_training)

    def augment_predictions(
        self, batch: Dict[str, torch.Tensor], pred: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """Add both sides of the damped Newton system to the predictions.

        Runs after the model, whose step and damping it needs. The positions
        are made differentiable here rather than before the model runs: the
        model itself takes no derivative of them, and marking them earlier
        would add edges to the backward graph that change the order autograd
        accumulates in -- enough to move gradients in the last bits.
        """
        positions = batch[properties.R]
        positions.requires_grad_()

        with self._reference_in_train_mode():
            ref_out = self.ref_model(batch)

        forces = ref_out[self.ref_model_force_key]
        if forces.grad_fn is None:
            raise RuntimeError(
                f"the reference model's {self.ref_model_force_key!r} are not "
                "differentiable with respect to the positions, so the "
                "Hessian-vector product cannot be taken. This happens when the "
                "model was exported for inference (frozen, or optimized for "
                "inference), when it predicts forces directly instead of as "
                "-dE/dR, or when gradients are globally disabled."
            )

        step = pred[self.newton_step_key]
        damping = pred[self.damping_factor_key]

        # F = -dE/dR, so d(F.p)/dR = -H p. create_graph must be True in
        # training: it is what lets gradients reach the model that produced the
        # step and the damping. In eval mode there is no backward pass, so the
        # graph is not needed.
        hessian_product = -grad(
            forces,
            positions,
            step,
            create_graph=self.training,
            retain_graph=True,
        )[0]

        damped_part = step * expand_per_atom(damping, batch[properties.n_atoms])

        pred[self.ref_force_key] = forces
        pred[self.damped_hvp_key] = hessian_product + damped_part
        return pred

    def _apply(self, fn, *args, **kwargs):
        """Keep the unregistered reference potential on the task's device/dtype."""
        super()._apply(fn, *args, **kwargs)
        self.ref_model._apply(fn, *args, **kwargs)
        return self

    def __getstate__(self):
        # Pickled by value the reference potential would be written into every
        # checkpoint. It is frozen and lives on disk, so store the path instead.
        state = super().__getstate__()
        state.pop("ref_model", None)
        return state

    def __setstate__(self, state):
        super().__setstate__(state)
        self._load_ref_model()
