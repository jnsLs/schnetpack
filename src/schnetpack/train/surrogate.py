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

Where the damping comes from is set by ``damping_source``:

* ``"predicted"`` (default): the student predicts :math:`\\lambda` (the
  ``DampingFactor`` head), as in the paper's learned damping.
* ``"batch"``: eigenvalue damping from a per-structure smallest eigenvalue
  stored in the dataset, :math:`\\lambda = \\mathrm{relu}(-e_{\\min}) +
  \\varepsilon`.
* ``"lanczos"``: the same rule, with :math:`e_{\\min}` computed on the fly by
  Lanczos on the reference potential's Hessian-vector products
  (:mod:`schnetpack.train.lanczos`), without forming the Hessian.
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
from schnetpack.train.lanczos import lanczos_min_eigenvalue, remove_molecule_mean
from schnetpack.units import convert_units
from schnetpack.utils import load_model

__all__ = [
    "NewtonSurrogateTask",
    "load_reference_model",
    "REF_MODEL_FORMATS",
    "DAMPING_SOURCES",
]

#: Accepted values of ``ref_model_format``. ``"auto"`` tries TorchScript first
#: and falls back to a pickled module.
REF_MODEL_FORMATS = ("auto", "torch", "torchscript")

#: Accepted values of ``damping_source`` of :class:`NewtonSurrogateTask`.
DAMPING_SOURCES = ("predicted", "batch", "lanczos")

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

    The student predicts the step :math:`p` and, with the default
    ``damping_source="predicted"``, the damping :math:`\\lambda` (otherwise the
    task supplies :math:`\\lambda` by eigenvalue damping, see ``damping_source``).
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
        damping_source: str = "predicted",
        damping_eps: float = 0.0,
        damping_eps_unit: Optional[str] = None,
        damping_unit: Optional[str] = None,
        emin_key: str = "e_min",
        lanczos_steps: int = 20,
        lanczos_seed: Optional[int] = 0,
        lanczos_project_translations: bool = True,
        remove_teacher_net_force: bool = False,
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
            damping_source: Where :math:`\\lambda` comes from, one of
                :data:`DAMPING_SOURCES`. ``"predicted"`` reads the student's
                ``damping_factor_key`` output. ``"batch"`` and ``"lanczos"``
                use eigenvalue damping, :math:`\\lambda = \\mathrm{relu}(-e_{\\min})
                + \\varepsilon`, with :math:`e_{\\min}` read from
                ``batch[emin_key]`` or computed by Lanczos on the reference
                potential. In those two modes the student must not predict
                ``damping_factor_key`` itself: the task writes the damping it
                used there (detached), and :math:`e_{\\min}` under ``emin_key``,
                so both can be logged as metrics.
            damping_eps: :math:`\\varepsilon`, in the energy/length² units of
                the reference model's forces (e.g. kcal/mol/Å²). Ignored for
                ``"predicted"``. The paper's 0.1 Ha/Bohr² is 9.717 eV/Å² or
                224.1 kcal/mol/Å².
            damping_eps_unit, damping_unit: Optional unit conversion of
                ``damping_eps``: if both are given (e.g. ``"eV/Ang/Ang"`` and
                ``"kcal/mol/Ang/Ang"``), ``damping_eps`` is read in
                ``damping_eps_unit`` and converted to ``damping_unit``, which must
                be the energy/length² unit of the reference forces.
            emin_key: Batch key of the per-structure smallest Hessian eigenvalue
                for ``"batch"`` (same units as ``damping_eps``), and the
                prediction key :math:`e_{\\min}` is written to in both
                eigenvalue modes. Whether :math:`e_{\\min}` was computed with
                the translations projected out (as the Lanczos precompute does)
                or from the raw Hessian makes no difference to the damping: the
                raw Hessian's extra eigenvalues are exact zeros.
            lanczos_steps: Lanczos steps (= extra reference backward passes per
                batch) for ``"lanczos"``.
            lanczos_seed: Seed of the Lanczos start vector.
            lanczos_project_translations: Run Lanczos on :math:`P H P`, the
                Hessian with the three translations projected out. See
                :mod:`schnetpack.train.lanczos`.
            remove_teacher_net_force: Subtract each molecule's mean from the
                reference forces before they are used, on both sides of the
                Newton system. For direct-force teachers (e.g. MD-ET) the net
                force is not zero, and with small damping it would produce a
                rigid translation of about :math:`F_{net} / \\lambda` in the
                step. The subtraction is differentiable, so the Hessian-vector
                product becomes :math:`-J^T P p` and ignores the step's own
                translation. A no-op (up to rounding) for conservative
                teachers.
        """
        if damping_source not in DAMPING_SOURCES:
            raise ValueError(
                f"unknown damping_source {damping_source!r}, expected one of "
                f"{DAMPING_SOURCES}"
            )
        if lanczos_steps < 1:
            raise ValueError(f"lanczos_steps must be >= 1, got {lanczos_steps}")
        if (damping_eps_unit is None) != (damping_unit is None):
            raise ValueError(
                "give both damping_eps_unit and damping_unit, or neither "
                f"(got {damping_eps_unit!r} and {damping_unit!r})"
            )
        # Converted into a new name: save_hyperparameters() records this
        # frame's arguments, and the task is rebuilt from them when a checkpoint
        # is loaded -- rebinding damping_eps here would convert it twice.
        eps_in_damping_unit = float(damping_eps)
        if damping_eps_unit is not None:
            eps_in_damping_unit *= convert_units(damping_eps_unit, damping_unit)

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
        self.damping_source = damping_source
        self.damping_eps = eps_in_damping_unit
        self.emin_key = emin_key
        self.lanczos_steps = int(lanczos_steps)
        self.lanczos_seed = lanczos_seed
        self.lanczos_project_translations = lanczos_project_translations
        self.remove_teacher_net_force = remove_teacher_net_force

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

        if self.remove_teacher_net_force:
            forces = remove_molecule_mean(
                forces, batch[properties.idx_m], batch[properties.n_atoms]
            )

        step = pred[self.newton_step_key]
        damping = self._damping(batch, pred, forces, positions)

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

    def _damping(
        self,
        batch: Dict[str, torch.Tensor],
        pred: Dict[str, torch.Tensor],
        forces: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        """The per-molecule damping, from the source chosen at construction.

        For the eigenvalue sources the damping is a constant of the loss: it is
        detached, and written to the predictions (with :math:`e_{\\min}`) only so
        that it can be logged.
        """
        if self.damping_source == "predicted":
            return pred[self.damping_factor_key]

        if self.damping_factor_key in pred:
            raise ValueError(
                f"damping_source={self.damping_source!r} computes the damping, "
                f"but the model also predicts {self.damping_factor_key!r}. "
                "Remove the DampingFactor head from model.output_modules (or "
                "give it another output_key)."
            )

        n_mol = batch[properties.n_atoms].shape[0]
        if self.damping_source == "batch":
            if self.emin_key not in batch:
                raise KeyError(
                    f"damping_source='batch' needs {self.emin_key!r} in the batch; "
                    f"load it with data.dataset.load_properties=[{self.emin_key}] "
                    "(and convert it to the model's energy/length^2 units with "
                    "data.dataset.property_units)"
                )
            e_min = batch[self.emin_key].detach().reshape(-1)
            if e_min.shape[0] != n_mol:
                raise ValueError(
                    f"batch[{self.emin_key!r}] has {e_min.shape[0]} entries for "
                    f"{n_mol} molecules; expected one scalar per structure"
                )
        else:
            e_min = self._lanczos_e_min(batch, forces, positions)

        e_min = e_min.to(dtype=forces.dtype, device=forces.device)
        damping = torch.relu(-e_min) + self.damping_eps
        pred[self.emin_key] = e_min
        pred[self.damping_factor_key] = damping
        return damping

    def _lanczos_e_min(
        self,
        batch: Dict[str, torch.Tensor],
        forces: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        """Smallest eigenvalue of every molecule's reference Hessian, by Lanczos.

        Each matvec is one more backward pass through the reference graph that
        is already there, without ``create_graph``: nothing flows back into the
        student from here. Reverse mode yields :math:`H^T v`; see
        :mod:`schnetpack.train.lanczos` for what that means for direct-force
        teachers.
        """

        def hvp(vector: torch.Tensor) -> torch.Tensor:
            return -grad(
                forces,
                positions,
                vector.to(forces.dtype),
                retain_graph=True,
            )[0]

        result = lanczos_min_eigenvalue(
            hvp,
            batch[properties.idx_m],
            batch[properties.n_atoms],
            k=self.lanczos_steps,
            project_translations=self.lanczos_project_translations,
            seed=self.lanczos_seed,
        )
        return result.e_min

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
