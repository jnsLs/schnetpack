"""Exporting a trained model to a TorchScript archive.

Two consumers want a compiled model, and they want different things from it:

* **LAMMPS** runs the model through the ``pair_schnetpack`` pair style, which
  reads the cutoff out of the archive's extra files and expects postprocessing
  to be applied -- minus the dtype casts, which it does itself. See
  ``docs/howtos/lammps.rst``.
* **The Newton-step surrogate** (:mod:`schnetpack.train.surrogate`) uses the
  archive as a frozen reference potential. It reads only the forces, and it
  differentiates *through* them a second time to build the Hessian-vector
  product, so the export has to keep its backward graph intact.

The second requirement is the reason nothing here freezes the scripted module:
:func:`torch.jit.freeze` and :func:`torch.jit.optimize_for_inference` drop the
graph that the second derivative needs, and the failure only surfaces once
training has already started.

"""

import warnings
from copy import deepcopy
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from ase import Atoms

import schnetpack.properties as properties
from schnetpack.transform import AddOffsets, CastTo32, CastTo64, MatScipyNeighborList

__all__ = ["export_for_lammps", "export_for_surrogate"]

#: Cutoff used to build the smoke-test batch when the model does not carry one.
DEFAULT_CHECK_CUTOFF = 5.0


def _model_cutoff(model: nn.Module) -> Optional[float]:
    """The representation's cutoff, if it has one."""
    cutoff = getattr(getattr(model, "representation", None), "cutoff", None)
    if cutoff is None:
        return None
    return float(cutoff)


def export_for_lammps(model: nn.Module, model_path: str) -> None:
    """Script a model and save it for use with the LAMMPS pair style.

    Postprocessing is kept, since LAMMPS expects the model's own energy shifts
    to be applied, but the dtype casts are dropped: the pair style hands over
    float32 tensors and wants float32 back.

    Args:
        model: The trained model. It is not modified; a copy is scripted.
        model_path: Where to write the TorchScript archive.
    """
    model = deepcopy(model)

    jit_postprocessors = nn.ModuleList()
    for postprocessor in model.postprocessors:
        # ignore type casting
        if type(postprocessor) in [CastTo64, CastTo32]:
            continue
        # ensure offset mean is float
        if type(postprocessor) == AddOffsets:
            postprocessor.mean = postprocessor.mean.float()

        jit_postprocessors.append(postprocessor)
    model.postprocessors = jit_postprocessors

    jit_model = torch.jit.script(model)

    # the pair style reads the cutoff out of the archive to size its neighbor
    # lists. A representation without one is not usable from LAMMPS, but that
    # is for LAMMPS to complain about -- exporting is still useful.
    metadata = dict()
    cutoff = _model_cutoff(jit_model)
    if cutoff is None:
        warnings.warn(
            "the model's representation has no `cutoff`, so no cutoff metadata "
            "was written into the archive. The LAMMPS pair style reads that "
            "field and will not be able to load this model."
        )
    else:
        metadata["cutoff"] = str(cutoff).encode("ascii")

    torch.jit.save(jit_model, model_path, _extra_files=metadata)


def export_for_surrogate(
    model: nn.Module,
    model_path: str,
    force_key: str = properties.forces,
    check: bool = True,
    check_cutoff: Optional[float] = None,
) -> None:
    """Script a potential and save it for use as a frozen reference potential.

    The archive is exported in the state
    :func:`~schnetpack.train.surrogate.load_reference_model` would put it in
    anyway: postprocessing removed and switched off. Postprocessing is an
    inference-time concern, and ``CastTo64`` in particular would return float64
    forces that no longer combine with the float32 step. Dropping ``AddOffsets``
    along with it is harmless -- a per-atom energy offset is constant in the
    positions and so does not reach the forces.

    Args:
        model: The trained potential. It is not modified; a copy is scripted.
        model_path: Where to write the TorchScript archive.
        force_key: Key of the forces in the model's output. This is the only
            output the surrogate task reads, so it has to be there.
        check: If True, run the exported model on a small molecule and take the
            second derivative through its forces, the way the surrogate task
            will. Cheap, and it turns a failure that would otherwise appear at
            the first training step into one at export time.
        check_cutoff: Cutoff for the neighbor list of the smoke-test batch.
            Defaults to the representation's cutoff, or
            :data:`DEFAULT_CHECK_CUTOFF` if it has none.

    Raises:
        ValueError: If the model does not declare ``force_key`` as an output.
        RuntimeError: If ``check`` is set and the exported forces turn out not
            to be differentiable with respect to the positions.
    """
    model = deepcopy(model)

    model_outputs = getattr(model, "model_outputs", None)
    if model_outputs is not None and force_key not in model_outputs:
        raise ValueError(
            f"the model does not output {force_key!r} (it outputs {sorted(model_outputs)}), "
            "but that is the only output the Newton-step surrogate reads. Export a "
            "potential with a `Forces` response module, or point "
            "`task.ref_model_force_key` at the key it does use."
        )

    # matches what load_reference_model does at load time
    model.postprocessors = nn.ModuleList()
    model.do_postprocessing = False

    # deliberately not frozen: freezing drops the backward graph that the
    # Hessian-vector product is taken through.
    jit_model = torch.jit.script(model)

    if check:
        if check_cutoff is None:
            check_cutoff = _model_cutoff(model) or DEFAULT_CHECK_CUTOFF
        _check_double_backward(jit_model, force_key, check_cutoff)

    torch.jit.save(jit_model, model_path)


def _check_double_backward(jit_model: nn.Module, force_key: str, cutoff: float) -> None:
    """Run the exported model the way the surrogate task will, on one molecule.

    The reference potential has to run in train mode: ``Forces`` builds the
    graph of its own gradient with ``create_graph=self.training``, so in eval
    mode the forces come back detached and no second derivative can be taken.
    """
    from schnetpack.interfaces import AtomsConverter

    converter = AtomsConverter(
        neighbor_list=MatScipyNeighborList(cutoff=cutoff),
        dtype=torch.float32,
    )
    batch = converter(
        Atoms(
            numbers=[8, 1, 1],
            positions=np.array(
                [[0.0, 0.0, 0.0], [0.0, 0.757, 0.587], [0.0, -0.757, 0.587]]
            ),
        )
    )

    was_training = jit_model.training
    jit_model.train(True)
    try:
        with torch.enable_grad():
            positions = batch[properties.R]
            positions.requires_grad_()
            forces = jit_model(batch)[force_key]

            if forces.grad_fn is None:
                raise RuntimeError(
                    f"the exported model's {force_key!r} are not differentiable with "
                    "respect to the positions, so the Newton-step surrogate cannot "
                    "take the Hessian-vector product. This happens when the model "
                    "predicts forces directly instead of as -dE/dR, or when it was "
                    "already frozen or optimized for inference. Pass check=False "
                    "(`--no-check`) to export anyway."
                )

            hessian_product = torch.autograd.grad(
                forces, positions, torch.ones_like(forces), create_graph=False
            )[0]
    finally:
        jit_model.train(was_training)

    if not torch.isfinite(hessian_product).all():
        raise RuntimeError(
            "the exported model produced a non-finite Hessian-vector product on a "
            "water molecule. The archive would train against garbage targets."
        )
