"""Training a Newton-step model against precomputed Hessians with eigenvalue damping.

The counterpart of :mod:`schnetpack.train.surrogate` for datasets that store the
full Hessian, e.g. from DFT. The model is trained on the residual of the damped
Newton system

.. math::
    (H + \\lambda I)\\, p = F ,

with both :math:`H` and :math:`F` read from the batch. Because the Hessian is
materialized, its smallest eigenvalue :math:`e_\\mathrm{min}` is known and the
damping is set from it rather than learned (*eigenvalue damping*):

.. math::
    \\lambda = \\begin{cases}
        0 & e_\\mathrm{min} > 0 \\\\
        -e_\\mathrm{min} + \\epsilon & \\text{otherwise,}
    \\end{cases}

so that :math:`H + \\lambda I` is positive definite and the step points downhill.

The Hessian is a training label, so the product is built on the task and not by
an output module: the exported model only predicts a step.
"""

from typing import Any, Dict, List, Optional, Type

import torch

from schnetpack import properties
from schnetpack.atomistic.response import expand_per_atom
from schnetpack.model.base import AtomisticModel
from schnetpack.task import AtomisticTask, ModelOutput
from schnetpack.units import convert_units

__all__ = ["DenseHessianNewtonTask", "pad_hessians", "pad_vectors"]


def pad_hessians(hessian: torch.Tensor, n_atoms: torch.Tensor) -> torch.Tensor:
    """Unpack the collated Hessians of a batch into a zero-padded dense stack.

    Each molecule's ``(3N, 3N)`` Hessian is read in row-major order from
    ``hessian.reshape(-1)``, so both layouts a dataset can hold are accepted: a
    ``(3N, 3N)`` matrix per molecule, which collates to ``(sum 3N, 3N)`` and
    therefore only for molecules of equal size, and a flat ``(9N²,)`` vector per
    molecule, which collates for any mix of sizes.

    Args:
        hessian: The collated Hessians.
        n_atoms: Number of atoms per molecule, shape ``(n_molecules,)``.

    Returns:
        Shape ``(n_molecules, 3 N_max, 3 N_max)``, zero outside each molecule's block.
    """
    n3 = 3 * n_atoms
    n_mols = n_atoms.shape[0]
    n3_max = int(n3.max())
    sizes = n3 * n3
    if hessian.numel() != int(sizes.sum()):
        raise ValueError(
            f"the batch holds {hessian.numel()} Hessian entries, but its molecules "
            f"need {int(sizes.sum())} (9 N² each)"
        )
    mol = torch.repeat_interleave(torch.arange(n_mols, device=n_atoms.device), sizes)
    start = torch.cumsum(sizes, 0) - sizes
    k = torch.arange(mol.shape[0], device=n_atoms.device) - start[mol]
    row, col = k // n3[mol], k % n3[mol]
    flat = (mol * n3_max + row) * n3_max + col
    dense = hessian.new_zeros(n_mols * n3_max * n3_max)
    return dense.index_put((flat,), hessian.reshape(-1)).reshape(n_mols, n3_max, n3_max)


def pad_vectors(vectors: torch.Tensor, n_atoms: torch.Tensor):
    """Per-atom vectors ``(n_atoms_total, 3)`` as a zero-padded ``(n_molecules, 3 N_max)``.

    Returns:
        The padded vectors and the ``(molecule, position)`` index of every
        Cartesian component, which maps a padded result back to per-atom vectors
        as ``padded[index].reshape(-1, 3)``.
    """
    n3 = 3 * n_atoms
    n_mols = n_atoms.shape[0]
    mol = torch.repeat_interleave(torch.arange(n_mols, device=n_atoms.device), n3)
    start = torch.cumsum(n3, 0) - n3
    k = torch.arange(mol.shape[0], device=n_atoms.device) - start[mol]
    padded = vectors.new_zeros(n_mols, int(n3.max()))
    index = (mol, k)
    return padded.index_put(index, vectors.reshape(-1)), index


class DenseHessianNewtonTask(AtomisticTask):
    """Trains a model to solve the eigenvalue-damped Newton system of stored Hessians.

    The model predicts the step :math:`p`. This task reads the Hessian from the
    batch, sets the damping from its smallest eigenvalue, and adds two entries to
    the predictions:

    * ``damping_factor_key``: :math:`\\lambda`, one per molecule, for logging, and
    * ``damped_hvp_key``: :math:`(H + \\lambda I) p`, the left-hand side.

    The right-hand side is the batch forces, so the residual is an ordinary
    supervised :class:`~schnetpack.task.ModelOutput` on ``damped_hvp_key`` with
    ``target_property="forces"``.

    The damping is computed without gradient: it is a label, not a prediction.
    """

    def __init__(
        self,
        model: AtomisticModel,
        outputs: List[ModelOutput],
        hessian_unit: str,
        damping_offset: float = 0.1,
        damping_offset_unit: str = "Hartree/Bohr/Bohr",
        hessian_key: str = properties.hessian,
        newton_step_key: str = properties.newton_step,
        damping_factor_key: str = properties.damping_factor,
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
            model: The model under training, predicting the step.
            outputs: List of outputs and their loss functions.
            hessian_unit: Unit of the Hessian as it arrives in the batch, i.e.
                after the dataset's unit conversion, e.g. ``"kcal/mol/Ang/Ang"``.
                Only used to convert ``damping_offset``.
            damping_offset: The offset :math:`\\epsilon` added to
                :math:`-e_\\mathrm{min}` when the Hessian is not positive definite.
            damping_offset_unit: Unit of ``damping_offset``. The default
                0.1 Hartree/Bohr² is 9.72 eV/Å².
            hessian_key: Key of the Hessian in the batch.
            newton_step_key: Key of the step predicted by the model.
            damping_factor_key: Key under which the damping is added to the
                predictions.
            damped_hvp_key: Key under which :math:`(H + \\lambda I) p` is added to
                the predictions.
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
        self.hessian_unit = hessian_unit
        self.damping_offset = damping_offset * convert_units(
            damping_offset_unit, hessian_unit
        )
        self.hessian_key = hessian_key
        self.newton_step_key = newton_step_key
        self.damping_factor_key = damping_factor_key
        self.damped_hvp_key = damped_hvp_key

    def eigenvalue_damping(self, hessian: torch.Tensor, n_atoms: torch.Tensor):
        """The damping per molecule: 0 if ``e_min > 0``, else ``-e_min + damping_offset``.

        Args:
            hessian: Padded Hessians from :func:`pad_hessians`.
            n_atoms: Number of atoms per molecule.
        """
        n3 = hessian.shape[-1]
        padded = (
            torch.arange(n3, device=hessian.device)[None, :] >= 3 * n_atoms[:, None]
        )
        # padding gets eigenvalues above all real ones (Gershgorin bound), so the
        # first eigenvalue of the ascending spectrum is the molecule's own e_min
        bound = hessian.abs().sum(-1).amax(-1, keepdim=True) + 1.0
        e_min = torch.linalg.eigvalsh(hessian + torch.diag_embed(padded * bound))[:, 0]
        return torch.where(
            e_min > 0, torch.zeros_like(e_min), -e_min + self.damping_offset
        )

    def augment_predictions(
        self, batch: Dict[str, torch.Tensor], pred: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """Add the damping and the damped Hessian-vector product to the predictions."""
        n_atoms = batch[properties.n_atoms]
        hessian = pad_hessians(batch[self.hessian_key], n_atoms)
        with torch.no_grad():
            damping = self.eigenvalue_damping(hessian, n_atoms)

        step = pred[self.newton_step_key]
        padded_step, index = pad_vectors(step, n_atoms)
        hessian_product = torch.bmm(hessian, padded_step[..., None]).squeeze(-1)
        hessian_product = hessian_product[index].reshape(-1, 3)

        damped_part = step * expand_per_atom(damping, n_atoms)

        pred[self.damping_factor_key] = damping
        pred[self.damped_hvp_key] = hessian_product + damped_part
        return pred
