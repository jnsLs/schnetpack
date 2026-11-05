from typing import Dict

import torch
import torch.nn as nn

import schnetpack.properties as properties

__all__ = ["ForceInput"]


class ForceInput(nn.Module):
    """Prepare forces as an input to the representation.

    The representation should see the forces at every scale, from MD frames to
    structures near a minimum, where they are orders of magnitude smaller. The
    forces are therefore split into a per-molecule scale

    .. math::
        s = \\sqrt{\\langle F^2 \\rangle + \\mathrm{floor}^2}

    (mean over atoms and components) and the dimensionless direction
    :math:`F / s`, written under ``direction_key`` as a vector per atom. Two
    scalars per atom, :math:`\\log s` and :math:`\\log |F_i / s|`, are written
    under ``scalars_key``, so the scale is still available. Structures with an
    rms force component below the floor are treated as if they were at it.

    The direction is what makes this input different from the geometry alone:
    near a minimum the step is :math:`(H + \\lambda I)^{-1} F`, and from the
    geometry a student has to infer :math:`F` to within its own force error,
    which the forces there fall below.

    Args:
        force_key: Batch key of the forces, in the model's energy/distance
            units (e.g. the frozen reference potential's forces).
        floor: The floor of the scale, in the same units.
        direction_key: Key under which :math:`F / s` is written.
        scalars_key: Key under which the two scalars are written.
        scale_key: Key under which :math:`s` (one per molecule) is written.
    """

    def __init__(
        self,
        force_key: str = "input_forces",
        floor: float = 0.02306,
        direction_key: str = "force_input_dir",
        scalars_key: str = "force_input_scalars",
        scale_key: str = "force_input_scale",
        eps: float = 1e-6,
    ):
        super().__init__()
        self.force_key = force_key
        self.floor = float(floor)
        self.direction_key = direction_key
        self.scalars_key = scalars_key
        self.scale_key = scale_key
        self.eps = float(eps)

    def forward(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        forces = inputs[self.force_key]
        idx_m = inputs[properties.idx_m]
        n_atoms = inputs[properties.n_atoms]

        sum_sq = torch.zeros(
            n_atoms.shape[0], dtype=forces.dtype, device=forces.device
        ).index_add(0, idx_m, (forces**2).sum(-1))
        mean_sq = sum_sq / (3 * n_atoms.to(forces.dtype))
        scale = torch.sqrt(mean_sq + self.floor**2)

        scale_atom = scale[idx_m]
        direction = forces / scale_atom.unsqueeze(-1)
        norm = torch.sqrt((direction**2).sum(-1) + self.eps)
        scalars = torch.stack([torch.log(scale_atom), torch.log(norm)], dim=-1)

        inputs[self.direction_key] = direction
        inputs[self.scalars_key] = scalars
        inputs[self.scale_key] = scale
        return inputs
