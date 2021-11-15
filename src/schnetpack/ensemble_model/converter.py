import torch
import torch.nn as nn
import schnetpack.properties as structure
from schnetpack.data.loader import _atoms_collate_fn
from ase import Atoms
from typing import Optional, List


__all__ = ["AtomsConverter"]


class AtomsConverter:
    def __init__(self, transforms: Optional[List[torch.nn.Module]] = [nn.Identity()]):
        # todo: remove this hack
        for transf in transforms:
            if hasattr(transf, "mode"):
                transf.mode = "pre"
        self.transform_module = torch.nn.Sequential(*transforms)

    def __call__(self, atoms: Atoms):
        properties = self._get_properties(atoms)

        return _atoms_collate_fn([properties])

    def _get_properties(self, atoms: Atoms):
        """
        Similar to loading structure from dataset.
        """
        atms = atoms.copy()

        properties = {}
        properties[structure.idx] = torch.tensor([0])

        Z = atms.numbers.copy()
        properties[structure.n_atoms] = torch.tensor([Z.shape[0]])
        properties[structure.Z] = torch.tensor(Z, dtype=torch.long)
        properties[structure.position] = torch.tensor(atms.get_positions(wrap=True))
        properties[structure.cell] = torch.tensor(atms.cell.copy())
        properties[structure.pbc] = torch.tensor(atms.pbc)

        # apply transforms
        properties = self.transform_module(properties)

        return properties