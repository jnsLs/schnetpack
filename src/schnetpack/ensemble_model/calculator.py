import numpy as np
from schnetpack import properties

__all__ = ["Calculator", "DummyCalculator", "NNCalculator"]


class TrajLogger:
    """
    Logging module for ase calculators.

    """
    def __init__(self):
        self.e = []
        self.f = []
        self.s = []
        self.calculator_type = []
        self.structures = []

    def reset(self):
        self.e = []
        self.f = []
        self.s = []
        self.calculator_type = []
        self.structures = []

    def __call__(self, atoms, energy, forces, stress, calculator_type):
        print("called logger")
        self.e.append(energy)
        self.f.append(forces)
        self.s.append(stress)
        self.calculator_type.append(calculator_type)
        atms_copy = atoms.copy()
        atms_copy.calc = None
        self.structures.append(atms_copy)

    @property
    def results_dict(self):
        results = {
            properties.energy: self.e.copy(),
            properties.forces: self.f.copy(),
            properties.position: self.structures.copy(),
            "calculator_type": self.calculator_type.copy(),
        }
        self.reset()
        return results


class Calculator:
    """
    Base class for ase calculators.

    """
    def __init__(self):
        self.traj_logger = TrajLogger()
        self.results = None
        self.atoms = None

    def calculation_required(
        self,
        atoms,
    ):
        if self.atoms is None or not self.atoms == atoms:
            return True
        return False

    def get_forces(
        self,
        atoms,
    ):
        if self.calculation_required(atoms):
            self.calculate(atoms)
        return self.results["forces"]

    def get_potential_energy(
        self,
        atoms,
    ):
        if self.calculation_required(atoms):
            self.calculate(atoms)
        return self.results["energy"]

    def get_stress(
        self,
        atoms,
    ):
        if self.calculation_required(atoms):
            self.calculate(atoms)
        return self.results["stress"]

    def calculate(self, atoms):
        pass


class DummyCalculator(Calculator):
    """
    Only for testing purposes. Replace with QE, LAMMPS, VASP, ... calculator from ase!

    """
    def __init__(self):
        super(DummyCalculator, self).__init__()

    def calculate(self, atoms):
        self.results = dict(
            energy=np.random.random(1).item(),
            forces=np.random.random(atoms.positions.shape),
            stress=np.random.random((3, 3)),
        )


class NNCalculator(Calculator):
    """
    Calculator for neural network models with uncertainty prediction.
    NN-models must return a prediction and an uncertainty dict.
    """
    def __init__(
        self,
        model,
        atoms_converter,
        device="cpu",
    ):
        super(NNCalculator, self).__init__()

        self.model = model
        self.device = device
        self.atoms_converter = atoms_converter

    def calculate(
        self,
        atoms,
    ):
        self.model.eval()
        # build spk inputs
        inputs = self.atoms_converter(atoms)

        # compute properties
        self.model.to(self.device)
        self.model.eval()
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        means, stds = self.model(inputs)

        # update stored results
        self.results = {
            properties.energy: means[properties.energy].detach().cpu().squeeze().numpy(),
            properties.forces: means[properties.forces].detach().cpu().squeeze().numpy(),
            "forces_uncertainty": stds[properties.forces].detach().cpu().squeeze().numpy(),
        }

        # update _atoms
        self.atoms = atoms.copy()

    def get_uncertainties(self, atoms):
        if self.calculation_required(atoms):
            self.calculate(atoms)

        return self.results["forces_uncertainty"], self.results["forces_uncertainty"]

    def get_rel_uncertainty(self, atoms):
        f_u, s_u = self.get_uncertainties(atoms)
        f = self.get_forces(atoms)
        s = self.get_stress(atoms)

        s = s * 10
        s_u = s_u * 10

        f_rel = np.mean(np.sqrt((f_u ** 2).sum(1)) / (np.sqrt((f ** 2).sum(1)) + 1))
        s_rel = np.mean(np.sqrt((s_u ** 2).sum()) / (np.sqrt((s ** 2).sum()) + 1))

        uncertainty_rel = (f_rel + s_rel) / 2
        print(f"uncertainties: forces: {f_rel} --- stress: {s_rel} --- total: {uncertainty_rel}")
        return uncertainty_rel