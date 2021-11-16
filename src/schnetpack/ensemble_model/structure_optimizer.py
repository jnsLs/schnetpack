from pymatgen.io.ase import AseAtomsAdaptor
from ase.optimize import LBFGS, FIRE, QuasiNewton
from ase.constraints import ExpCellFilter

from pymatgen.core import Structure
from ase import Atoms
import signal

import os


__all__ = ["ASEStructureOptimizer"]


def timeout_handler(signum, frame):
    print(signum, frame)
    raise StructureOptimizerException("Timeout!")


class StructureOptimizerException(Exception):
    pass


class ASEStructureOptimizer:
    def __init__(
        self,
        calculator,
        max_steps: int = 300,
        force_th: float = 0.05,
        optimizer_class: type = LBFGS,
        allow_unconverged=False,
        timeout_limit=86400,
    ):
        self.calculator = calculator
        self.relax_id = 0
        self.max_steps = max_steps
        self.force_th = force_th
        self.optimizer_class = optimizer_class
        self.allow_unconverged = allow_unconverged
        self.timeout_limit = timeout_limit

    def relax(
        self,
        atoms: Atoms,
        relax_id: int = None,
    ):

        if relax_id is None:
            relax_id = self.relax_id
        self.relax_id += 1

        print("Relaxator: starting relaxation process {}.".format(relax_id))

        if type(atoms) == Structure:
            atoms = AseAtomsAdaptor.get_atoms(atoms)

        ase_optimiser = self.optimizer_class(atoms, force_consistent=False)
        atoms.set_calculator(self.calculator)

        # start run with timeout
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(self.timeout_limit)
        converged = ase_optimiser.run(
            steps=ase_optimiser.nsteps + self.max_steps, fmax=self.force_th
        )
        signal.alarm(0)

        # exception if not converged
        #if not converged and not self.allow_unconverged:
        #    raise StructureOptimizerException(f"No convergence after max_steps: {self.max_steps}")

        return self.calculator.traj_logger.results_dict
