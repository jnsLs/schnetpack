import os

import ase
from ase import units
from ase.calculators.calculator import Calculator, all_changes
from ase.io import read, write
from ase.io.trajectory import Trajectory
from ase.md import VelocityVerlet, Langevin, MDLogger
from ase.md.velocitydistribution import (
    MaxwellBoltzmannDistribution,
    Stationary,
    ZeroRotation,
)
from ase.optimize import QuasiNewton
from ase.vibrations import Vibrations

import torch
import schnetpack
import logging

import schnetpack.task
from schnetpack import properties
from schnetpack.data.loader import _atoms_collate_fn
from schnetpack.transform import CastTo32, CastTo64
from schnetpack.units import convert_units

from typing import Optional, List, Union
from ase import Atoms

log = logging.getLogger(__name__)

__all__ = ["AseInterface"]


class AseInterface:
    """
    Interface for ASE calculations (optimization and molecular dynamics)
    """

    def __init__(
        self,
        molecule_path: str,
        working_dir: str,
        model: schnetpack.model.AtomisticModel,
        converter: None,
        calculator: None,
        energy: str = "energy",
        forces: str = "forces",
        stress: str = "stress",
        energy_units: Union[str, float] = "kcal/mol",
        forces_units: Union[str, float] = "kcal/mol/Angstrom",
        stress_units: Union[str, float] = "kcal/mol/Angstrom/Angstrom/Angstrom",
    ):
        """
        Args:
            molecule_path: Path to initial geometry
            working_dir: Path to directory where files should be stored
            model: Trained model
            neighbor_list: neighbor list for computing interatomic distances.
            device: select to run calculations on 'cuda' or 'cpu'
            energy: name of energy property in provided model.
            forces: name of forces in provided model.
            stress: name of stress property in provided model.
            energy_units: energy units used by model
            forces_units: force units used by model
            stress_units: stress units used by model
            precision: toggle model precision
        """
        # Setup directory
        self.working_dir = working_dir
        if not os.path.exists(self.working_dir):
            os.makedirs(self.working_dir)

        # Load the molecule
        self.molecule = read(molecule_path)

        # Set up calculator

        #calculator = SpkCalculator(
        #    model,
        #    converter=converter,
        #    energy=energy,
        #    forces=forces,
        #    stress=stress,
        #    energy_units=energy_units,
        #    forces_units=forces_units,
        #    stress_units=stress_units,
        #)

        self.molecule.set_calculator(calculator)

        self.dynamics = None

    def save_molecule(self, name: str, file_format: str = "xyz", append: bool = False):
        """
        Save the current molecular geometry.

        Args:
            name: Name of save-file.
            file_format: Format to store geometry (default xyz).
            append: If set to true, geometry is added to end of file (default False).
        """
        molecule_path = os.path.join(
            self.working_dir, "{:s}.{:s}".format(name, file_format)
        )
        write(molecule_path, self.molecule, format=file_format, append=append)

    def calculate_single_point(self):
        """
        Perform a single point computation of the energies and forces and
        store them to the working directory. The format used is the extended
        xyz format. This functionality is mainly intended to be used for
        interfaces.
        """
        energy = self.molecule.get_potential_energy()
        forces = self.molecule.get_forces()
        self.molecule.energy = energy
        self.molecule.forces = forces

        self.save_molecule("single_point", file_format="xyz")

    def init_md(
        self,
        name: str,
        time_step: float = 0.5,
        temp_init: float = 300,
        temp_bath: Optional[float] = None,
        reset: bool = False,
        interval: int = 1,
    ):
        """
        Initialize an ase molecular dynamics trajectory. The logfile needs to
        be specifies, so that old trajectories are not overwritten. This
        functionality can be used to subsequently carry out equilibration and
        production.

        Args:
            name: Basic name of logfile and trajectory
            time_step: Time step in fs (default=0.5)
            temp_init: Initial temperature of the system in K (default is 300)
            temp_bath: Carry out Langevin NVT dynamics at the specified
                temperature. If set to None, NVE dynamics are performed
                instead (default=None)
            reset: Whether dynamics should be restarted with new initial
                conditions (default=False)
            interval: Data is stored every interval steps (default=1)
        """

        # If a previous dynamics run has been performed, don't reinitialize
        # velocities unless explicitly requested via restart=True
        if self.dynamics is None or reset:
            self._init_velocities(temp_init=temp_init)

        # Set up dynamics
        if temp_bath is None:
            self.dynamics = VelocityVerlet(self.molecule, time_step * units.fs)
        else:
            self.dynamics = Langevin(
                self.molecule,
                time_step * units.fs,
                temp_bath * units.kB,
                1.0 / (100.0 * units.fs),
            )

        # Create monitors for logfile and a trajectory file
        logfile = os.path.join(self.working_dir, "{:s}.log".format(name))
        trajfile = os.path.join(self.working_dir, "{:s}.traj".format(name))
        logger = MDLogger(
            self.dynamics,
            self.molecule,
            logfile,
            stress=False,
            peratom=False,
            header=True,
            mode="a",
        )
        trajectory = Trajectory(trajfile, "w", self.molecule)

        # Attach monitors to trajectory
        self.dynamics.attach(logger, interval=interval)
        self.dynamics.attach(trajectory.write, interval=interval)

    def _init_velocities(
        self,
        temp_init: float = 300,
        remove_translation: bool = True,
        remove_rotation: bool = True,
    ):
        """
        Initialize velocities for molecular dynamics

        Args:
            temp_init: Initial temperature in Kelvin (default 300)
            remove_translation: Remove translation components of velocity (default True)
            remove_rotation: Remove rotation components of velocity (default True)
        """
        MaxwellBoltzmannDistribution(self.molecule, temp_init * units.kB)
        if remove_translation:
            Stationary(self.molecule)
        if remove_rotation:
            ZeroRotation(self.molecule)

    def run_md(self, steps: int):
        """
        Perform a molecular dynamics simulation using the settings specified
        upon initializing the class.

        Args:
            steps: Number of simulation steps performed
        """
        if not self.dynamics:
            raise AttributeError(
                "Dynamics need to be initialized using the" " 'setup_md' function"
            )

        self.dynamics.run(steps)

    def optimize(self, fmax: float = 1.0e-2, steps: int = 1000):
        """
        Optimize a molecular geometry using the Quasi Newton optimizer in ase
        (BFGS + line search)

        Args:
            fmax: Maximum residual force change (default 1.e-2)
            steps: Maximum number of steps (default 1000)
        """
        name = "optimization"
        optimize_file = os.path.join(self.working_dir, name)
        optimizer = QuasiNewton(
            self.molecule,
            trajectory="{:s}.traj".format(optimize_file),
            restart="{:s}.pkl".format(optimize_file),
        )
        optimizer.run(fmax, steps)

        # Save final geometry in xyz format
        self.save_molecule(name)

    def compute_normal_modes(self, write_jmol: bool = True):
        """
        Use ase calculator to compute numerical frequencies for the molecule

        Args:
            write_jmol: Write frequencies to input file for visualization in jmol (default=True)
        """
        freq_file = os.path.join(self.working_dir, "normal_modes")

        # Compute frequencies
        frequencies = Vibrations(self.molecule, name=freq_file)
        frequencies.run()

        # Print a summary
        frequencies.summary()

        # Write jmol file if requested
        if write_jmol:
            frequencies.write_jmol()