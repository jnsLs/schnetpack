"""Self-contained fixtures for the Newton-step surrogate pipeline.

Everything here is CPU + float32 and fully seeded, so the golden regression
values in ``tests/testdata/newton_step_golden.pt`` are reproducible on any
machine. Nothing depends on the real reference checkpoint or on a dataset.
"""

import numpy as np
import pytest
import torchmetrics.regression
import torch
from ase import Atoms

import schnetpack as spk
import schnetpack.train.loss
import schnetpack.train.metrics
from schnetpack.data.loader import _atoms_collate_fn

CUTOFF = 5.0
N_ATOM_BASIS = 16
N_INTERACTIONS = 2
N_RBF = 8

#: Molecules used to build the regression batch. Deliberately small and of
#: differing size, so that per-molecule aggregation and the ``_idx_m``
#: bookkeeping are actually exercised.
MOLECULES = [
    ("H2O", [8, 1, 1], [[0.0, 0.0, 0.0], [0.0, 0.757, 0.587], [0.0, -0.757, 0.587]]),
    ("CH2", [6, 1, 1], [[0.1, 0.0, 0.0], [0.0, 1.09, 0.0], [1.02, -0.38, 0.0]]),
    (
        "NH3",
        [7, 1, 1, 1],
        [
            [0.0, 0.0, 0.12],
            [0.0, 0.94, -0.28],
            [0.81, -0.47, -0.28],
            [-0.81, -0.47, -0.28],
        ],
    ),
]


def _representation():
    return spk.representation.PaiNN(
        n_atom_basis=N_ATOM_BASIS,
        n_interactions=N_INTERACTIONS,
        radial_basis=spk.nn.GaussianRBF(n_rbf=N_RBF, cutoff=CUTOFF),
        cutoff_fn=spk.nn.CosineCutoff(CUTOFF),
    )


@pytest.fixture
def newton_batch():
    """A collated batch of three small molecules, float32, no PBC."""
    transforms = [
        spk.transform.MatScipyNeighborList(cutoff=CUTOFF),
        spk.transform.CastTo32(),
    ]

    inputs = []
    for i, (_, numbers, positions) in enumerate(MOLECULES):
        atoms = Atoms(numbers=numbers, positions=np.array(positions))
        props = {
            spk.properties.idx: torch.tensor([i]),
            spk.properties.n_atoms: torch.tensor([len(atoms)]),
            spk.properties.Z: torch.from_numpy(atoms.get_atomic_numbers()),
            spk.properties.R: torch.from_numpy(atoms.get_positions()),
            spk.properties.cell: torch.from_numpy(atoms.cell[:][None].copy()),
            spk.properties.pbc: torch.from_numpy(atoms.pbc)[None],
        }
        for transform in transforms:
            props = transform(props)
        inputs.append(props)

    return _atoms_collate_fn(inputs)


@pytest.fixture
def student_model():
    """The model under training: predicts the Newton step and the damping factor."""
    torch.manual_seed(0)
    return spk.model.NeuralNetworkPotential(
        representation=_representation(),
        input_modules=[spk.atomistic.PairwiseDistances()],
        output_modules=[
            spk.atomistic.NewtonStep(
                newton_step_key=spk.properties.newton_step,
                n_in=N_ATOM_BASIS,
                n_hidden=12,
            ),
            spk.atomistic.DampingFactor(
                output_key=spk.properties.damping_factor,
                n_in=N_ATOM_BASIS,
                positivity="abs",
            ),
        ],
        postprocessors=[spk.transform.CastTo64()],
    )


@pytest.fixture
def ref_model_path(tmp_path):
    """A pickled reference model with the same layout as a real checkpoint.

    ``output_modules[1]`` is a :class:`~schnetpack.atomistic.Forces` module,
    which is what ``AtomisticTaskSurrogate`` replaces with an ``HVP``.
    """
    torch.manual_seed(1)
    ref_model = spk.model.NeuralNetworkPotential(
        representation=_representation(),
        input_modules=[spk.atomistic.PairwiseDistances()],
        output_modules=[
            spk.atomistic.Atomwise(n_in=N_ATOM_BASIS, output_key="energy"),
            spk.atomistic.Forces(),
        ],
    )
    path = tmp_path / "ref_model"
    torch.save(ref_model, path)
    return str(path)


@pytest.fixture
def surrogate_outputs():
    """The ``ModelOutput`` list mirroring ``newton_step_training_horm_hvp.yaml``."""
    return [
        spk.task.ModelOutput(
            name=spk.properties.damped_hvp,
            target_property=spk.properties.ref_forces,
            loss_fn=torch.nn.MSELoss(),
            metrics={
                "mae": torchmetrics.regression.MeanAbsoluteError(),
                "mse": torchmetrics.regression.MeanSquaredError(),
            },
            loss_weight=1.0,
        ),
        spk.task.ModelOutput(
            name=spk.properties.damping_factor,
            target_property="target_damping_factor",
            loss_fn=torch.nn.MSELoss(),
            metrics={
                "mae": torchmetrics.regression.MeanAbsoluteError(),
                "mse": torchmetrics.regression.MeanSquaredError(),
            },
            loss_weight=0.00001,
        ),
        spk.task.ModelOutput(
            name=spk.properties.newton_step,
            target_property=spk.properties.ref_forces,
            loss_fn=spk.train.loss.DescendingLoss(),
            metrics={
                "ascent_loss": spk.train.metrics.IsDescendingMetric(clamp_at_zero=True),
                "mean_ascent": spk.train.metrics.IsDescendingMetric(
                    clamp_at_zero=False
                ),
            },
            loss_weight=0.0,
        ),
    ]


@pytest.fixture
def surrogate_task(student_model, ref_model_path, surrogate_outputs):
    return spk.task.AtomisticTaskSurrogate(
        model=student_model,
        outputs=surrogate_outputs,
        optimizer_cls=torch.optim.AdamW,
        optimizer_args={"lr": 1e-3, "weight_decay": 0.01},
        ref_model_path=ref_model_path,
    )
