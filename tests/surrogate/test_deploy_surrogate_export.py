"""Exporting a reference potential with ``spkdeploy --target surrogate``.

The surrogate task already accepts a TorchScript archive; what these tests pin
is that the archive :mod:`schnetpack.deploy` produces is the *same potential*,
down to the last bit, and that the ways an export can be useless are caught at
export time rather than at the first training step.
"""

import copy
import warnings
from typing import Dict

import pytest
import torch
import torch.nn as nn

import schnetpack as spk
from schnetpack import properties
from schnetpack.deploy import export_for_lammps, export_for_surrogate
from schnetpack.train.surrogate import load_reference_model


class DetachedForces(nn.Module):
    """A model exported for inference: it claims to need R, but its forces
    carry no graph back to it.

    The eager equivalent lives in ``test_reference_model_loading.py``; this one
    is annotated so that it survives ``torch.jit.script``, which is the whole
    point of exercising it here.
    """

    def __init__(self):
        super().__init__()
        self.model_outputs = [properties.forces]
        self.required_derivatives = [properties.R]
        self.force_key = properties.forces

    def forward(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        inputs[self.force_key] = torch.zeros_like(inputs[properties.R])
        return inputs


class CutoffFreeRepresentation(nn.Module):
    """A representation that does not advertise a ``cutoff``."""

    def __init__(self, n_atom_basis: int):
        super().__init__()
        self.embedding = nn.Embedding(100, n_atom_basis)

    def forward(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        inputs["scalar_representation"] = self.embedding(inputs[properties.Z])
        return inputs


def _detached_forces_model():
    return spk.model.NeuralNetworkPotential(
        representation=spk.representation.PaiNN(
            n_atom_basis=16,
            n_interactions=1,
            radial_basis=spk.nn.GaussianRBF(n_rbf=8, cutoff=5.0),
            cutoff_fn=spk.nn.CosineCutoff(5.0),
        ),
        input_modules=[spk.atomistic.PairwiseDistances()],
        output_modules=[DetachedForces()],
    )


@pytest.fixture
def deployed_ref_model_path(tmp_path, ref_model_path):
    """The pickled reference model, run through the surrogate export."""
    model = torch.load(ref_model_path, map_location="cpu", weights_only=False)
    path = tmp_path / "ref_deployed.pt"
    export_for_surrogate(model, str(path))
    return str(path)


def test_deployed_model_loads_as_a_frozen_reference(deployed_ref_model_path):
    model = load_reference_model(deployed_ref_model_path, "torchscript")
    assert isinstance(model, torch.jit.RecursiveScriptModule)
    assert not any(p.requires_grad for p in model.parameters())
    assert model.do_postprocessing is False


def test_deployed_reference_reproduces_the_eager_newton_system(
    student_model,
    surrogate_outputs,
    ref_model_path,
    deployed_ref_model_path,
    newton_batch,
):
    """The whole point of the export: same numbers, other container.

    Bit-identical, not merely close -- compiling the reference must not move
    the training targets at all, or a run against the archive is no longer
    comparable to a run against the checkpoint it came from.
    """

    def run(path, fmt):
        torch.manual_seed(0)
        task = spk.train.NewtonSurrogateTask(
            model=copy.deepcopy(student_model),
            outputs=surrogate_outputs,
            ref_model_path=path,
            ref_model_format=fmt,
            optimizer_args={"lr": 1e-3},
        )
        task.train()
        torch.set_grad_enabled(True)
        batch = {k: v.clone() for k, v in newton_batch.items()}
        pred = task.predict_without_postprocessing(batch)
        pred = task.augment_predictions(batch, pred)
        return pred[properties.ref_forces], pred[properties.damped_hvp]

    eager_f, eager_hvp = run(ref_model_path, "torch")
    deployed_f, deployed_hvp = run(deployed_ref_model_path, "torchscript")

    torch.testing.assert_close(deployed_f, eager_f, rtol=0, atol=0)
    torch.testing.assert_close(deployed_hvp, eager_hvp, rtol=0, atol=0)


def test_export_does_not_modify_the_model_it_is_handed(tmp_path, ref_model_path):
    """A library function that eats its argument is a trap for its callers."""
    model = torch.load(ref_model_path, map_location="cpu", weights_only=False)
    model.postprocessors = nn.ModuleList([spk.transform.CastTo64()])

    export_for_surrogate(model, str(tmp_path / "out.pt"))

    assert len(model.postprocessors) == 1
    assert model.do_postprocessing is True


def test_export_strips_postprocessing(tmp_path, ref_model_path):
    """Postprocessing is inference-only; CastTo64 would break the step arithmetic."""
    model = torch.load(ref_model_path, map_location="cpu", weights_only=False)
    model.postprocessors = nn.ModuleList([spk.transform.CastTo64()])

    path = tmp_path / "out.pt"
    export_for_surrogate(model, str(path))

    deployed = torch.jit.load(str(path), map_location="cpu")
    assert len(list(deployed.postprocessors.children())) == 0
    assert deployed.do_postprocessing is False


def test_model_without_forces_is_rejected(tmp_path, ref_model_path):
    model = torch.load(ref_model_path, map_location="cpu", weights_only=False)
    model.model_outputs = ["energy"]

    with pytest.raises(ValueError, match="forces"):
        export_for_surrogate(model, str(tmp_path / "out.pt"))


def test_non_differentiable_forces_are_caught_at_export_time(tmp_path):
    """The failure `augment_predictions` raises, moved forward to the export."""
    with pytest.raises(RuntimeError, match="not differentiable"):
        export_for_surrogate(_detached_forces_model(), str(tmp_path / "out.pt"))


def test_the_check_can_be_skipped(tmp_path):
    """`--no-check` still exports, for a model the smoke test cannot handle."""
    path = tmp_path / "out.pt"
    export_for_surrogate(_detached_forces_model(), str(path), check=False)
    assert path.exists()


def test_lammps_export_still_writes_the_cutoff(tmp_path, ref_model_path):
    model = torch.load(ref_model_path, map_location="cpu", weights_only=False)
    path = tmp_path / "lammps.pt"

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        export_for_lammps(model, str(path))
    assert not [w for w in caught if issubclass(w.category, UserWarning)]

    extra_files = {"cutoff": ""}
    torch.jit.load(str(path), map_location="cpu", _extra_files=extra_files)
    assert float(extra_files["cutoff"]) == pytest.approx(5.0)


def test_lammps_export_without_a_cutoff_warns_instead_of_failing(tmp_path):
    """A representation with no `cutoff` is unusable from LAMMPS, but exportable."""
    model = spk.model.NeuralNetworkPotential(
        representation=CutoffFreeRepresentation(n_atom_basis=16),
        input_modules=[spk.atomistic.PairwiseDistances()],
        output_modules=[
            spk.atomistic.Atomwise(n_in=16, output_key="energy"),
            spk.atomistic.Forces(),
        ],
    )
    path = tmp_path / "no_cutoff.pt"

    with pytest.warns(UserWarning, match="cutoff"):
        export_for_lammps(model, str(path))

    assert path.exists()
