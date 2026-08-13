"""Loading a reference potential without modifying it.

The point of the loader is that the reference model is used exactly as saved:
nothing inspects or replaces its output modules, so a potential whose internals
are unknown can supervise training as long as it follows SchNetPack's
``Dict[str, Tensor]`` interface. That makes the *format* it was saved in the
only thing the loader has to know about.
"""

import warnings

import pytest
import torch
import torch.nn as nn

import schnetpack as spk
from schnetpack import properties
from schnetpack.train.surrogate import load_reference_model


def test_pickled_model_loads_and_is_frozen(ref_model_path):
    model = load_reference_model(ref_model_path, "torch")
    assert not any(p.requires_grad for p in model.parameters())
    assert model.do_postprocessing is False


def test_scripted_model_loads(scripted_ref_model_path):
    model = load_reference_model(scripted_ref_model_path, "torchscript")
    assert isinstance(model, torch.jit.RecursiveScriptModule)
    assert not any(p.requires_grad for p in model.parameters())


@pytest.mark.parametrize("fixture", ["ref_model_path", "scripted_ref_model_path"])
def test_auto_detects_both_formats(fixture, request):
    path = request.getfixturevalue(fixture)
    assert load_reference_model(path, "auto") is not None


def test_unknown_format_is_rejected(ref_model_path):
    with pytest.raises(ValueError, match="ref_model_format"):
        load_reference_model(ref_model_path, "onnx")


def test_scripted_reference_reproduces_the_eager_newton_system(
    student_model, surrogate_outputs, ref_model_path, scripted_ref_model_path,
    newton_batch,
):
    """The whole point of supporting TorchScript: same numbers, other container.

    This also pins that the double backward survives scripting -- the second
    derivative is taken through the scripted forces.
    """
    import copy

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
    scripted_f, scripted_hvp = run(scripted_ref_model_path, "torchscript")

    torch.testing.assert_close(scripted_f, eager_f, rtol=0, atol=0)
    torch.testing.assert_close(scripted_hvp, eager_hvp, rtol=0, atol=0)


class DetachedForces(nn.Module):
    """Stands in for a model exported for inference: forces with no graph."""

    def __init__(self):
        super().__init__()
        self.model_outputs = [properties.forces]

    def forward(self, inputs):
        inputs[properties.forces] = torch.zeros_like(inputs[properties.R])
        return inputs


class WithDropout(nn.Module):
    """Stands in for a model whose forward differs between train and eval."""

    def __init__(self):
        super().__init__()
        self.dropout = nn.Dropout(0.5)
        self.model_outputs = []

    def forward(self, inputs):
        return inputs


def test_detached_forces_raise_a_clear_error(
    student_model, surrogate_outputs, tmp_path, newton_batch
):
    """A model exported for inference gives forces with no graph to R."""
    frozen = spk.model.NeuralNetworkPotential(
        representation=spk.representation.PaiNN(
            n_atom_basis=16,
            n_interactions=1,
            radial_basis=spk.nn.GaussianRBF(n_rbf=8, cutoff=5.0),
            cutoff_fn=spk.nn.CosineCutoff(5.0),
        ),
        input_modules=[spk.atomistic.PairwiseDistances()],
        output_modules=[DetachedForces()],
    )
    path = tmp_path / "frozen_model"
    torch.save(frozen, path)

    task = spk.train.NewtonSurrogateTask(
        model=student_model,
        outputs=surrogate_outputs,
        ref_model_path=str(path),
        optimizer_args={"lr": 1e-3},
    )
    task.train()
    torch.set_grad_enabled(True)
    pred = task.predict_without_postprocessing(newton_batch)

    with pytest.raises(RuntimeError, match="not\\s+differentiable"):
        task.augment_predictions(newton_batch, pred)


def test_train_mode_sensitive_submodules_are_reported(tmp_path):
    """The reference runs in train mode; dropout would silently change targets."""
    model = spk.model.NeuralNetworkPotential(
        representation=nn.Identity(),
        input_modules=[],
        output_modules=[WithDropout()],
    )
    path = tmp_path / "dropout_model"
    torch.save(model, path)

    with pytest.warns(UserWarning, match="train/eval-sensitive"):
        load_reference_model(str(path), "torch")


def test_plain_model_warns_about_nothing(ref_model_path):
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        load_reference_model(ref_model_path, "torch")
