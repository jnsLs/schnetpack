"""The eigenvalue-damping experiment configs compose and run a step."""

import pytest
import torch
import schnetpack as spk
from hydra import compose, initialize_config_module
from hydra.utils import instantiate
from omegaconf import open_dict


def _config(experiment, ref_model_path, *extra):
    with initialize_config_module(
        config_module="schnetpack.configs", version_base="1.2"
    ):
        config = compose(
            config_name="train",
            overrides=[
                f"experiment={experiment}",
                f"globals.ref_model_path={ref_model_path}",
                "model.representation.n_atom_basis=16",
                "model.representation.n_interactions=2",
                "run.work_dir=.",
                "run.path=.",
                *extra,
            ],
            return_hydra_config=True,
        )
    with open_dict(config):
        del config["hydra"]
    return config


@pytest.mark.integration
@pytest.mark.parametrize(
    "experiment, source",
    [
        ("newton_step_hvp_eigdamp", "lanczos"),
        ("newton_step_hvp_eigdamp_batch", "batch"),
        ("newton_step_hvp_eigdamp_batch_rel", "batch"),
        ("newton_step_hvp_eigdamp_batch_fin", "batch"),
        ("newton_step_hvp_eigdamp_batch_fin_rel", "batch"),
        ("newton_step_hvp_eigdamp_batch_rel_step", "batch"),
        ("newton_step_hvp_eigdamp_batch_fin_rel_step", "batch"),
    ],
)
def test_eigdamp_config_runs_a_step(experiment, source, ref_model_path, newton_batch):
    config = _config(experiment, ref_model_path, "globals.lanczos_steps=8")
    assert config.task.damping_source == source
    if source == "batch":
        props = ["e_min"]
        if experiment.endswith("_step"):
            props += ["newton_step_ref", "has_newton_step_ref"]
            newton_batch["newton_step_ref"] = 0.1 * torch.randn(
                newton_batch["_positions"].shape, dtype=torch.float64
            )
            newton_batch["has_newton_step_ref"] = torch.tensor([1.0, 0.0, 1.0])
        assert list(config.data.dataset.load_properties) == props
        assert config.data.dataset.property_units.e_min == "kcal/mol/Ang/Ang"
        newton_batch["e_min"] = torch.tensor([-50.0, -10.0, 3.0])

    model = instantiate(config.model)
    task = instantiate(config.task, model=model, _convert_="partial", _recursive_=True)
    # 9.717 eV/Ang^2 in kcal/mol/Ang^2
    assert task.damping_eps == pytest.approx(9.717 * 23.0605, rel=1e-4)

    if "_rel" in experiment:
        assert task.residual_floor == pytest.approx(0.02306)
        assert task.outputs[0].name == "damped_hvp_rel"
        (absolute,) = [o for o in task.outputs if o.name == "damped_hvp"]
        assert absolute.loss_weight == 0.0
    if "_fin" in experiment:
        assert task.force_input_key == "input_forces"
        assert model.representation.vector_input_key == "force_input_dir"
        assert model.representation.scalar_input_net.weight.shape == (16, 2)
        assert isinstance(model.input_modules[1], spk.atomistic.ForceInput)
    else:
        assert task.force_input_key is None
    if experiment.endswith("_step"):
        assert task.step_target_key == "newton_step_ref"
        assert task.step_target_mask_key == "has_newton_step_ref"
        assert task.step_floor == pytest.approx(0.001)
        names = [o.name for o in task.outputs]
        assert names[:2] == ["damped_hvp_rel", "newton_step_rel"]
        assert task.outputs[1].target_property == "newton_step_ref_rel"
        assert task.outputs[1].loss_weight == 1.0
    else:
        assert task.step_target_key is None
    loss = task.training_step(newton_batch, 0)
    if "_fin" in experiment:
        assert "input_forces" in newton_batch
        assert model.representation.vector_input_weight.grad is None
    assert torch.isfinite(loss)
    loss.backward()
    assert any(
        p.grad is not None and p.grad.abs().sum() > 0 for p in task.model.parameters()
    )


@pytest.mark.integration
def test_old_config_is_unchanged_by_default(ref_model_path):
    config = _config("newton_step_training_horm_hvp", ref_model_path)
    assert "damping_source" not in config.task
