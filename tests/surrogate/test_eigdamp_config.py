"""The eigenvalue-damping experiment configs compose and run a step."""

import pytest
import torch
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
    ],
)
def test_eigdamp_config_runs_a_step(experiment, source, ref_model_path, newton_batch):
    config = _config(experiment, ref_model_path, "globals.lanczos_steps=8")
    assert config.task.damping_source == source
    if source == "batch":
        assert list(config.data.dataset.load_properties) == ["e_min"]
        assert config.data.dataset.property_units.e_min == "kcal/mol/Ang/Ang"
        newton_batch["e_min"] = torch.tensor([-50.0, -10.0, 3.0])

    model = instantiate(config.model)
    task = instantiate(config.task, model=model, _convert_="partial", _recursive_=True)
    # 9.717 eV/Ang^2 in kcal/mol/Ang^2
    assert task.damping_eps == pytest.approx(9.717 * 23.0605, rel=1e-4)

    loss = task.training_step(newton_batch, 0)
    assert torch.isfinite(loss)
    loss.backward()
    assert any(
        p.grad is not None and p.grad.abs().sum() > 0 for p in task.model.parameters()
    )


@pytest.mark.integration
def test_old_config_is_unchanged_by_default(ref_model_path):
    config = _config("newton_step_training_horm_hvp", ref_model_path)
    assert "damping_source" not in config.task
