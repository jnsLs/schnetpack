"""The experiment config, the task and the modules must agree on key names.

Nothing else in the test suite composes a hydra config, so a rename that
touches the source but not the yaml (or vice versa) would otherwise only be
caught by launching a real training run.
"""

import pytest
import torch
from hydra import compose, initialize_config_module
from hydra.utils import instantiate
from omegaconf import OmegaConf, open_dict

EXPERIMENT = "newton_step_training_horm_hvp"


@pytest.fixture
def experiment_config(ref_model_path):
    with initialize_config_module(
        config_module="schnetpack.configs", version_base="1.2"
    ):
        config = compose(
            config_name="train",
            overrides=[
                f"experiment={EXPERIMENT}",
                f"task.ref_model_path={ref_model_path}",
                "model.representation.n_atom_basis=16",
                "model.representation.n_interactions=2",
                "run.work_dir=.",
                "run.path=.",
            ],
            return_hydra_config=True,
        )
    with open_dict(config):
        del config["hydra"]
    return config


@pytest.mark.integration
def test_experiment_config_composes(experiment_config):
    assert experiment_config.task._target_ == "schnetpack.task.AtomisticTaskSurrogate"


@pytest.mark.integration
def test_config_keys_match_the_property_constants(experiment_config):
    """The yaml must not drift from schnetpack.properties."""
    import schnetpack.properties as properties

    globals_ = experiment_config.globals
    assert globals_.newton_step_key == properties.newton_step
    assert globals_.damping_factor_key == properties.damping_factor
    assert globals_.damped_hvp_key == properties.damped_hvp
    assert globals_.ref_forces_key == properties.ref_forces


@pytest.mark.integration
def test_ref_model_path_is_mandatory():
    """The machine-specific path must be supplied, not silently defaulted."""
    from omegaconf.errors import MissingMandatoryValue

    with initialize_config_module(
        config_module="schnetpack.configs", version_base="1.2"
    ):
        config = compose(config_name="train", overrides=[f"experiment={EXPERIMENT}"])
    with pytest.raises(MissingMandatoryValue):
        _ = config.task.ref_model_path


@pytest.mark.integration
def test_configured_task_runs_a_step(experiment_config, newton_batch, ref_model_path):
    """Instantiate model and task straight from the yaml and take one step."""
    model = instantiate(experiment_config.model)
    task = instantiate(
        experiment_config.task, model=model, _convert_="partial", _recursive_=True
    )

    loss = task.training_step(newton_batch, 0)

    assert torch.isfinite(loss), loss
    loss.backward()
    grads = [p.grad for p in task.model.parameters() if p.grad is not None]
    assert grads, "no gradients reached the student model"
    assert any(g.abs().sum() > 0 for g in grads)


@pytest.mark.integration
def test_reference_model_stays_frozen(experiment_config, newton_batch):
    """The reference potential must never be updated by training."""
    model = instantiate(experiment_config.model)
    task = instantiate(
        experiment_config.task, model=model, _convert_="partial", _recursive_=True
    )
    assert all(not p.requires_grad for p in task.ref_model.parameters())

    before = [p.detach().clone() for p in task.ref_model.parameters()]
    task.training_step(newton_batch, 0).backward()
    assert all(p.grad is None for p in task.ref_model.parameters())
    for old, new in zip(before, task.ref_model.parameters()):
        assert torch.equal(old, new)
