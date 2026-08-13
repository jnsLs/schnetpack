"""Tests for the three ways of obtaining a damping factor.

``DampingFactor`` selects between a fixed, a learnable and a predicted damping
factor. The selection used to be written as a truthiness test on a tensor, so
the values that make ``log`` or the tensor itself falsy silently selected the
wrong branch.
"""

import pytest
import torch

import schnetpack as spk
from schnetpack import properties


def _inputs(n_atoms=(3, 2)):
    idx_m = torch.cat([torch.full((n,), i) for i, n in enumerate(n_atoms)])
    torch.manual_seed(3)
    return {
        properties.n_atoms: torch.tensor(n_atoms),
        properties.idx_m: idx_m,
        "scalar_representation": torch.randn(sum(n_atoms), 8),
    }


def _module(**kwargs):
    torch.manual_seed(0)
    return spk.atomistic.DampingFactor(
        n_in=8, output_key=properties.damping_factor, **kwargs
    )


@pytest.mark.parametrize("value", [0.5, 1.0, 0.0])
def test_fixed_damping_factor_is_used_verbatim(value):
    """Including 0.0 and 1.0, which a truthiness test gets wrong."""
    out = _module(fixed_damping_factor=value)(_inputs())
    torch.testing.assert_close(
        out[properties.damping_factor], torch.tensor([value, value])
    )


@pytest.mark.parametrize("value", [0.5, 1.0, 2.0])
def test_learnable_damping_factor_starts_at_its_init(value):
    """init=1.0 gives log(1)=0, which a truthiness test gets wrong."""
    module = _module(init_learnable_damping_factor=value)
    out = module(_inputs())
    torch.testing.assert_close(
        out[properties.damping_factor], torch.tensor([value, value])
    )
    assert module.learnable_damping_factor.requires_grad


def test_learnable_damping_factor_stays_positive_after_a_large_update():
    """It is parameterised in log space precisely so it cannot go negative."""
    module = _module(init_learnable_damping_factor=0.5)
    with torch.no_grad():
        module.learnable_damping_factor -= 20.0
    out = module(_inputs())
    assert (out[properties.damping_factor] > 0).all()


def test_predicted_damping_factor_is_per_molecule():
    out = _module(aggregation_mode="sum", positivity="abs")(_inputs(n_atoms=(3, 2)))
    assert out[properties.damping_factor].shape == (2,)


def test_predicted_damping_factor_is_non_negative():
    """(H + lambda I) is only guaranteed positive definite for lambda >= 0."""
    module = _module(aggregation_mode="sum", positivity="abs")
    with torch.no_grad():  # force the raw sum negative
        module.outnet[-1].bias.fill_(-50.0)
    out = module(_inputs())
    assert (out[properties.damping_factor] >= 0).all()


def test_sum_aggregation_may_be_negative():
    """The positivity is what 'positive' adds; plain sum does not have it."""
    module = _module(aggregation_mode="sum")
    with torch.no_grad():
        module.outnet[-1].bias.fill_(-50.0)
    out = module(_inputs())
    assert (out[properties.damping_factor] < 0).any()


def test_positive_matches_abs_of_sum():
    """'positive' is exactly sum aggregation followed by abs."""
    summed = _module(aggregation_mode="sum")(_inputs())[properties.damping_factor]
    positive = _module(aggregation_mode="sum", positivity="abs")(_inputs())[
        properties.damping_factor
    ]
    torch.testing.assert_close(positive, summed.abs())


def test_penalty_matches_the_zero_target_it_replaces():
    """The damping penalty used to be an MSE against a fabricated zero target.

    ``MeanSquaredMagnitude`` says the same thing without inventing a label, and
    must keep saying it to the last bit -- the golden regression values were
    recorded with the old formulation.
    """
    torch.manual_seed(5)
    damping = torch.rand(9).abs()

    penalty = spk.train.MeanSquaredMagnitude()(damping)
    as_regression = torch.nn.MSELoss()(damping, torch.zeros_like(damping))

    assert torch.equal(penalty, as_regression)
