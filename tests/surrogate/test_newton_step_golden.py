"""Regression test pinning the behaviour of the Newton-step surrogate pipeline.

The golden file was recorded from the pipeline as it stood before the cleanup
refactor (see ``generate_golden.py``). The refactor reorders no arithmetic, so
these values must be reproduced *bit for bit*. If a change legitimately moves a
value, regenerate the file and say which values moved and why in the commit
message -- do not loosen the tolerance.
"""

import pathlib

import pytest
import torch

from .golden import collect_golden

GOLDEN_PATH = (
    pathlib.Path(__file__).parent.parent / "testdata" / "newton_step_golden.pt"
)


@pytest.fixture
def golden():
    if not GOLDEN_PATH.exists():
        pytest.fail(
            f"golden file missing: {GOLDEN_PATH}\n"
            "regenerate with: python tests/surrogate/generate_golden.py"
        )
    return torch.load(GOLDEN_PATH, weights_only=False)


def test_pipeline_matches_golden(surrogate_task, newton_batch, golden):
    """Every recorded quantity is reproduced exactly."""
    actual = collect_golden(surrogate_task, newton_batch)

    assert set(actual) == set(golden), (
        f"missing from actual: {sorted(set(golden) - set(actual))}\n"
        f"unexpected in actual: {sorted(set(actual) - set(golden))}"
    )

    mismatched = []
    for key, expected in golden.items():
        try:
            torch.testing.assert_close(actual[key], expected, rtol=0, atol=0)
        except AssertionError as err:
            mismatched.append(f"--- {key}\n{err}")
    assert not mismatched, "\n".join(mismatched)


def test_golden_covers_the_whole_pipeline(golden):
    """Guard against silently recording an empty or truncated golden file."""
    groups = {key.split("/")[0] for key in golden}
    assert groups >= {"loss", "metric", "grad", "optim"}
    assert any(k.startswith("grad/") for k in golden), "no parameter gradients recorded"
    assert "damped_hvp" in golden and "ref_forces" in golden


def test_golden_detects_a_perturbed_weight(surrogate_task, newton_batch, golden):
    """Sanity-check the safety net itself: a tiny weight change must be caught."""
    with torch.no_grad():
        next(iter(surrogate_task.model.parameters())).add_(1e-4)

    actual = collect_golden(surrogate_task, newton_batch)
    assert not all(
        torch.equal(actual[k], v) for k, v in golden.items() if k in actual
    ), "the golden comparison did not notice a perturbed model weight"
