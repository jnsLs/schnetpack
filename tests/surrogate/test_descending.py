"""Direct tests for the descent loss and metric.

``DescendingLoss`` is not configured in the experiment at all -- the descent
of the predicted step is only logged, through ``IsDescendingMetric`` -- so the
golden regression file cannot pin either of them. These tests do.
"""

import math

import pytest
import torch

from schnetpack.train.loss import DescendingLoss
from schnetpack.train.metrics import IsDescendingMetric


@pytest.fixture
def vectors():
    torch.manual_seed(7)
    return torch.randn(6, 3), torch.randn(6, 3)


def _mean_negative_cosine(pred, target, eps=1e-8):
    dot = (pred * target).sum(dim=1)
    norms = pred.norm(dim=1) * target.norm(dim=1) + eps
    return -dot / norms


def test_loss_is_mean_relu_of_negative_cosine(vectors):
    pred, target = vectors
    expected = torch.relu(_mean_negative_cosine(pred, target)).mean()
    torch.testing.assert_close(DescendingLoss()(pred, target), expected)


def test_loss_is_zero_for_perfectly_aligned_vectors():
    pred = torch.tensor([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0]])
    assert DescendingLoss()(pred, pred * 3.0).item() == pytest.approx(0.0, abs=1e-6)


def test_loss_is_one_for_perfectly_opposed_vectors():
    pred = torch.tensor([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0]])
    assert DescendingLoss()(pred, -pred).item() == pytest.approx(1.0, abs=1e-6)


def test_loss_is_scale_invariant(vectors):
    pred, target = vectors
    loss = DescendingLoss()
    torch.testing.assert_close(loss(pred, target), loss(pred * 17.0, target * 0.03))


def test_margin_shifts_the_hinge(vectors):
    pred, target = vectors
    margin = 0.25
    expected = torch.relu(_mean_negative_cosine(pred, target) + margin).mean()
    torch.testing.assert_close(DescendingLoss(margin=margin)(pred, target), expected)


def test_unsupported_mode_raises():
    with pytest.raises(NotImplementedError):
        DescendingLoss(mode="squared")(torch.ones(2, 3), torch.ones(2, 3))


@pytest.mark.parametrize("clamp_at_zero", [True, False])
def test_metric_matches_the_loss_when_clamped(vectors, clamp_at_zero):
    """The clamped metric is the loss; the unclamped one keeps the sign."""
    pred, target = vectors
    metric = IsDescendingMetric(clamp_at_zero=clamp_at_zero)
    metric(pred, target)

    raw = _mean_negative_cosine(pred, target)
    expected = (torch.relu(raw) if clamp_at_zero else raw).mean()
    torch.testing.assert_close(metric.compute(), expected)

    if clamp_at_zero:
        torch.testing.assert_close(metric.compute(), DescendingLoss()(pred, target))


def test_metric_accumulates_across_batches(vectors):
    """Two half-batches must give the same result as one whole batch."""
    pred, target = vectors
    whole = IsDescendingMetric()
    whole(pred, target)

    split = IsDescendingMetric()
    split(pred[:3], target[:3])
    split(pred[3:], target[3:])

    torch.testing.assert_close(whole.compute(), split.compute())


def test_unclamped_metric_can_be_negative():
    """A descending prediction gives a negative mean ascent -- the sign matters."""
    pred = torch.tensor([[1.0, 0.0, 0.0]])
    metric = IsDescendingMetric(clamp_at_zero=False)
    metric(pred, pred)
    assert metric.compute().item() == pytest.approx(-1.0, abs=1e-6)


def test_eps_keeps_zero_vectors_finite():
    zeros = torch.zeros(2, 3)
    assert math.isfinite(DescendingLoss()(zeros, torch.ones(2, 3)).item())
