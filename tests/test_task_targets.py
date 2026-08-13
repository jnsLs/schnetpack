"""Where a ``ModelOutput`` takes its target from, and when it is read.

Models write their predictions into the batch dictionary in place, so a label
sharing a name with a model output -- the ordinary case for ``energy`` and
``forces`` -- is overwritten by the forward pass. The targets must therefore be
read out before the model runs, and these tests pin that: a step that quietly
compared the prediction against itself would otherwise report a perfect loss.
"""

import pytest
import torch
from torch import nn

import schnetpack as spk
from schnetpack import properties


class _OverwritingModel(nn.Module):
    """Minimal stand-in for a potential: writes its prediction over the label."""

    def __init__(self, value=7.0):
        super().__init__()
        self.value = value
        self.weight = nn.Parameter(torch.tensor([1.0]))
        self.required_derivatives = []
        self.do_postprocessing = False

    def forward(self, inputs):
        inputs["energy"] = self.weight * torch.full_like(inputs["energy"], self.value)
        return {"energy": inputs["energy"]}


def _batch(label=1.0):
    return {
        properties.idx: torch.tensor([0, 1]),
        "energy": torch.tensor([label, label]),
    }


def _task(outputs):
    return spk.task.AtomisticTask(
        model=_OverwritingModel(), outputs=outputs, optimizer_args={"lr": 1e-3}
    )


def test_batch_target_is_read_before_the_model_overwrites_it():
    task = _task(
        [spk.task.ModelOutput(name="energy", loss_fn=nn.MSELoss(), metrics={})]
    )

    loss = task.training_step(_batch(label=1.0), 0)

    # prediction 7, label 1 -> 36, not the 0 a self-comparison would give
    assert loss.item() == pytest.approx(36.0)


def test_prediction_target_reads_the_model_output():
    task = _task(
        [
            spk.task.ModelOutput(
                name="energy",
                target_property="energy",
                target_source="prediction",
                loss_fn=nn.MSELoss(),
                metrics={},
            )
        ]
    )

    loss = task.training_step(_batch(label=1.0), 0)

    assert loss.item() == pytest.approx(0.0)


def test_prediction_targets_are_detached():
    task = _task(
        [
            spk.task.ModelOutput(
                name="energy",
                target_source="prediction",
                loss_fn=nn.MSELoss(),
                metrics={},
            )
        ]
    )
    batch = _batch()
    targets = task._collect_predicted_targets(task.predict_without_postprocessing(batch))

    assert targets["energy"].grad_fn is None


def test_unknown_target_source_is_rejected():
    with pytest.raises(ValueError, match="target_source"):
        spk.task.ModelOutput(name="energy", target_source="somewhere", metrics={})
