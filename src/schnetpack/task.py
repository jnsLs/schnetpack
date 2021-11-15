from pathlib import Path
import torchmetrics.regression
from typing import Optional, Dict, List, Type, Any, Union
import pytorch_lightning as pl
from torch import Tensor
from torchmetrics.functional.regression.mean_absolute_error import (
    _mean_absolute_error_compute,
    _mean_absolute_error_update,
)
from torch.nn import functional as F
import torch
from torch import nn as nn
from torchmetrics import Metric

from schnetpack.model.base import AtomisticModel

__all__ = ["ModelOutput", "AtomisticTask", "SelectedAtomsMAE", "SelectedAtomsMSELoss"]


class ModelOutput(nn.Module):
    """
    Defines an output of a model, including mappings to a loss function and weight for training
    and metrics to be logged.
    """

    def __init__(
        self,
        name: str,
        loss_fn: Optional[nn.Module] = None,
        loss_weight: float = 1.0,
        metrics: Optional[Dict[str, Metric]] = None,
        target_name: Optional[str] = None,
        preprocessing: Optional[nn.Module] = None,
    ):
        """
        Args:
            name: name of output in results dict
            target_name: Name of target in training batch. Only required for supervised training.
                If not given, the output name is assumed to also be the target name.
            loss_fn: function to compute the loss
            loss_weight: loss weight in the composite loss: $l = w_1 l_1 + \dots + w_n l_n$
            metrics: dictionary of
        """
        super().__init__()
        self.property = name
        self.target_name = target_name or name
        self.loss_fn = loss_fn
        self.loss_weight = loss_weight
        self.metrics = nn.ModuleDict(metrics)
        self.preprocessing = preprocessing


class AtomisticTask(pl.LightningModule):
    """
    Defines a learning task in SchNetPack including model, losses and optimizer.

    """

    def __init__(
        self,
        model: AtomisticModel,
        outputs: List[ModelOutput],
        optimizer_cls: Type[torch.optim.Optimizer] = torch.optim.Adam,
        optimizer_args: Optional[Dict[str, Any]] = None,
        scheduler_cls: Optional[Type] = None,
        scheduler_args: Optional[Dict[str, Any]] = None,
        scheduler_monitor: Optional[str] = None,
    ):
        """
        Args:
            outputs: list of outputs an optional loss functions
            optimizer_cls: type of torch optimizer,e.g. torch.optim.Adam
            optimizer_args: dict of optimizer keyword arguments
            scheduler_cls: type of torch learning rate scheduler
            scheduler_args: dict of scheduler keyword arguments
            scheduler_monitor: name of metric to be observed for ReduceLROnPlateau
        """
        super().__init__()
        self.model = model
        self.optimizer_cls = optimizer_cls
        self.optimizer_kwargs = optimizer_args
        self.scheduler_cls = scheduler_cls
        self.scheduler_kwargs = scheduler_args
        self.schedule_monitor = scheduler_monitor
        self.outputs = nn.ModuleList(outputs)

        self.grad_enabled = len(self.model.required_derivatives) > 0
        self.inference_mode = False

    def setup(self, stage=None):
        if stage == "fit":
            self.model.initialize_postprocessors(self.trainer.datamodule)

    def forward(self, inputs: Dict[str, torch.Tensor]):
        results = self.model(inputs)
        return results

    def loss_fn(self, pred, batch):
        loss = 0.0
        for output in self.outputs:

            # preprocessing
            if output.preprocessing is not None:
                if output.property != output.target_name:
                    raise ValueError("property name is not consistent in results and input dict")
                pred, batch = output.preprocessing(pred, batch, output.target_name)

            loss_p = output.loss_weight * output.loss_fn(
                pred[output.property], batch[output.target_name]
            )
            loss += loss_p
        return loss

    def log_metrics(self, pred, targets, subset):
        for output in self.outputs:
            for metric_name, pmetric in output.metrics.items():
                self.log(
                    f"{subset}_{output.property}_{metric_name}",
                    pmetric(pred[output.property], targets[output.target_name]),
                    on_step=False,
                    on_epoch=True,
                    prog_bar=False,
                )

    def training_step(self, batch, batch_idx):
        targets = {
            output.target_name: batch[output.target_name] for output in self.outputs
        }
        pred = self(batch)
        loss = self.loss_fn(pred, targets)

        self.log("train_loss", loss, on_step=True, on_epoch=False, prog_bar=False)
        self.log_metrics(targets, pred, "train")
        return loss

    def validation_step(self, batch, batch_idx):
        torch.set_grad_enabled(self.grad_enabled)
        targets = {
            output.target_name: batch[output.target_name] for output in self.outputs
        }
        pred = self(batch)
        loss = self.loss_fn(pred, targets)

        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log_metrics(targets, pred, "val")

        return {"val_loss": loss}

    def test_step(self, batch, batch_idx):
        torch.set_grad_enabled(self.grad_enabled)
        targets = {
            output.target_name: batch[output.target_name] for output in self.outputs
        }
        pred = self(batch)
        loss = self.loss_fn(pred, targets)

        self.log("test_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log_metrics(targets, pred, "test")
        return {"test_loss": loss}

    def configure_optimizers(self):
        optimizer = self.optimizer_cls(
            params=self.parameters(), **self.optimizer_kwargs
        )

        if self.scheduler_cls:
            schedule = self.scheduler_cls(optimizer=optimizer, **self.scheduler_kwargs)

            optimconf = {"scheduler": schedule, "name": "lr_schedule"}
            if self.schedule_monitor:
                optimconf["monitor"] = self.schedule_monitor
            return [optimizer], [optimconf]
        else:
            return optimizer

    def to_torchscript(
        self,
        file_path: Optional[Union[str, Path]] = None,
        method: Optional[str] = "script",
        example_inputs: Optional[Any] = None,
        **kwargs,
    ) -> Union[torch.ScriptModule, Dict[str, torch.ScriptModule]]:
        imode = self.inference_mode
        self.inference_mode = True
        script = super().to_torchscript(file_path, method, example_inputs, **kwargs)
        self.inference_mode = imode
        return script


class SelectedAtomsMAE(torchmetrics.regression.MeanAbsoluteError):

    def __init__(self, considered_atoms=None):

        super().__init__()

        if considered_atoms is None:
            # self.considered_atoms = [_ for _ in range(1008, 1046)]
            self.considered_atoms = [_ for _ in range(144, 182)]
        else:
            self.considered_atoms = considered_atoms

    def update(self, preds: Tensor, target: Tensor):
        """
        Update state with predictions and targets.

        Args:
            preds: Predictions from model
            target: Ground truth values
        """

        sum_abs_error, n_obs = _mean_absolute_error_update(preds[self.considered_atoms], target[self.considered_atoms])

        self.sum_abs_error += sum_abs_error
        self.total += n_obs


class SelectedAtomsMSELoss(nn.MSELoss):

    def __init__(self, considered_atoms=None):
        super().__init__()

    def forward(self, input: Tensor, target: Tensor) -> Tensor:

        batch_size = input.shape[0] // 238
        if batch_size - input.shape[0] / 238 > 1e-9:
            raise ImportError("each sample must contain 238 nodes")
        considered_atoms = []
        for spl_idx in range(batch_size):
            considered_atoms += [_ for _ in range(144+238*spl_idx, 182+238*spl_idx)]

        new_input = input[considered_atoms]
        new_target = target[considered_atoms]

        return F.mse_loss(new_input, new_target, reduction=self.reduction)


class ConsiderOnlySelectedAtoms(nn.Module):

    def __init__(self, considered_atoms_path):
        super().__init__()
        self.considered_atoms = torch.load(considered_atoms_path)

    def forward(self, pred, batch, target_name):

        device = "cuda" if pred["_n_atoms"].is_cuda else "cpu"
        self.considered_atoms = self.considered_atoms.to(device=device)

        considered_atoms = []
        for spl_idx, n_atoms in enumerate(pred["_n_atoms"]):
            considered_atoms += (self.considered_atoms + n_atoms * spl_idx).tolist()

        pred[target_name] = pred[target_name][considered_atoms]
        batch[target_name] = batch[target_name][considered_atoms]

        return pred, batch
