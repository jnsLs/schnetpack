from copy import deepcopy
import torch.nn as nn
import torch
from typing import Dict, Optional, List
from pytorch_lightning import LightningModule


__all__ = ["NNEnsemble"]


class NNEnsemble(LightningModule):
    def __init__(self, models: nn.ModuleList, properties: List[str]):
        super(NNEnsemble, self).__init__()
        self.models = models
        if type(properties) == str:
            properties = [properties]
        self.properties = properties

    def setup(self, stage: Optional[str] = None) -> None:
        for model in self.models:
            model.setup(stage)

    def forward(
        self,
        x,
    ):
        results = {}
        for p in self.properties:
            results[p] = []

        for model in self.models:
            x_tmp = deepcopy(x)
            predictions = model(x_tmp)
            for prop, values in predictions.items():
                if prop in self.properties:
                    results[prop].append(values.detach())

        means = {}
        stds = {}
        for prop, values in results.items():
            stacked_values = torch.stack(values)
            means[prop] = stacked_values.mean(0)
            stds[prop] = stacked_values.std(0)

        return means, stds