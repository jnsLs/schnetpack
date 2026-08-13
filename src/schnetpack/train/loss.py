import torch
from torch import Tensor
from torch import nn
import torch.nn.functional as F

__all__ = ["MeanSquaredMagnitude", "DescendingLoss"]


class MeanSquaredMagnitude(nn.Module):
    """Penalises an output for being large, regardless of any label.

    Meant for regularisers such as the damping factor of a damped Newton step,
    which should stay as small as the system allows.
    """

    def forward(self, input: Tensor) -> Tensor:
        return input.pow(2).mean()


class DescendingLoss(nn.Module):
    """Penalises predicted steps that do not point downhill.

    The hinge is on the cosine between the predicted step and the reference
    forces, so the loss is scale invariant: it constrains the direction of the
    step only, never its length.
    """

    def __init__(self, margin=0.0, eps=1e-8, mode="hinge") -> None:
        super().__init__()
        self.margin = margin
        self.eps = eps
        self.mode = mode

    def forward(self, input: Tensor, target: Tensor) -> Tensor:

        dot = torch.sum(input * target, dim=1)
        input_norm = torch.norm(input, dim=1)
        target_norm = torch.norm(target, dim=1)

        if self.mode == "hinge":
            cos = dot / (input_norm * target_norm + self.eps)
            loss = F.relu(-cos + self.margin)
            return loss.mean()
        else:
            raise NotImplementedError("mode not implemented")
