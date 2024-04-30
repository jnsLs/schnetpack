from typing import Callable, Union, Optional

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.init import xavier_uniform_

from torch.nn.init import zeros_


__all__ = ["Dense"]


class Dense(nn.Linear):
    r"""Fully connected linear layer with activation function.

    .. math::
       y = activation(x W^T + b)
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        activation: Union[Callable, nn.Module] = None,
        weight_init: Callable = xavier_uniform_,
        bias_init: Callable = zeros_,
        xai_mod: bool = False,
    ):
        """
        Args:
            in_features: number of input feature :math:`x`.
            out_features: umber of output features :math:`y`.
            bias: If False, the layer will not adapt bias :math:`b`.
            activation: if None, no activation function is used.
            weight_init: weight initializer from current weight.
            bias_init: bias initializer from current bias.
        """
        self.xai_mod = xai_mod
        self.gamma = 0.
        self.weight_init = weight_init
        self.bias_init = bias_init
        super(Dense, self).__init__(in_features, out_features, bias)

        self.activation = activation
        if self.activation is None:
            self.activation = nn.Identity()

    def reset_parameters(self):
        self.weight_init(self.weight)
        if self.bias is not None:
            self.bias_init(self.bias)

    def _set_xai(self, xai_mod: bool, gamma: float):
        self.xai_mod = xai_mod
        self.gamma = gamma

    def forward(self, input: torch.Tensor):
        if hasattr(self, "xai_mod") and self.xai_mod:
            return self._forward_xai(input)
        else:
            y = F.linear(input, self.weight, self.bias)
            y = self.activation(y)
            return y

    def _forward_xai(self, input:torch.Tensor, rule='gamma_stable'):
        if rule == 'gen_gamma':
            # We apply the general gamma rule
            y = F.linear(input, self.weight, self.bias)
            y = self.activation(y)

            # positive output
            yp = F.linear(input.clamp(0),
                          self.weight + self.gamma*self.weight.clamp(0),
                          self.bias + self.gamma*self.bias.clamp(0)) # positive activation
            yp += F.linear( -(-input).clamp(0),
                           self.weight + self.gamma*-(-self.weight).clamp(0)) # negative activation
            yp *= (y > 1e-6).float()

            # negative output
            ym = F.linear(input.clamp(0),
                          self.weight + self.gamma*(-(-self.weight).clamp(0)),
                          self.bias + self.gamma*(-(-self.bias).clamp(0)) ) # positive activation
            ym += F.linear( -(-input).clamp(0),
                          self.weight + self.gamma*self.weight.clamp(0)) # negative activation
            ym *= (y < -1e-6 ).float()

            yo = yp + ym

            out = yo * torch.nan_to_num(y/yo).detach()

        elif rule == 'gamma_stable':
            # We use the rule where we normalize the relevance either by
            # the bias or by the sum over the activations
            y = F.linear(input, self.weight, self.bias)
            y = self.activation(y)

            # positive output
            yp = F.linear(input.clamp(0),
                          self.weight + self.gamma*self.weight.clamp(0),
                          self.bias + self.gamma*self.bias.clamp(0)) # positive activation
            yp += F.linear( -(-input).clamp(0),
                           self.weight + self.gamma*-(-self.weight).clamp(0)) # negative activation
            yp *= (y > 1e-6).float()

            # Now the denominator
            ypb = F.linear(input.clamp(0),
                          self.weight + self.gamma*self.weight.clamp(0)) # positive activation
            ypb += F.linear( -(-input).clamp(0),
                           self.weight + self.gamma*-(-self.weight).clamp(0)) # negative activation

            ypb = torch.maximum(ypb,
                    torch.vstack(ypb.shape[0]*[self.bias + self.gamma*self.bias.clamp(0)]))

            ypb *= (y > 1e-6).float()

            # negative output
            ym = F.linear(input.clamp(0),
                          self.weight + self.gamma*(-(-self.weight).clamp(0)),
                          self.bias + self.gamma*(-(-self.bias).clamp(0)) ) # positive activation

            ym += F.linear( -(-input).clamp(0),
                          self.weight + self.gamma*self.weight.clamp(0)) # negative activation
            ym *= (y < -1e-6 ).float()

            # And the denominator
            ymb = F.linear(input.clamp(0),
                          self.weight + self.gamma*(-(-self.weight).clamp(0))) # positive activation
            ymb += F.linear( -(-input).clamp(0),
                          self.weight + self.gamma*self.weight.clamp(0)) # negative activation

            ymb = torch.minimum(ymb,
                    torch.vstack(ypb.shape[0]*[ self.bias + self.gamma*(-(-self.bias).clamp(0)) ]))

            ymb *= (y < -1e-6 ).float()

            # Add positiv and negative up
            yo = yp + ym
            yob = ypb + ymb

            out = yo * torch.nan_to_num(y/yob).detach()
        else:
            raise Exception(f"{rule} rule doesn't exist.")


        return out
