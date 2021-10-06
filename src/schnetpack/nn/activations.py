import numpy as np
from torch.nn import functional


def shifted_softplus(x):
    r"""Compute shifted soft-plus activation function.

    .. math::
       y = \ln\left(1 + e^{-x}\right) - \ln(2)

    Args:
        x (torch.Tensor): input tensor.

    Returns:
        torch.Tensor: shifted soft-plus of input.

    """
    return functional.softplus(x) - np.log(2.0)


def relu(x):
    return(functional.relu(x))


def get_activation_by_string(key):
    if key == "relu":
        activation = relu
    elif key == "none":
        activation = None
    elif key == "ssp":
        activation = shifted_softplus
    else:
        raise NotImplementedError("activation_function {} is unknown".format(key))
    return activation
