import torch
import torch.nn as nn


class Model(nn.Module):
    """
    ReLU Activation: max(0, x)
    """
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(x)


batch_size = 4096
dim = 393216


def get_inputs():
    x = torch.rand(batch_size, dim)
    return [x]


def get_init_inputs():
    return []
