import torch
import torch.nn as nn


class Model(nn.Module):
    """
    Softmax: exp(x_i) / sum(exp(x_j))
    """
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.softmax(x, dim=-1)


batch_size = 2048
seq_len = 512
dim = 64


def get_inputs():
    x = torch.rand(batch_size, seq_len, dim)
    return [x]


def get_init_inputs():
    return []
