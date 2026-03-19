import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def relu_kernel(
    x_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    output = tl.where(x > 0, x, 0.0)
    tl.store(output_ptr + offsets, output, mask=mask)


def triton_relu(x: torch.Tensor) -> torch.Tensor:
    n_elements = x.numel()
    output = torch.empty_like(x)
    BLOCK_SIZE = 128
    grid = ((n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE,)
    relu_kernel[grid](x, output, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    return output


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return triton_relu(x)
