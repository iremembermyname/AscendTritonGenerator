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
    # 每个core处理多个数据块（grid-stride loop）
    pid = tl.program_id(0)
    num_cores = tl.num_programs(0)

    # 当前core处理的所有block
    for block_idx in range(pid, (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE, num_cores):
        block_start = block_idx * BLOCK_SIZE
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
        output = tl.where(x > 0, x, 0.0)
        tl.store(output_ptr + offsets, output, mask=mask)


def triton_relu(x: torch.Tensor) -> torch.Tensor:
    n_elements = x.numel()
    output = torch.empty_like(x)
    BLOCK_SIZE = 128
    # NPU只有20个核心，限制grid大小
    NPU_CORES = 20
    grid_size = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    grid = (min(grid_size, NPU_CORES),)
    relu_kernel[grid](x, output, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    return output


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return triton_relu(x)
