import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def softmax_kernel(
    x_ptr,
    output_ptr,
    row_stride,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0)
    row_start = x_ptr + row_idx * row_stride
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols
    row = tl.load(row_start + col_offsets, mask=mask, other=float("-inf"))
    row_minus_max = row - tl.max(row, axis=0)
    numerator = tl.exp(row_minus_max)
    denominator = tl.sum(numerator, axis=0)
    softmax_output = numerator / denominator
    output_start = output_ptr + row_idx * row_stride
    tl.store(output_start + col_offsets, softmax_output, mask=mask)


def triton_softmax(x: torch.Tensor) -> torch.Tensor:
    # 保存原始shape
    original_shape = x.shape
    # reshape为2D: (batch_size * seq_len, dim)
    x_2d = x.reshape(-1, original_shape[-1])
    n_rows, n_cols = x_2d.shape
    output = torch.empty_like(x_2d)
    BLOCK_SIZE = triton.next_power_of_2(n_cols)
    grid = (n_rows,)
    softmax_kernel[grid](x_2d, output, x_2d.stride(0), n_cols, BLOCK_SIZE=BLOCK_SIZE)
    # 恢复原始shape
    return output.reshape(original_shape)


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return triton_softmax(x)
