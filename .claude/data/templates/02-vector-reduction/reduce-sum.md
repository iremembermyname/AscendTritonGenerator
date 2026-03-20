# Reduce Sum

Reduce Sum 是最基础的归约操作，用于计算张量元素的和。

## 算子描述

**功能**：计算张量沿某个维度的和

**特征**：
- 需要归约操作（`tl.sum`）
- 可能需要原子操作（`tl.atomic_add`）进行跨 block 累加
- 简单但需要注意并行化策略

## 硬件约束

| 约束 | 值 | 说明 |
|------|-----|------|
| UB 占用 | ≤ 85KB/循环 | 单次循环内 UB 占用需小于此值 |
| BLOCK_SIZE | 推荐 512-1024 | 太大可能导致 UB 溢出 |

### UB 占用计算

```
UB_usage = (load_x + store_sum) * element_size
         = 2 * BLOCK_SIZE * 2 bytes (FP16)
         = 4KB (对于 BLOCK_SIZE=1024)
```

## 核心代码

### 单 Block 版本（小 tensor）

```python
import triton
import triton.language as tl


@triton.jit
def reduce_sum_kernel(
    x_ptr,
    sum_ptr,
    n_elements: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Reduce Sum kernel
    
    参数:
        x_ptr: 输入张量指针
        sum_ptr: 输出和的指针
        n_elements: 元素总数
        BLOCK_SIZE: 每个 program 处理的元素数
    """
    pid = tl.program_id(axis=0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset < n_elements
    
    x = tl.load(x_ptr + offset, mask=mask, other=0.0).to(tl.float32)
    
    # 计算 block 内的和
    block_sum = tl.sum(x, axis=0)
    
    # 使用原子操作累加到全局
    tl.atomic_add(sum_ptr, block_sum)


def reduce_sum(x: torch.Tensor) -> torch.Tensor:
    """
    Reduce Sum wrapper
    
    参数:
        x: 输入张量
    
    返回:
        标量和
    """
    n_elements = x.numel()
    
    # 分配输出（单个 float32）
    sum_out = torch.zeros(1, device=x.device, dtype=torch.float32)
    
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    reduce_sum_kernel[grid](x, sum_out, n_elements, BLOCK_SIZE)
    
    return sum_out[0]
```

### 分层归约版本（大 tensor）

```python
@triton.jit
def reduce_sum_block_kernel(
    x_ptr,
    partial_sum_ptr,
    n_elements: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    第一层：每个 block 计算部分和
    """
    pid = tl.program_id(axis=0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset < n_elements
    
    x = tl.load(x_ptr + offset, mask=mask, other=0.0).to(tl.float32)
    
    block_sum = tl.sum(x, axis=0)
    
    # 存储部分和到数组
    tl.store(partial_sum_ptr + pid, block_sum)


@triton.jit
def reduce_sum_final_kernel(
    partial_sum_ptr,
    sum_ptr,
    n_blocks: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    第二层：归约所有部分和
    """
    pid = tl.program_id(axis=0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset < n_blocks
    
    partial_sums = tl.load(partial_sum_ptr + offset, mask=mask, other=0.0).to(tl.float32)
    
    block_sum = tl.sum(partial_sums, axis=0)
    
    tl.atomic_add(sum_ptr, block_sum)


def reduce_sum_hierarchical(x: torch.Tensor) -> torch.Tensor:
    n_elements = x.numel()
    
    BLOCK_SIZE = 1024
    n_blocks = triton.cdiv(n_elements, BLOCK_SIZE)
    
    # 第一层：部分和
    partial_sums = torch.empty(n_blocks, device=x.device, dtype=torch.float32)
    
    reduce_sum_block_kernel[grid](x, partial_sums, n_elements, BLOCK_SIZE)
    
    # 第二层：最终和
    sum_out = torch.zeros(1, device=x.device, dtype=torch.float32)
    
    final_grid = (triton.cdiv(n_blocks, BLOCK_SIZE),)
    reduce_sum_final_kernel[final_grid](partial_sums, sum_out, n_blocks, BLOCK_SIZE)
    
    return sum_out[0]
```

### 沿维度归约（Row-wise Sum）

```python
@triton.jit
def row_sum_kernel(
    x_ptr,
    sum_ptr,
    M,
    N,
    stride_m,
    BLOCK_N: tl.constexpr,
):
    """
    沿最后一个维度求和：sum(x, dim=-1)
    
    参数:
        x_ptr: 输入张量指针 (M, N)
        sum_ptr: 输出和指针 (M,)
        M: batch 维度
        N: feature 维度
        stride_m: 行 stride
        BLOCK_N: 每个 program 处理的 feature 数
    """
    row = tl.program_id(0)
    
    row_start = row * stride_m
    
    total_sum = 0.0
    
    for block_start in range(0, N, BLOCK_N):
        offsets = block_start + tl.arange(0, BLOCK_N)
        mask = offsets < N
        x = tl.load(x_ptr + row_start + offsets, mask=mask, other=0.0).to(tl.float32)
        total_sum += tl.sum(x, axis=0)
    
    tl.store(sum_ptr + row, total_sum)


def row_sum(x: torch.Tensor) -> torch.Tensor:
    """
    沿最后一个维度求和
    
    参数:
        x: 输入张量 (M, N)
    
    返回:
        和 (M,)
    """
    M, N = x.shape
    
    sum_out = torch.empty(M, device=x.device, dtype=torch.float32)
    
    BLOCK_N = 1024
    
    grid = (M,)
    row_sum_kernel[grid](x, sum_out, M, N, x.stride(0), BLOCK_N)
    
    return sum_out
```

## 使用示例

```python
import torch

# 全局求和
x = torch.randn(1024, device='npu', dtype=torch.float16)
result = reduce_sum(x)
expected = x.sum()
assert torch.allclose(result, expected, rtol=1e-3, atol=1e-3)
print("✅ 全局求和正确性验证通过")

# 沿维度求和
x_2d = torch.randn(256, 1024, device='npu', dtype=torch.float16)
result = row_sum(x_2d)
expected = x_2d.sum(dim=-1)
assert torch.allclose(result, expected, rtol=1e-3, atol=1e-3)
print("✅ 沿维度求和正确性验证通过")
```

## 常见错误

| 错误 | 原因 | 解决方案 |
|------|------|---------|
| 结果偏小 | 未使用原子操作累加 | 使用 `tl.atomic_add` |
| 精度损失 | 使用低精度累加 | 使用 float32 进行累加 |
| UB 溢出 | BLOCK_SIZE 过大 | 减小 BLOCK_SIZE 或使用多循环 |
| 结果不一致 | 输入不是连续内存 | 使用 `.contiguous()` 或确保输入连续 |
