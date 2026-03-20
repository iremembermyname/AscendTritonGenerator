# Vector Add

向量加法是最基础的逐元素操作（element-wise），是学习 Triton 的入门算子。

## 算子描述

**功能**：对应元素相加 `z = x + y`

**特征**：
- 不使用 `tl.dot`
- 不需要归约操作
- 完全并行，每个元素独立计算

## 硬件约束

| 约束 | 值 | 说明 |
|------|-----|------|
| UB 占用 | ≤ 85KB/循环 | 单次循环内 UB 占用需小于此值 |
| BLOCK_SIZE | 推荐 512-1024 | 太大可能导致 UB 溢出 |

### UB 占用计算

```
UB_usage = (load_x + load_y + store_z) * element_size
         = 3 * BLOCK_SIZE * 2 bytes (FP16)
         = 3 * 512 * 2 = 3KB (远小于 85KB 限制)
```

## 核心代码

### Kernel 实现

```python
import triton
import triton.language as tl


@triton.jit
def vector_add_kernel(
    x_ptr,
    y_ptr,
    z_ptr,
    vector_len: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    向量加法 kernel
    
    参数:
        x_ptr: 输入向量 x 的指针
        y_ptr: 输入向量 y 的指针
        z_ptr: 输出向量 z 的指针
        vector_len: 向量长度
        BLOCK_SIZE: 每个 program 处理的最大元素数
    """
    pid = tl.program_id(axis=0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset < vector_len
    
    x = tl.load(x_ptr + offset, mask=mask, other=0.0)
    y = tl.load(y_ptr + offset, mask=mask, other=0.0)
    z = x + y
    
    tl.store(z_ptr + offset, z, mask=mask)
```

### Wrapper 函数

```python
import torch


def vector_add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    向量加法 wrapper
    
    参数:
        x: 输入张量 shape (N,) 或 (B, N)
        y: 输入张量，shape 必须与 x 相同
    
    返回:
        输出张量，shape 与 x 相同
    """
    assert x.shape == y.shape, "输入 shape 必须相同"
    assert x.is_contiguous(), "输入张量必须是连续的"
    
    z = torch.empty_like(x)
    vector_len = x.numel()
    BLOCK_SIZE = 512
    
    grid = (triton.cdiv(vector_len, BLOCK_SIZE),)
    vector_add_kernel[grid](x, y, z, vector_len, BLOCK_SIZE)
    
    return z
```

## 使用示例

```python
import torch

x = torch.randn(1024, device='npu', dtype=torch.float16)
y = torch.randn(1024, device='npu', dtype=torch.float16)

z = vector_add(x, y)

expected = x + y
assert torch.allclose(z, expected, rtol=1e-3, atol=1e-3)
print("✅ 正确性验证通过")
```

## 变体

### 2D Tensor 矩阵逐元素相加

```python
@triton.jit
def matrix_add_kernel(
    x_ptr,
    y_ptr,
    z_ptr,
    M,
    N,
    stride_m,
    stride_n,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    offsets_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offsets_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    
    mask = (offsets_m[:, None] < M) & (offsets_n[None, :] < N)
    
    x = tl.load(x_ptr + offsets_m[:, None] * stride_m + offsets_n[None, :] * stride_n, mask=mask, other=0.0)
    y = tl.load(y_ptr + offsets_m[:, None] * stride_m + offsets_n[None, :] * stride_n, mask=mask, other=0.0)
    z = x + y
    
    tl.store(z_ptr + offsets_m[:, None] * stride_m + offsets_n[None, :] * stride_n, z, mask=mask)


def matrix_add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    M, N = x.shape
    BLOCK_M = 32
    BLOCK_N = 64
    
    z = torch.empty_like(x)
    
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    matrix_add_kernel[grid](
        x, y, z,
        M, N,
        x.stride(0), x.stride(1),
        BLOCK_M, BLOCK_N
    )
    
    return z
```

### 固定核心数启动（Ascend 推荐）

```python
import torch_npu


@triton.jit
def vector_add_fixed_core_kernel(
    x_ptr,
    y_ptr,
    z_ptr,
    vector_len: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    CORE_NUM: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    
    # 交错循环：每个核心处理 pid, pid+CORE_NUM, pid+2*CORE_NUM, ...
    for offset in range(pid, vector_len, CORE_NUM * BLOCK_SIZE):
        offsets = offset + tl.arange(0, BLOCK_SIZE)
        mask = offsets < vector_len
        
        x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
        y = tl.load(y_ptr + offsets, mask=mask, other=0.0)
        z = x + y
        
        tl.store(z_ptr + offsets, z, mask=mask)


def vector_add_fixed_core(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    try:
        num_cores = torch_npu.npu.npu_config.get_device_limit(0).get("vector_core_num", 40)
    except:
        num_cores = 40
    
    z = torch.empty_like(x)
    vector_len = x.numel()
    BLOCK_SIZE = 512
    
    grid = (num_cores,)
    vector_add_fixed_core_kernel[grid](x, y, z, vector_len, BLOCK_SIZE, num_cores)
    
    return z
```

## 常见错误

| 错误 | 原因 | 解决方案 |
|------|------|---------|
| 结果全为 0 | mask 设置错误或未使用 mask | 确保 `mask = offset < vector_len` |
| UB 溢出 | BLOCK_SIZE 过大 | 减小 BLOCK_SIZE 至 512 或 256 |
| 结果不一致 | 输入不是连续内存 | 使用 `.contiguous()` 或确保输入连续 |
