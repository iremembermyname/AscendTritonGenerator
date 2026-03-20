# Softmax

Softmax 是深度学习中的基础归一化操作，属于归约操作（reduction）。

## 算子描述

**功能**：将输入向量归一化为概率分布

**公式**：
```
softmax(x)_i = exp(x_i - max(x)) / sum(exp(x_j - max(x)))
```

**特征**：
- 需要计算最大值（reduction 操作）
- 需要计算指数和（reduction 操作）
- 使用 max 减除保证数值稳定性
- 多循环处理大维度

## 硬件约束

| 约束 | 值 | 说明 |
|------|-----|------|
| UB 占用 | ≤ 85KB/循环 | 单次循环内 UB 占用需小于此值 |
| BLOCK_N | 推荐 512-1024 | 单次循环处理的元素数 |

### UB 占用计算

```
UB_usage = (load_x + store_out) * element_size + intermediate
         = 2 * BLOCK_N * 2 bytes (FP16) + 2 * BLOCK_N * 4 bytes (FP32 intermediate)
         = 4KB + 8KB = 12KB (对于 BLOCK_N=1024)
```

## 核心代码

### Kernel 实现（多循环版本）

```python
import triton
import triton.language as tl


@triton.jit
def softmax_kernel(
    x_ptr,
    out_ptr,
    M,
    N,
    stride_m,
    BLOCK_N: tl.constexpr,
):
    """
    Softmax kernel
    
    参数:
        x_ptr: 输入张量指针 (M, N)
        out_ptr: 输出张量指针 (M, N)
        M: batch 维度
        N: feature 维度
        stride_m: 行 stride
        BLOCK_N: 每个 program 处理的 feature 数
    """
    row = tl.program_id(0)
    
    row_start = row * stride_m
    
    # 第一遍：找最大值
    max_val = float("-inf")
    for block_start in range(0, N, BLOCK_N):
        offsets = block_start + tl.arange(0, BLOCK_N)
        mask = offsets < N
        x = tl.load(x_ptr + row_start + offsets, mask=mask, other=float("-inf"))
        max_val = tl.maximum(max_val, tl.max(x, axis=0))
    
    # 第二遍：计算 exp 和
    sum_exp = 0.0
    for block_start in range(0, N, BLOCK_N):
        offsets = block_start + tl.arange(0, BLOCK_N)
        mask = offsets < N
        x = tl.load(x_ptr + row_start + offsets, mask=mask, other=0.0)
        exp_x = tl.exp(x - max_val)
        sum_exp += tl.sum(exp_x, axis=0)
    
    # 第三遍：归一化
    for block_start in range(0, N, BLOCK_N):
        offsets = block_start + tl.arange(0, BLOCK_N)
        mask = offsets < N
        x = tl.load(x_ptr + row_start + offsets, mask=mask, other=0.0)
        exp_x = tl.exp(x - max_val)
        out = exp_x / sum_exp
        tl.store(out_ptr + row_start + offsets, out, mask=mask)
```

### Wrapper 函数

```python
import torch


def softmax(x: torch.Tensor) -> torch.Tensor:
    """
    Softmax wrapper
    
    参数:
        x: 输入张量 (M, N)
    
    返回:
        输出张量 (M, N)
    """
    M, N = x.shape
    
    out = torch.empty_like(x)
    
    BLOCK_N = 1024
    
    grid = (M,)
    softmax_kernel[grid](x, out, M, N, x.stride(0), BLOCK_N)
    
    return out
```

## 使用示例

```python
import torch

M, N = 256, 2048
x = torch.randn(M, N, device='npu', dtype=torch.float16)

out = softmax(x)

# 与 PyTorch 对比
expected = torch.nn.functional.softmax(x, dim=-1)
assert torch.allclose(out, expected, rtol=1e-3, atol=1e-3)
print("✅ Softmax 正确性验证通过")
```

## 变体

### 单行版本（N 较小）

```python
@triton.jit
def softmax_single_kernel(
    x_ptr,
    out_ptr,
    N,
    BLOCK_N: tl.constexpr,
):
    """
    单行 softmax 版本，适用于 N 较小的情况
    """
    pid = tl.program_id(0)
    
    offsets = tl.arange(0, BLOCK_N)
    mask = offsets < N
    
    x = tl.load(x_ptr + pid * N + offsets, mask=mask, other=0.0).to(tl.float32)
    
    # 找最大值
    max_val = tl.max(x, axis=0)
    
    # 计算 exp
    exp_x = tl.exp(x - max_val)
    
    # 计算和
    sum_exp = tl.sum(exp_x, axis=0)
    
    # 归一化
    out = exp_x / sum_exp
    
    tl.store(out_ptr + pid * N + offsets, out, mask=mask)


def softmax_single(x: torch.Tensor) -> torch.Tensor:
    N = x.shape[-1]
    out = torch.empty_like(x)
    
    BLOCK_N = triton.next_power_of_2(N)
    
    # 如果是 2D tensor
    if x.dim() == 2:
        M = x.shape[0]
        grid = (M,)
    else:
        # 展平为 2D
        M = x.numel() // N
        x_flat = x.view(M, N).contiguous()
        out_flat = out.view(M, N)
        grid = (M,)
    
    softmax_single_kernel[grid](x, out, N, BLOCK_N)
    
    return out
```

### 固定核心数启动（Ascend 推荐）

```python
import torch_npu


@triton.jit
def softmax_fixed_core_kernel(
    x_ptr,
    out_ptr,
    M,
    N,
    stride_m,
    BLOCK_N: tl.constexpr,
    CORE_NUM: tl.constexpr,
):
    """
    固定核心数版本：每个核心处理多行
    """
    pid = tl.program_id(0)
    
    # 交错处理多行
    for row in range(pid, M, CORE_NUM):
        row_start = row * stride_m
        
        # 第一遍：找最大值
        max_val = float("-inf")
        for block_start in range(0, N, BLOCK_N):
            offsets = block_start + tl.arange(0, BLOCK_N)
            mask = offsets < N
            x = tl.load(x_ptr + row_start + offsets, mask=mask, other=float("-inf"))
            max_val = tl.maximum(max_val, tl.max(x, axis=0))
        
        # 第二遍：计算 exp 和
        sum_exp = 0.0
        for block_start in range(0, N, BLOCK_N):
            offsets = block_start + tl.arange(0, BLOCK_N)
            mask = offsets < N
            x = tl.load(x_ptr + row_start + offsets, mask=mask, other=0.0)
            exp_x = tl.exp(x - max_val)
            sum_exp += tl.sum(exp_x, axis=0)
        
        # 第三遍：归一化
        for block_start in range(0, N, BLOCK_N):
            offsets = block_start + tl.arange(0, BLOCK_N)
            mask = offsets < N
            x = tl.load(x_ptr + row_start + offsets, mask=mask, other=0.0)
            exp_x = tl.exp(x - max_val)
            out = exp_x / sum_exp
            tl.store(out_ptr + row_start + offsets, out, mask=mask)


def softmax_fixed_core(x: torch.Tensor) -> torch.Tensor:
    M, N = x.shape
    
    out = torch.empty_like(x)
    
    try:
        num_cores = torch_npu.npu.npu_config.get_device_limit(0).get("vector_core_num", 40)
    except:
        num_cores = 40
    
    BLOCK_N = 1024
    
    grid = (num_cores,)
    softmax_fixed_core_kernel[grid](x, out, M, N, x.stride(0), BLOCK_N, num_cores)
    
    return out
```

## 常见错误

| 错误 | 原因 | 解决方案 |
|------|------|---------|
| 数值溢出 | 未减除最大值 | 使用 `exp(x - max_val)` 而非 `exp(x)` |
| 精度损失 | 使用低精度计算 | 使用 float32 进行中间计算 |
| UB 溢出 | BLOCK_N 过大 | 使用多循环，减小单次循环的 BLOCK_N |
| 结果不一致 | 输入不是连续内存 | 使用 `.contiguous()` 或确保输入连续 |
