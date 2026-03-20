# LayerNorm

LayerNorm（层归一化）是 Transformer 等模型中的关键组件，属于归约操作（reduction）。

## 算子描述

**功能**：对输入张量进行归一化，然后应用仿射变换

**公式**：
```
y = (x - mean) / sqrt(var + eps) * weight + bias
```

**特征**：
- 需要计算均值和方差（reduction 操作）
- 使用 float32 进行中间计算保证精度
- Ascend NPU 需要特殊处理比较操作

## 硬件约束

| 约束 | 值 | 说明 |
|------|-----|------|
| UB 占用 | ≤ 85KB/循环 | 单次循环内 UB 占用需小于此值 |
| BLOCK_N | < 65536 | 特征维度上限 |
| Vector Core | 40-48 | 获取方式：`torch_npu.npu.npu_config.get_device_limit(0).get("vector_core_num", 40)` |

### UB 占用计算

```
UB_usage = (load_x + store_out + load_weight + load_bias) * element_size + intermediate
         = 4 * BLOCK_N * 2 bytes (FP16) + 3 * BLOCK_N * 4 bytes (FP32 intermediate)
         = 8KB + 12KB = 20KB (对于 BLOCK_N=1024)
```

## 核心代码

### Kernel 实现

```python
import triton
import triton.language as tl


@triton.jit
def layernorm_kernel(
    x_ptr,
    out_ptr,
    mean_ptr,
    rstd_ptr,
    weight_ptr,
    bias_ptr,
    M,
    N,
    stride_x_row,
    stride_out_row,
    eps,
    BLOCK_N: tl.constexpr,
):
    """
    LayerNorm kernel
    
    参数:
        x_ptr: 输入张量指针 (M, N)
        out_ptr: 输出张量指针 (M, N)
        mean_ptr: 均值输出指针 (M,)
        rstd_ptr: 1/std 输出指针 (M,)
        weight_ptr: weight 指针 (N,)
        bias_ptr: bias 指针 (N,)
        M: batch 维度
        N: feature 维度
        stride_x_row: x 的行 stride
        stride_out_row: out 的行 stride
        eps: epsilon 防止除零
        BLOCK_N: 每个 program 处理的 feature 数
    """
    row = tl.program_id(0)
    
    # 计算行偏移
    x_row = x_ptr + row * stride_x_row
    out_row = out_ptr + row * stride_out_row
    
    # 列索引
    cols = tl.arange(0, BLOCK_N)
    mask = cols < N
    
    # 加载数据
    x = tl.load(x_row + cols, mask=mask, other=0.0).to(tl.float32)
    
    # 计算均值
    mean = tl.sum(x, axis=0) / N
    tl.store(mean_ptr + row, mean)
    
    # 计算方差
    # ⚠️ Ascend 特定：比较操作需要 float32
    cols_f32 = cols.to(tl.float32)
    xbar = tl.where(cols_f32 < N, x - mean, 0.0)
    var = tl.sum(xbar * xbar, axis=0) / N
    
    # 计算 1/std
    rstd = 1.0 / tl.sqrt(var + eps)
    tl.store(rstd_ptr + row, rstd)
    
    # 加载 weight 和 bias
    weight = tl.load(weight_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    bias = tl.load(bias_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    
    # 归一化 + 仿射变换
    out = (x - mean) * rstd * weight + bias
    
    # 存储结果
    tl.store(out_row + cols, out.to(out_ptr.dtype.element_ty), mask=mask)
```

### Wrapper 函数

```python
import torch


def layernorm(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    eps: float = 1e-6
) -> torch.Tensor:
    """
    LayerNorm wrapper
    
    参数:
        x: 输入张量 (M, N)
        weight: weight 张量 (N,)
        bias: bias 张量 (N,)
        eps: epsilon
    
    返回:
        输出张量 (M, N)
    """
    M, N = x.shape
    
    # 分配输出和中间结果
    out = torch.empty_like(x)
    mean = torch.empty(M, device=x.device, dtype=torch.float32)
    rstd = torch.empty(M, device=x.device, dtype=torch.float32)
    
    # 设置 BLOCK_N
    BLOCK_N = triton.next_power_of_2(N)
    
    # 启动 kernel
    grid = (M,)
    layernorm_kernel[grid](
        x, out, mean, rstd,
        weight, bias,
        M, N,
        x.stride(0), out.stride(0),
        eps,
        BLOCK_N
    )
    
    return out
```

## 使用示例

```python
import torch

M, N = 256, 1024
x = torch.randn(M, N, device='npu', dtype=torch.float16)
weight = torch.ones(N, device='npu', dtype=torch.float16)
bias = torch.zeros(N, device='npu', dtype=torch.float16)

out = layernorm(x, weight, bias, eps=1e-6)

# 与 PyTorch 对比
expected = torch.nn.functional.layer_norm(x, (N,), weight, bias, eps=1e-6)
assert torch.allclose(out, expected, rtol=1e-3, atol=1e-3)
print("✅ LayerNorm 正确性验证通过")
```

## 变体

### 无 weight/bias 的简化版

```python
@triton.jit
def layernorm_simple_kernel(
    x_ptr,
    out_ptr,
    mean_ptr,
    rstd_ptr,
    M,
    N,
    stride_x_row,
    stride_out_row,
    eps,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    
    x_row = x_ptr + row * stride_x_row
    out_row = out_ptr + row * stride_out_row
    
    cols = tl.arange(0, BLOCK_N)
    mask = cols < N
    
    x = tl.load(x_row + cols, mask=mask, other=0.0).to(tl.float32)
    
    mean = tl.sum(x, axis=0) / N
    tl.store(mean_ptr + row, mean)
    
    cols_f32 = cols.to(tl.float32)
    xbar = tl.where(cols_f32 < N, x - mean, 0.0)
    var = tl.sum(xbar * xbar, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + eps)
    tl.store(rstd_ptr + row, rstd)
    
    out = (x - mean) * rstd
    tl.store(out_row + cols, out.to(out_ptr.dtype.element_ty), mask=mask)


def layernorm_simple(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    M, N = x.shape
    
    out = torch.empty_like(x)
    mean = torch.empty(M, device=x.device, dtype=torch.float32)
    rstd = torch.empty(M, device=x.device, dtype=torch.float32)
    
    BLOCK_N = triton.next_power_of_2(N)
    
    grid = (M,)
    layernorm_simple_kernel[grid](
        x, out, mean, rstd,
        M, N,
        x.stride(0), out.stride(0),
        eps,
        BLOCK_N
    )
    
    return out
```

### 固定核心数启动（Ascend 推荐）

```python
import torch_npu


@triton.jit
def layernorm_fixed_core_kernel(
    x_ptr,
    out_ptr,
    weight_ptr,
    bias_ptr,
    mean_ptr,
    rstd_ptr,
    N,
    stride_x_row,
    stride_out_row,
    eps,
    BLOCK_N: tl.constexpr,
    CORE_NUM: tl.constexpr,
):
    """
    固定核心数版本：每个核心处理多行
    """
    pid = tl.program_id(0)
    
    # 交错处理多行
    for row in range(pid, tl.num_programs(0), CORE_NUM):
        x_row = x_ptr + row * stride_x_row
        out_row = out_ptr + row * stride_out_row
        
        cols = tl.arange(0, BLOCK_N)
        mask = cols < N
        
        x = tl.load(x_row + cols, mask=mask, other=0.0).to(tl.float32)
        
        mean = tl.sum(x, axis=0) / N
        tl.store(mean_ptr + row, mean)
        
        cols_f32 = cols.to(tl.float32)
        xbar = tl.where(cols_f32 < N, x - mean, 0.0)
        var = tl.sum(xbar * xbar, axis=0) / N
        rstd = 1.0 / tl.sqrt(var + eps)
        tl.store(rstd_ptr + row, rstd)
        
        weight = tl.load(weight_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        bias = tl.load(bias_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        
        out = (x - mean) * rstd * weight + bias
        tl.store(out_row + cols, out.to(out_ptr.dtype.element_ty), mask=mask)


def layernorm_fixed_core(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    eps: float = 1e-6
) -> torch.Tensor:
    M, N = x.shape
    
    out = torch.empty_like(x)
    mean = torch.empty(M, device=x.device, dtype=torch.float32)
    rstd = torch.empty(M, device=x.device, dtype=torch.float32)
    
    try:
        num_cores = torch_npu.npu.npu_config.get_device_limit(0).get("vector_core_num", 40)
    except:
        num_cores = 40
    
    BLOCK_N = triton.next_power_of_2(N)
    
    grid = (num_cores,)
    layernorm_fixed_core_kernel[grid](
        x, out, weight, bias, mean, rstd,
        N,
        x.stride(0), out.stride(0),
        eps,
        BLOCK_N,
        num_cores
    )
    
    return out
```

## 常见错误

| 错误 | 原因 | 解决方案 |
|------|------|---------|
| 精度损失 | 使用低精度计算 | 使用 float32 进行中间计算 |
| 性能极差 | int64 比较 | 使用 `cols.to(tl.float32)` 转换 |
| UB 溢出 | BLOCK_N 过大 | 减小 BLOCK_N 或使用多循环 |
| 结果不一致 | 输入不是连续内存 | 使用 `.contiguous()` 或确保输入连续 |
