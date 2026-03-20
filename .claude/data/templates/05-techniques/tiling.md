# Tiling

Tiling（分块）是将大计算任务分解为小块的技术，是 Ascend NPU 性能优化的核心。

## 算子描述

**功能**：将大 tensor 分解为小块，在 UB/L0 容量内高效计算

**适用场景**：
- 大维度 tensor（超过 UB/L0 容量）
- 复杂算子（需要多个中间结果）
- 循环计算（流水线优化）

## 核心代码

### 基础 Tiling 模式

```python
import triton
import triton.language as tl


@triton.jit
def tiling_kernel(
    x_ptr,
    out_ptr,
    M,
    N,
    stride_m,
    stride_n,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """
    2D Tiling 示例
    
    每个 program 处理一个 BLOCK_M x BLOCK_N 的块
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    # 计算当前块的起始位置
    m_start = pid_m * BLOCK_M
    n_start = pid_n * BLOCK_N
    
    # 在块内循环
    for m_off in range(0, BLOCK_M, 32):  # 更小的子块
        for n_off in range(0, BLOCK_N, 32):
            m_idx = m_start + m_off + tl.arange(0, 32)
            n_idx = n_start + n_off + tl.arange(0, 32)
            
            mask = (m_idx[:, None] < M) & (n_idx[None, :] < N)
            
            x = tl.load(x_ptr + m_idx[:, None] * stride_m + n_idx[None, :] * stride_n, mask=mask, other=0.0)
            out = compute_function(x)
            
            tl.store(out_ptr + m_idx[:, None] * stride_m + n_idx[None, :] * stride_n, out, mask=mask)
```

### 循环 Tiling（1D）

```python
@triton.jit
def loop_tiling_kernel(
    x_ptr,
    out_ptr,
    n_elements: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    TILE_SIZE: tl.constexpr,
):
    """
    1D 循环 Tiling
    
    每个 program 处理一个大块，内部再分小 tiles
    """
    pid = tl.program_id(0)
    
    # 计算当前 program 的起始位置
    start = pid * BLOCK_SIZE
    
    # 在块内循环处理小 tiles
    for tile_start in range(start, start + BLOCK_SIZE, TILE_SIZE):
        offsets = tile_start + tl.arange(0, TILE_SIZE)
        mask = offsets < n_elements
        
        x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
        out = compute_function(x)
        
        tl.store(out_ptr + offsets, out, mask=mask)
```

### 固定核心数 Tiling（Ascend 推荐）

```python
import torch_npu


@triton.jit
def fixed_core_tiling_kernel(
    x_ptr,
    out_ptr,
    M,
    N,
    stride_m,
    stride_n,
    BLOCK_N: tl.constexpr,
    CORE_NUM: tl.constexpr,
):
    """
    固定核心数 + 交错循环 Tiling
    
    每个核心处理 pid, pid+CORE_NUM, pid+2*CORE_NUM, ... 行
    """
    pid = tl.program_id(0)
    
    # 交错处理多行
    for row in range(pid, M, CORE_NUM):
        row_ptr = x_ptr + row * stride_m
        out_row_ptr = out_ptr + row * stride_m
        
        # 在行内循环
        for col_start in range(0, N, BLOCK_N):
            col_offsets = col_start + tl.arange(0, BLOCK_N)
            mask = col_offsets < N
            
            x = tl.load(row_ptr + col_offsets * stride_n, mask=mask, other=0.0)
            out = compute_function(x)
            
            tl.store(out_row_ptr + col_offsets * stride_n, out, mask=mask)


def fixed_core_tiling(x: torch.Tensor) -> torch.Tensor:
    M, N = x.shape
    
    try:
        num_cores = torch_npu.npu.npu_config.get_device_limit(0).get("vector_core_num", 40)
    except:
        num_cores = 40
    
    out = torch.empty_like(x)
    BLOCK_N = 256
    
    grid = (num_cores,)
    fixed_core_tiling_kernel[grid](
        x, out, M, N,
        x.stride(0), x.stride(1),
        BLOCK_N, num_cores
    )
    
    return out
```

## UB 占用计算

```python
def calculate_ub_usage(BLOCK_SIZE, dtype, load_count=2, store_count=1, intermediate_count=0):
    """
    计算 UB 占用
    
    参数:
        BLOCK_SIZE: 分块大小
        dtype: 数据类型
        load_count: 加载的 tensor 数量
        store_count: 存储的 tensor 数量
        intermediate_count: 中间变量数量
    """
    dtype_size = {
        torch.float16: 2,
        torch.bfloat16: 2,
        torch.float32: 4,
    }
    
    size = dtype_size.get(dtype, 2)
    
    # UB = (load + store + intermediate) * BLOCK_SIZE * element_size
    ub_usage = (load_count + store_count + intermediate_count) * BLOCK_SIZE * size
    
    return ub_usage


# 示例：计算 vector add 的 UB 占用
ub = calculate_ub_usage(512, torch.float16, load_count=2, store_count=1)
print(f"UB usage: {ub / 1024:.2f} KB")  # 应该远小于 85KB
```
