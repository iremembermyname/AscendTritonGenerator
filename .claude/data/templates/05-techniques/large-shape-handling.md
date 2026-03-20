# Large Shape Handling

大 Shape 处理是 Ascend NPU 算子开发中的常见挑战，需要特殊的优化策略。

## 问题描述

当 tensor 的 shape 超过硬件限制时：
- UB 溢出：单次循环无法容纳所有数据
- Grid 超限：grid 大小超过 65535
- 性能下降：资源利用率低

## 解决方案

### 1. 固定核心数 + 交错循环

```python
import triton
import triton.language as tl
import torch_npu


@triton.jit
def large_shape_kernel(
    x_ptr,
    out_ptr,
    M,
    N,
    stride_m,
    stride_n,
    BLOCK_N: tl.constexpr,
    CORE_NUM: tl.constexpr,
):
    pid = tl.program_id(0)
    
    # 交错处理多行
    for row in range(pid, M, CORE_NUM):
        row_ptr = x_ptr + row * stride_m
        out_row_ptr = out_ptr + row * stride_m
        
        for col_start in range(0, N, BLOCK_N):
            col_offsets = col_start + tl.arange(0, BLOCK_N)
            mask = col_offsets < N
            
            x = tl.load(row_ptr + col_offsets * stride_n, mask=mask, other=0.0)
            out = compute_function(x)
            
            tl.store(out_row_ptr + col_offsets * stride_n, out, mask=mask)


def large_shape_op(x: torch.Tensor) -> torch.Tensor:
    M, N = x.shape
    
    try:
        num_cores = torch_npu.npu.npu_config.get_device_limit(0).get("vector_core_num", 40)
    except:
        num_cores = 40
    
    out = torch.empty_like(x)
    BLOCK_N = 256
    
    grid = (num_cores,)
    large_shape_kernel[grid](x, out, M, N, x.stride(0), x.stride(1), BLOCK_N, num_cores)
    
    return out
```

### 2. 多级 Tiling

```python
@triton.jit
def multi_level_tiling_kernel(
    x_ptr,
    out_ptr,
    M,
    N,
    stride_m,
    stride_n,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    TILE_M: tl.constexpr,
    TILE_N: tl.constexpr,
):
    """
    多级 Tiling：
    - 第一级：BLOCK_M x BLOCK_N (与 core 对应)
    - 第二级：TILE_M x TILE_N (UB 容量内)
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    # 计算当前块的起始位置
    block_start_m = pid_m * BLOCK_M
    block_start_n = pid_n * BLOCK_N
    
    # 在块内进行第二级 Tiling
    for tile_m in range(0, BLOCK_M, TILE_M):
        for tile_n in range(0, BLOCK_N, TILE_N):
            m_idx = block_start_m + tile_m + tl.arange(0, TILE_M)
            n_idx = block_start_n + tile_n + tl.arange(0, TILE_N)
            
            mask = (m_idx[:, None] < M) & (n_idx[None, :] < N)
            
            x = tl.load(x_ptr + m_idx[:, None] * stride_m + n_idx[None, :] * stride_n, mask=mask, other=0.0)
            out = compute_function(x)
            
            tl.store(out_ptr + m_idx[:, None] * stride_m + n_idx[None, :] * stride_n, out, mask=mask)
```

### 3. 减少 UB 占用的技巧

```python
# ❌ 错误：UB 占用过大
@triton.jit
def bad_kernel(x_ptr, out_ptr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    
    # 同时加载多个大 tensor
    x1 = tl.load(x_ptr + offsets)
    x2 = tl.load(x_ptr + offsets + 100000)  # 另一个大 tensor
    x3 = tl.load(x_ptr + offsets + 200000)  # 又一个大 tensor
    
    out = compute_with_all(x1, x2, x3)  # UB 溢出！
    tl.store(out_ptr + offsets, out)


# ✅ 正确：分次加载，减少 UB 占用
@triton.jit
def good_kernel(x_ptr, out_ptr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    
    # 分批处理
    x1 = tl.load(x_ptr + offsets)
    out1 = compute_phase1(x1)
    
    x2 = tl.load(x_ptr + offsets + 100000)
    out2 = compute_phase2(x2)
    
    x3 = tl.load(x_ptr + offsets + 200000)
    out3 = compute_phase3(x3)
    
    out = combine(out1, out2, out3)
    tl.store(out_ptr + offsets, out)
```

### 4. 使用 Double Buffering

```python
@triton.jit
def double_buffer_kernel(
    x_ptr,
    out_ptr,
    N,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Double Buffering：计算当前块时预加载下一块
    """
    pid = tl.program_id(0)
    
    # 初始化：第一块
    offset = pid * BLOCK_SIZE
    buf = tl.load(x_ptr + offset, mask=offset < N, other=0.0)
    
    result = compute_phase1(buf)
    
    # Double Buffering 循环
    for i in range(1, 2):  # 根据需要调整
        next_offset = offset + BLOCK_SIZE
        
        # 预加载下一块（与计算并行）
        next_buf = tl.load(x_ptr + next_offset, mask=next_offset < N, other=0.0)
        
        # 存储当前结果
        tl.store(out_ptr + offset, result, mask=offset < N)
        
        # 更新
        offset = next_offset
        buf = next_buf
        result = compute_phase1(buf)
    
    # 存储最后一块
    tl.store(out_ptr + offset, result, mask=offset < N)
```

## 使用示例

```python
import torch

# 大 shape tensor (例如 M=100000, N=100000)
M, N = 100000, 100000
x = torch.randn(M, N, device='npu', dtype=torch.float16)

out = large_shape_op(x)
print(f"Output shape: {out.shape}")
print("✅ Large Shape 处理完成")
```

## 常见错误

| 错误 | 原因 | 解决方案 |
|------|------|---------|
| UB 溢出 | BLOCK_SIZE 过大或中间变量过多 | 使用多级 Tiling 或固定核心数 |
| Grid 超限 | grid 总大小超过 65535 | 使用固定核心数 + 交错循环 |
| 性能差 | 未充分利用核心 | 使用固定核心数策略 |
