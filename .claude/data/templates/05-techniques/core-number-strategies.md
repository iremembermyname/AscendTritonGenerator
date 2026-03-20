# Core Number Strategies

固定核心数启动是 Ascend NPU 性能优化的关键策略，用于匹配物理核心数和逻辑 kernel 数。

## 为什么需要固定核心数？

### 问题

当 grid 总大小超过 65535 时，或者逻辑 kernel 数与物理核心数不匹配时，会导致：
- 核心调度开销
- 资源利用率低
- 性能下降

### 解决方案

使用固定核心数启动，让每个核心通过循环处理多个任务：

```python
# ❌ 传统方式：grid = (total_tasks,)
# 可能导致核心数过多或过少

# ✅ 推荐方式：grid = (num_cores,)
# 每个核心循环处理 total_tasks / num_cores 个任务
```

## 核心数获取

### 获取 Vector Core 数

```python
import torch_npu


def get_vector_core_num():
    """获取 Vector Core 数量"""
    try:
        num_cores = torch_npu.npu.npu_config.get_device_limit(0).get("vector_core_num", 40)
    except:
        num_cores = 40
    return num_cores


# 示例
VECTOR_CORE_NUM = get_vector_core_num()
print(f"Vector Core 数：{VECTOR_CORE_NUM}")  # 通常 40-48
```

### 获取 Cube Core 数

```python
import torch_npu


def get_cube_core_num():
    """获取 Cube Core 数量"""
    try:
        num_cores = torch_npu.npu.npu_config.get_device_limit(0).get("cube_core_num", 20)
    except:
        num_cores = 20
    return num_cores


# 示例
CUBE_CORE_NUM = get_cube_core_num()
print(f"Cube Core 数：{CUBE_CORE_NUM}")  # 通常 20-24
```

## 核心代码

### Element-wise 算子

```python
import triton
import triton.language as tl
import torch_npu


@triton.jit
def elementwise_fixed_core_kernel(
    x_ptr,
    out_ptr,
    n_elements: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    CORE_NUM: tl.constexpr,
):
    """
    固定核心数的 element-wise kernel
    
    每个核心处理 pid, pid+CORE_NUM, pid+2*CORE_NUM, ... 位置的元素
    """
    pid = tl.program_id(axis=0)
    
    # 交错循环：每个核心处理多个 block
    for offset in range(pid, n_elements, CORE_NUM * BLOCK_SIZE):
        offsets = offset + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        
        x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
        out = compute_function(x)
        
        tl.store(out_ptr + offsets, out, mask=mask)


def elementwise_fixed_core(x: torch.Tensor) -> torch.Tensor:
    n_elements = x.numel()
    
    try:
        num_cores = torch_npu.npu.npu_config.get_device_limit(0).get("vector_core_num", 40)
    except:
        num_cores = 40
    
    out = torch.empty_like(x)
    BLOCK_SIZE = 512
    
    grid = (num_cores,)
    elementwise_fixed_core_kernel[grid](x, out, n_elements, BLOCK_SIZE, num_cores)
    
    return out
```

### 2D Tensor 算子

```python
@triton.jit
def matrix_fixed_core_kernel(
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
    固定核心数的 2D matrix kernel
    
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


def matrix_fixed_core(x: torch.Tensor) -> torch.Tensor:
    M, N = x.shape
    
    try:
        num_cores = torch_npu.npu.npu_config.get_device_limit(0).get("vector_core_num", 40)
    except:
        num_cores = 40
    
    out = torch.empty_like(x)
    BLOCK_N = 256
    
    grid = (num_cores,)
    matrix_fixed_core_kernel[grid](x, out, M, N, x.stride(0), x.stride(1), BLOCK_N, num_cores)
    
    return out
```

### MatMul 算子

```python
@triton.jit
def matmul_fixed_core_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    CORE_NUM: tl.constexpr,
):
    """
    固定核心数的 MatMul kernel
    
    每个核心处理多个 output blocks
    """
    pid = tl.program_id(0)
    
    num_blocks_m = tl.cdiv(M, BLOCK_M)
    num_blocks_n = tl.cdiv(N, BLOCK_N)
    total_blocks = num_blocks_m * num_blocks_n
    
    # 每个核心循环处理多个 blocks
    for block_idx in range(pid, total_blocks, CORE_NUM):
        block_m = block_idx // num_blocks_n
        block_n = block_idx % num_blocks_n
        
        rm = block_m * BLOCK_M + tl.arange(0, BLOCK_M)
        rn = block_n * BLOCK_N + tl.arange(0, BLOCK_N)
        
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        
        for k in range(0, K, BLOCK_K):
            rk = k + tl.arange(0, BLOCK_K)
            
            a = tl.load(
                a_ptr + rm[:, None] * stride_am + rk[None, :] * stride_ak,
                mask=(rm[:, None] < M) & (rk[None, :] < K),
                other=0.0
            )
            b = tl.load(
                b_ptr + rk[:, None] * stride_bk + rn[None, :] * stride_bn,
                mask=(rk[:, None] < K) & (rn[None, :] < N),
                other=0.0
            )
            
            acc += tl.dot(a, b)
        
        c = acc.to(tl.float16)
        tl.store(
            c_ptr + rm[:, None] * stride_cm + rn[None, :] * stride_cn,
            c,
            mask=(rm[:, None] < M) & (rn[None, :] < N)
        )


def matmul_fixed_core(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    M, K = a.shape
    K2, N = b.shape
    
    c = torch.empty((M, N), device=a.device, dtype=a.dtype)
    
    try:
        num_cores = torch_npu.npu.npu_config.get_device_limit(0).get("cube_core_num", 20)
    except:
        num_cores = 20
    
    BLOCK_M, BLOCK_N, BLOCK_K = 128, 256, 256
    
    grid = (num_cores,)
    matmul_fixed_core_kernel[grid](
        a, b, c,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        BLOCK_M, BLOCK_N, BLOCK_K,
        num_cores
    )
    
    return c
```

## 使用示例

```python
import torch

# Element-wise
x = torch.randn(10000, device='npu', dtype=torch.float16)
out = elementwise_fixed_core(x)

# 2D Matrix
x_2d = torch.randn(1024, 2048, device='npu', dtype=torch.float16)
out_2d = matrix_fixed_core(x_2d)

# MatMul
a = torch.randn(512, 256, device='npu', dtype=torch.float16)
b = torch.randn(256, 512, device='npu', dtype=torch.float16)
c = matmul_fixed_core(a, b)

print("✅ 固定核心数策略完成")
```

## 常见错误

| 错误 | 原因 | 解决方案 |
|------|------|---------|
| 性能差 | grid 过大或过小 | 使用固定核心数 |
| 核心数获取失败 | 未处理异常 | 使用 try-except 和默认值 |
| 循环错误 | 交错步长计算错误 | 使用 `range(pid, total, CORE_NUM * BLOCK_SIZE)` |
