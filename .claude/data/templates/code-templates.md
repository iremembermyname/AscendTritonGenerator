# 代码模板

本文档提供常用Triton算子的代码模板。

---

## 1. 向量操作模板

### 1.1 向量加法

```python
import torch
import triton
import triton.language as tl

@triton.jit
def add_kernel(
    x_ptr,
    y_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    output = x + y
    tl.store(output_ptr + offsets, output, mask=mask)


def add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    output = torch.empty_like(x)
    n_elements = output.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    add_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=1024)
    return output
```

---

## 2. 规约模板

### 2.1 向量求和

```python
@triton.jit
def sum_kernel(
    x_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    block_sum = tl.sum(x, axis=0)
    tl.atomic_add(output_ptr, block_sum)


def sum_reduce(x: torch.Tensor) -> torch.Tensor:
    output = torch.zeros(1, device=x.device, dtype=x.dtype)
    n_elements = x.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    sum_kernel[grid](x, output, n_elements, BLOCK_SIZE=1024)
    return output
```

---

## 3. Softmax模板

```python
@triton.jit
def softmax_kernel(
    input_ptr,
    output_ptr,
    M, N,
    stride_m,
    BLOCK_N: tl.constexpr,
):
    row_idx = tl.program_id(0)
    row_start = row_idx * stride_m
    
    # 找最大值
    max_val = float("-inf")
    for block_start in range(0, N, BLOCK_N):
        offsets = block_start + tl.arange(0, BLOCK_N)
        mask = offsets < N
        x = tl.load(input_ptr + row_start + offsets, mask=mask)
        max_val = tl.maximum(max_val, tl.max(x, axis=0))
    
    # 计算exp和求和
    sum_exp = 0.0
    for block_start in range(0, N, BLOCK_N):
        offsets = block_start + tl.arange(0, BLOCK_N)
        mask = offsets < N
        x = tl.load(input_ptr + row_start + offsets, mask=mask)
        exp_x = tl.exp(x - max_val)
        sum_exp += tl.sum(exp_x, axis=0)
    
    # 归一化
    for block_start in range(0, N, BLOCK_N):
        offsets = block_start + tl.arange(0, BLOCK_N)
        mask = offsets < N
        x = tl.load(input_ptr + row_start + offsets, mask=mask)
        exp_x = tl.exp(x - max_val)
        out = exp_x / sum_exp
        tl.store(output_ptr + row_start + offsets, out, mask=mask)


def softmax(x: torch.Tensor) -> torch.Tensor:
    M, N = x.shape
    output = torch.empty_like(x)
    grid = (M,)
    softmax_kernel[grid](x, output, M, N, x.stride(0), BLOCK_N=1024)
    return output
```

---

## 4. LayerNorm模板

```python
@triton.jit
def layernorm_kernel(
    x_ptr,
    y_ptr,
    weight_ptr,
    bias_ptr,
    M, N,
    stride_m,
    eps: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    row_idx = tl.program_id(0)
    row_start = row_idx * stride_m
    
    # 计算均值
    sum_x = 0.0
    for block_start in range(0, N, BLOCK_N):
        offsets = block_start + tl.arange(0, BLOCK_N)
        mask = offsets < N
        x = tl.load(x_ptr + row_start + offsets, mask=mask)
        sum_x += tl.sum(x, axis=0)
    mean = sum_x / N
    
    # 计算方差
    sum_var = 0.0
    for block_start in range(0, N, BLOCK_N):
        offsets = block_start + tl.arange(0, BLOCK_N)
        mask = offsets < N
        x = tl.load(x_ptr + row_start + offsets, mask=mask)
        diff = x - mean
        sum_var += tl.sum(diff * diff, axis=0)
    var = sum_var / N
    rstd = tl.rsqrt(var + eps)
    
    # 归一化
    for block_start in range(0, N, BLOCK_N):
        offsets = block_start + tl.arange(0, BLOCK_N)
        mask = offsets < N
        x = tl.load(x_ptr + row_start + offsets, mask=mask)
        w = tl.load(weight_ptr + offsets, mask=mask)
        b = tl.load(bias_ptr + offsets, mask=mask)
        y = (x - mean) * rstd * w + b
        tl.store(y_ptr + row_start + offsets, y, mask=mask)


def layernorm(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    eps: float = 1e-5,
) -> torch.Tensor:
    M, N = x.shape
    y = torch.empty_like(x)
    grid = (M,)
    layernorm_kernel[grid](x, y, weight, bias, M, N, x.stride(0), eps=eps, BLOCK_N=1024)
    return y
```

---

## 5. MatMul模板（固定核心数启动）

```python
import torch_npu

@triton.jit
def matmul_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    num_cores: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(0)
    NUM_BLOCKS_M = triton.cdiv(M, BLOCK_M)
    NUM_BLOCKS_N = triton.cdiv(N, BLOCK_N)
    NUM_BLOCKS = NUM_BLOCKS_M * NUM_BLOCKS_N

    # 每个核心循环处理多个块
    for block_idx in range(pid, NUM_BLOCKS, num_cores):
        block_m = block_idx // NUM_BLOCKS_N
        block_n = block_idx % NUM_BLOCKS_N

        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

        for k in range(0, K, BLOCK_K):
            a = tl.load(a_ptr + ...)
            b = tl.load(b_ptr + ...)
            acc += tl.dot(a, b)

        tl.store(c_ptr + ..., acc)


class MatMulModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        try:
            self.CUBE_CORE_NUM = torch_npu.npu.npu_config.get_device_limit(0).get("cube_core_num", 20)
        except:
            self.CUBE_CORE_NUM = 20

    def forward(self, a, b):
        M, K = a.shape
        K2, N = b.shape
        c = torch.empty((M, N), device=a.device, dtype=a.dtype)
        
        grid = (self.CUBE_CORE_NUM,)
        matmul_kernel[grid](a, b, c, M, N, K, self.CUBE_CORE_NUM, BLOCK_M=128, BLOCK_N=256, BLOCK_K=256)
        return c
```

**关键点**：
- 使用 `grid=(num_cores,)` 固定启动核心数
- 每个核心通过 `for block_idx in range(pid, NUM_BLOCKS, num_cores)` 循环处理多个块

---

## 6. autotune使用示例

```python
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128}),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 64}),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def matmul_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    pass

# 调用时不要传递configs中的参数
grid = lambda meta: (triton.cdiv(M, meta['BLOCK_SIZE_M']) * triton.cdiv(N, meta['BLOCK_SIZE_N']),)
matmul_kernel[grid](a, b, c, M, N, K, a.stride(0), a.stride(1), b.stride(0), b.stride(1), c.stride(0), c.stride(1))
```

---

## 7. 大shape算子Grid处理（交错循环）

```python
import torch_npu

@triton.jit
def row_processing_kernel(
    input_ptr, output_ptr, 
    M, N,
    stride_m, stride_n,
    BLOCK_N: tl.constexpr,
    CORE_NUM: tl.constexpr,
):
    pid = tl.program_id(0)
    
    # 交错处理：每个核心处理 pid, pid+CORE_NUM, pid+2*CORE_NUM, ... 行
    for row_idx in range(pid, M, CORE_NUM):
        row_ptr = input_ptr + row_idx * stride_m
        out_row_ptr = output_ptr + row_idx * stride_m
        
        for col_start in range(0, N, BLOCK_N):
            col_offsets = col_start + tl.arange(0, BLOCK_N)
            mask = col_offsets < N
            
            data = tl.load(row_ptr + col_offsets * stride_n, mask=mask)
            result = compute_function(data)
            tl.store(out_row_ptr + col_offsets * stride_n, result, mask=mask)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()
        try:
            self.VEC_CORE_NUM = torch_npu.npu.npu_config.get_device_limit(0).get("vector_core_num", 40)
        except:
            self.VEC_CORE_NUM = 40

    def forward(self, input_tensor):
        M, N = input_tensor.shape
        output_tensor = torch.empty_like(input_tensor)
        
        grid = (self.VEC_CORE_NUM,)
        row_processing_kernel[grid](
            input_tensor, output_tensor,
            M, N,
            input_tensor.stride(0), input_tensor.stride(1),
            BLOCK_N=256,
            CORE_NUM=self.VEC_CORE_NUM,
        )
        return output_tensor
```

---

## 8. 测试模板

```python
import torch
import pytest

def test_correctness():
    """测试正确性"""
    torch.manual_seed(42)
    
    x = torch.randn(128, 1024, device='npu', dtype=torch.float16)
    
    output = my_operator(x)
    
    expected = torch_reference(x)
    
    assert torch.allclose(output, expected, rtol=1e-3, atol=1e-3)
    print("Correctness test passed!")


def test_performance():
    """测试性能"""
    import time
    
    x = torch.randn(128, 1024, device='npu', dtype=torch.float16)
    
    # Warmup
    for _ in range(10):
        _ = my_operator(x)
    
    # Benchmark
    torch.npu.synchronize()
    start = time.time()
    for _ in range(100):
        _ = my_operator(x)
    torch.npu.synchronize()
    end = time.time()
    
    avg_time_ms = (end - start) / 100 * 1000
    print(f"Average time: {avg_time_ms:.3f} ms")
    
    return avg_time_ms


if __name__ == "__main__":
    test_correctness()
    test_performance()
```
