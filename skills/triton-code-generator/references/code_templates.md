# 代码模板

本文档提供常用Triton算子的代码模板，可作为生成代码的参考。

---

## 目录

1. [基础模板](#1-基础模板)
   - [向量加法](#11-向量加法)
   - [向量乘法](#12-向量乘法)
2. [规约模板](#2-规约模板)
   - [向量求和](#21-向量求和)
   - [向量最大值](#22-向量最大值)
3. [Softmax模板](#3-softmax模板)
   - [标准Softmax](#31-标准softmax)
   - [融合Softmax](#32-融合softmax)
4. [LayerNorm模板](#4-layernorm模板)
5. [矩阵乘法模板](#5-矩阵乘法模板)
6. [Flash Attention模板](#6-flash-attention模板)
7. [测试模板](#7-测试模板)

---

## 1. 基础模板

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

### 1.2 向量乘法

```python
@triton.jit
def mul_kernel(
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
    output = x * y
    tl.store(output_ptr + offsets, output, mask=mask)


def mul(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    output = torch.empty_like(x)
    n_elements = output.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    mul_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=1024)
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

### 2.2 向量最大值

```python
@triton.jit
def max_kernel(
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
    block_max = tl.max(x, axis=0)
    tl.atomic_max(output_ptr, block_max)


def max_reduce(x: torch.Tensor) -> torch.Tensor:
    output = torch.full((1,), float('-inf'), device=x.device, dtype=x.dtype)
    n_elements = x.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    max_kernel[grid](x, output, n_elements, BLOCK_SIZE=1024)
    return output
```

---

## 3. Softmax模板

### 3.1 标准Softmax

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

### 3.2 融合Softmax

```python
@triton.jit
def fused_softmax_kernel(
    input_ptr,
    output_ptr,
    M, N,
    stride_m,
    scale: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    row_idx = tl.program_id(0)
    row_start = row_idx * stride_m
    
    max_val = float("-inf")
    for block_start in range(0, N, BLOCK_N):
        offsets = block_start + tl.arange(0, BLOCK_N)
        mask = offsets < N
        x = tl.load(input_ptr + row_start + offsets, mask=mask)
        x = x * scale  # 融合缩放
        max_val = tl.maximum(max_val, tl.max(x, axis=0))
    
    sum_exp = 0.0
    for block_start in range(0, N, BLOCK_N):
        offsets = block_start + tl.arange(0, BLOCK_N)
        mask = offsets < N
        x = tl.load(input_ptr + row_start + offsets, mask=mask)
        x = x * scale
        exp_x = tl.exp(x - max_val)
        sum_exp += tl.sum(exp_x, axis=0)
    
    for block_start in range(0, N, BLOCK_N):
        offsets = block_start + tl.arange(0, BLOCK_N)
        mask = offsets < N
        x = tl.load(input_ptr + row_start + offsets, mask=mask)
        x = x * scale
        exp_x = tl.exp(x - max_val)
        out = exp_x / sum_exp
        tl.store(output_ptr + row_start + offsets, out, mask=mask)


def fused_softmax(x: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
    M, N = x.shape
    output = torch.empty_like(x)
    grid = (M,)
    fused_softmax_kernel[grid](x, output, M, N, x.stride(0), scale=scale, BLOCK_N=1024)
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

## 5. 矩阵乘法模板

```python
@triton.jit
def matmul_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
    
    for k in range(0, K, BLOCK_K):
        # 加载A块 [BLOCK_M, BLOCK_K]
        a_offsets_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        a_offsets_k = k + tl.arange(0, BLOCK_K)
        a_mask_m = a_offsets_m < M
        a_mask_k = a_offsets_k < K
        a = tl.load(
            a_ptr + a_offsets_m[:, None] * stride_am + a_offsets_k[None, :] * stride_ak,
            mask=a_mask_m[:, None] & a_mask_k[None, :]
        )
        
        # 加载B块 [BLOCK_K, BLOCK_N]
        b_offsets_k = k + tl.arange(0, BLOCK_K)
        b_offsets_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        b_mask_k = b_offsets_k < K
        b_mask_n = b_offsets_n < N
        b = tl.load(
            b_ptr + b_offsets_k[:, None] * stride_bk + b_offsets_n[None, :] * stride_bn,
            mask=b_mask_k[:, None] & b_mask_n[None, :]
        )
        
        # 矩阵乘法
        acc += tl.dot(a, b)
    
    # 存储结果
    c_offsets_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    c_offsets_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    c_mask_m = c_offsets_m < M
    c_mask_n = c_offsets_n < N
    tl.store(
        c_ptr + c_offsets_m[:, None] * stride_cm + c_offsets_n[None, :] * stride_cn,
        acc,
        mask=c_mask_m[:, None] & c_mask_n[None, :]
    )


def matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    M, K = a.shape
    K, N = b.shape
    c = torch.empty((M, N), device=a.device, dtype=a.dtype)
    
    grid = lambda meta: (
        triton.cdiv(M, meta['BLOCK_M']),
        triton.cdiv(N, meta['BLOCK_N']),
    )
    
    matmul_kernel[grid](
        a, b, c,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        BLOCK_M=64,
        BLOCK_N=64,
        BLOCK_K=32,
    )
    
    return c
```

---

## 6. Flash Attention模板

```python
@triton.jit
def flash_attention_kernel(
    q_ptr, k_ptr, v_ptr, o_ptr,
    B, H, S, D,
    stride_qb, stride_qh, stride_qs, stride_qd,
    stride_kb, stride_kh, stride_ks, stride_kd,
    stride_vb, stride_vh, stride_vs, stride_vd,
    stride_ob, stride_oh, stride_os, stride_od,
    scale: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    batch = tl.program_id(0)
    head = tl.program_id(1)
    start_m = tl.program_id(2)
    
    # 初始化
    m_i = float("-inf")
    l_i = 0.0
    acc = tl.zeros([BLOCK_M, D], dtype=tl.float32)
    
    # Q块起始位置
    q_start = batch * stride_qb + head * stride_qh + start_m * BLOCK_M * stride_qs
    
    # 遍历K, V块
    for start_n in range(0, S, BLOCK_N):
        # 加载Q块 [BLOCK_M, D]
        q_offsets_s = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
        q_offsets_d = tl.arange(0, D)
        q_mask_s = q_offsets_s < S
        q = tl.load(
            q_ptr + q_offsets_s[:, None] * stride_qs + q_offsets_d[None, :] * stride_qd,
            mask=q_mask_s[:, None]
        )
        
        # 加载K块 [BLOCK_N, D]
        k_offsets_s = start_n + tl.arange(0, BLOCK_N)
        k_offsets_d = tl.arange(0, D)
        k_mask_s = k_offsets_s < S
        k = tl.load(
            k_ptr + batch * stride_kb + head * stride_kh + 
            k_offsets_s[:, None] * stride_ks + k_offsets_d[None, :] * stride_kd,
            mask=k_mask_s[:, None]
        )
        
        # 计算注意力分数
        qk = tl.dot(q, tl.trans(k)) * scale  # [BLOCK_M, BLOCK_N]
        
        # 加载V块 [BLOCK_N, D]
        v_offsets_s = start_n + tl.arange(0, BLOCK_N)
        v_offsets_d = tl.arange(0, D)
        v_mask_s = v_offsets_s < S
        v = tl.load(
            v_ptr + batch * stride_vb + head * stride_vh +
            v_offsets_s[:, None] * stride_vs + v_offsets_d[None, :] * stride_vd,
            mask=v_mask_s[:, None]
        )
        
        # 在线Softmax
        m_new = tl.maximum(m_i, tl.max(qk, axis=1))
        p = tl.exp(qk - m_new[:, None])
        l_new = l_i * tl.exp(m_i - m_new) + tl.sum(p, axis=1)
        
        # 更新累加器
        acc = acc * (l_i * tl.exp(m_i - m_new))[:, None] / l_new[:, None]
        acc += tl.dot(p / l_new[:, None], v)
        
        m_i = m_new
        l_i = l_new
    
    # 存储结果
    o_offsets_s = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    o_offsets_d = tl.arange(0, D)
    o_mask_s = o_offsets_s < S
    tl.store(
        o_ptr + batch * stride_ob + head * stride_oh +
        o_offsets_s[:, None] * stride_os + o_offsets_d[None, :] * stride_od,
        acc,
        mask=o_mask_s[:, None]
    )


def flash_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale: float = 1.0,
) -> torch.Tensor:
    B, H, S, D = q.shape
    o = torch.empty_like(q)
    
    grid = (B, H, triton.cdiv(S, 64))
    
    flash_attention_kernel[grid](
        q, k, v, o,
        B, H, S, D,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        o.stride(0), o.stride(1), o.stride(2), o.stride(3),
        scale=scale,
        BLOCK_M=64,
        BLOCK_N=64,
    )
    
    return o
```

---

## 7. 测试模板

```python
import torch
import pytest

def test_correctness():
    """测试正确性"""
    torch.manual_seed(42)
    
    # 准备测试数据
    x = torch.randn(128, 1024, device='npu', dtype=torch.float16)
    
    # 调用算子
    output = my_operator(x)
    
    # 参考实现
    expected = torch_reference(x)
    
    # 验证
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
