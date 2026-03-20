# MatMul with Bias

带偏置的矩阵乘法，在简单 MatMul 基础上增加逐元素加法。

## 算子描述

**功能**：计算 `C = A @ B + bias`

**特征**：
- 在 MatMul 基础上增加 bias 加法
- bias 通常是一维向量 (N,)
- 使用 broadcast 机制

## 硬件约束

与简单 MatMul 相同，额外需要 UB 空间存储 bias。

### L0 约束计算

```
L0A_usage = BLOCK_M * BLOCK_K * sizeof(A.dtype)
L0B_usage = BLOCK_K * BLOCK_N * sizeof(B.dtype)
L0C_usage = BLOCK_M * BLOCK_N * sizeof(C.dtype)
UB_bias = BLOCK_N * sizeof(bias.dtype)

约束：
- L0A ≤ 64KB
- L0B ≤ 64KB
- L0C ≤ 128KB
- UB_bias 通常很小（< 2KB）
```

## 核心代码

### Kernel 实现

```python
import triton
import triton.language as tl


@triton.jit
def matmul_bias_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    bias_ptr,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """
    带 bias 的矩阵乘法 kernel
    
    参数:
        a_ptr: 矩阵 A 指针 (M, K)
        b_ptr: 矩阵 B 指针 (K, N)
        c_ptr: 矩阵 C 指针 (M, N)
        bias_ptr: bias 指针 (N,)
        M, N, K: 矩阵维度
        stride_am, stride_ak: A 的 stride
        stride_bk, stride_bn: B 的 stride
        stride_cm, stride_cn: C 的 stride
        BLOCK_M, BLOCK_N, BLOCK_K: 分块大小
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    
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
    
    # 加载 bias 并应用
    bias = tl.load(bias_ptr + rn, mask=rn < N, other=0.0).to(tl.float32)
    c = (acc + bias[None, :]).to(tl.float16)
    
    tl.store(
        c_ptr + rm[:, None] * stride_cm + rn[None, :] * stride_cn,
        c,
        mask=(rm[:, None] < M) & (rn[None, :] < N)
    )
```

### Wrapper 函数

```python
import torch


def matmul_bias(
    a: torch.Tensor,
    b: torch.Tensor,
    bias: torch.Tensor,
    BLOCK_M: int = 128,
    BLOCK_N: int = 256,
    BLOCK_K: int = 256
) -> torch.Tensor:
    """
    带 bias 的矩阵乘法 wrapper
    
    参数:
        a: 输入矩阵 A (M, K)
        b: 输入矩阵 B (K, N)
        bias: 偏置向量 (N,)
        BLOCK_M, BLOCK_N, BLOCK_K: 分块大小
    
    返回:
        输出矩阵 C (M, N)
    """
    assert a.shape[1] == b.shape[0], "Incompatible dimensions"
    assert bias.shape == (b.shape[1],), "bias must be (N,)"
    
    M, K = a.shape
    K2, N = b.shape
    
    c = torch.empty((M, N), device=a.device, dtype=a.dtype)
    
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    
    matmul_bias_kernel[grid](
        a, b, c, bias,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        BLOCK_M, BLOCK_N, BLOCK_K
    )
    
    return c
```

## 使用示例

```python
import torch

M, K, N = 512, 256, 512
a = torch.randn(M, K, device='npu', dtype=torch.float16)
b = torch.randn(K, N, device='npu', dtype=torch.float16)
bias = torch.randn(N, device='npu', dtype=torch.float16)

c = matmul_bias(a, b, bias)

expected = torch.matmul(a, b) + bias
assert torch.allclose(c, expected, rtol=1e-3, atol=1e-3)
print("✅ MatMul + Bias 正确性验证通过")
```
