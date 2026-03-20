# Batched MatMul

批量矩阵乘法，同时计算多个矩阵对的乘法。

## 算子描述

**功能**：计算批量矩阵乘法 `C[i] = A[i] @ B[i]` for i in batch

**特征**：
- 输入是 3D tensor (batch, M, K), (batch, K, N)
- 输出是 3D tensor (batch, M, N)
- grid 使用 3D：grid = (batch, M_blocks, N_blocks)

## 核心代码

```python
import triton
import triton.language as tl


@triton.jit
def bmm_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    batch,
    M,
    N,
    K,
    stride_ab, stride_am, stride_ak,
    stride_bb, stride_bk, stride_bn,
    stride_cb, stride_cm, stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    # 3D program ID
    pid_b = tl.program_id(0)
    pid_m = tl.program_id(1)
    pid_n = tl.program_id(2)
    
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    
    # 计算当前 batch 的基地址
    a = a_ptr + pid_b * stride_ab
    b = b_ptr + pid_b * stride_bb
    c = c_ptr + pid_b * stride_cb
    
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    for k in range(0, K, BLOCK_K):
        rk = k + tl.arange(0, BLOCK_K)
        
        a_block = tl.load(
            a + rm[:, None] * stride_am + rk[None, :] * stride_ak,
            mask=(rm[:, None] < M) & (rk[None, :] < K),
            other=0.0
        )
        b_block = tl.load(
            b + rk[:, None] * stride_bk + rn[None, :] * stride_bn,
            mask=(rk[:, None] < K) & (rn[None, :] < N),
            other=0.0
        )
        
        acc += tl.dot(a_block, b_block)
    
    c_block = acc.to(tl.float16)
    tl.store(
        c + rm[:, None] * stride_cm + rn[None, :] * stride_cn,
        c_block,
        mask=(rm[:, None] < M) & (rn[:, None] < N)
    )


def bmm(a, b, BLOCK_M=128, BLOCK_N=256, BLOCK_K=256):
    batch, M, K = a.shape
    batch2, K2, N = b.shape
    assert batch == batch2 and K == K2
    
    c = torch.empty((batch, M, N), device=a.device, dtype=a.dtype)
    
    grid = (batch, triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    bmm_kernel[grid](
        a, b, c,
        batch, M, N, K,
        a.stride(0), a.stride(1), a.stride(2),
        b.stride(0), b.stride(1), b.stride(2),
        c.stride(0), c.stride(1), c.stride(2),
        BLOCK_M, BLOCK_N, BLOCK_K
    )
    
    return c
```
