# Simple MatMul

矩阵乘法是深度学习中最核心的计算密集型操作，使用 `tl.dot` 进行计算。

## 算子描述

**功能**：计算矩阵乘法 `C = A @ B`

**特征**：
- 使用 `tl.dot` 进行矩阵乘法计算
- 使用 L0A/L0B/L0C 存储矩阵块
- 使用 float32 累加器保证精度

## 硬件约束

| 约束 | 限制 | 说明 |
|------|------|------|
| L0A 容量 | ≤ 64KB | 左矩阵 A (m0×k0) |
| L0B 容量 | ≤ 64KB | 右矩阵 B (k0×n0) |
| L0C 容量 | ≤ 128KB | 结果矩阵 C (m0×n0)，支持累加 |
| Cube Core | 20-24 | 获取方式：`torch_npu.npu.npu_config.get_device_limit(0).get("cube_core_num", 20)` |

### L0 约束计算

```
L0A_usage = BLOCK_M * BLOCK_K * sizeof(A.dtype)
L0B_usage = BLOCK_K * BLOCK_N * sizeof(B.dtype)
L0C_usage = BLOCK_M * BLOCK_N * sizeof(C.dtype)

约束：
- L0A ≤ 64KB
- L0B ≤ 64KB
- L0C ≤ 128KB
```

### 推荐分块配置（FP16/BF16）

| 配置 | BLOCK_M | BLOCK_K | BLOCK_N | L0A | L0B | L0C |
|------|---------|---------|---------|-----|-----|-----|
| 推荐 | 128 | 256 | 256 | 64KB | 64KB | 64KB |

## 核心代码

### Kernel 实现

```python
import triton
import triton.language as tl


@triton.jit
def matmul_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
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
    矩阵乘法 kernel
    
    参数:
        a_ptr: 矩阵 A 指针 (M, K)
        b_ptr: 矩阵 B 指针 (K, N)
        c_ptr: 矩阵 C 指针 (M, N)
        M, N, K: 矩阵维度
        stride_am, stride_ak: A 的 stride
        stride_bk, stride_bn: B 的 stride
        stride_cm, stride_cn: C 的 stride
        BLOCK_M, BLOCK_N, BLOCK_K: 分块大小
    """
    # 计算 program ID 对应的输出块
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    # 计算当前块在 M 和 N 维度的偏移
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    
    # 初始化累加器（使用 float32）
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    # K 维度循环
    for k in range(0, K, BLOCK_K):
        rk = k + tl.arange(0, BLOCK_K)
        
        # 加载 A 的块
        a = tl.load(
            a_ptr + rm[:, None] * stride_am + rk[None, :] * stride_ak,
            mask=(rm[:, None] < M) & (rk[None, :] < K),
            other=0.0
        )
        
        # 加载 B 的块
        b = tl.load(
            b_ptr + rk[:, None] * stride_bk + rn[None, :] * stride_bn,
            mask=(rk[:, None] < K) & (rn[None, :] < N),
            other=0.0
        )
        
        # 矩阵乘法累加
        acc += tl.dot(a, b)
    
    # 转换回输出类型并存储
    c = acc.to(tl.float16)
    tl.store(
        c_ptr + rm[:, None] * stride_cm + rn[None, :] * stride_cn,
        c,
        mask=(rm[:, None] < M) & (rn[None, :] < N)
    )
```

### Wrapper 函数

```python
import torch


def matmul(
    a: torch.Tensor,
    b: torch.Tensor,
    BLOCK_M: int = 128,
    BLOCK_N: int = 256,
    BLOCK_K: int = 256
) -> torch.Tensor:
    """
    矩阵乘法 wrapper
    
    参数:
        a: 输入矩阵 A (M, K)
        b: 输入矩阵 B (K, N)
        BLOCK_M, BLOCK_N, BLOCK_K: 分块大小
    
    返回:
        输出矩阵 C (M, N)
    """
    # 形状检查
    assert a.shape[1] == b.shape[0], "Incompatible dimensions"
    assert a.is_contiguous(), "Matrix A must be contiguous"
    assert b.is_contiguous(), "Matrix B must be contiguous"
    
    M, K = a.shape
    K2, N = b.shape
    
    # 分配输出
    c = torch.empty((M, N), device=a.device, dtype=a.dtype)
    
    # 2D grid
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    
    matmul_kernel[grid](
        a, b, c,
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

c = matmul(a, b)

expected = torch.matmul(a, b)
assert torch.allclose(c, expected, rtol=1e-3, atol=1e-3)
print("✅ MatMul 正确性验证通过")
```

## L0 约束验证

```python
def validate_block_config(BLOCK_M, BLOCK_N, BLOCK_K, dtype):
    """
    验证分块配置是否满足 L0 约束
    """
    dtype_size = {
        torch.float16: 2,
        torch.bfloat16: 2,
        torch.float32: 4,
        torch.int8: 1,
    }
    
    size = dtype_size[dtype]
    
    L0A_usage = BLOCK_M * BLOCK_K * size
    L0B_usage = BLOCK_K * BLOCK_N * size
    L0C_usage = BLOCK_M * BLOCK_N * size
    
    L0A_limit = 64 * 1024
    L0B_limit = 64 * 1024
    L0C_limit = 128 * 1024
    
    valid = (
        L0A_usage <= L0A_limit and
        L0B_usage <= L0B_limit and
        L0C_usage <= L0C_limit
    )
    
    print(f"BLOCK_M={BLOCK_M}, BLOCK_K={BLOCK_K}, BLOCK_N={BLOCK_N}")
    print(f"L0A: {L0A_usage / 1024:.2f}KB / {L0A_limit / 1024:.2f}KB {'✅' if L0A_usage <= L0A_limit else '❌'}")
    print(f"L0B: {L0B_usage / 1024:.2f}KB / {L0B_limit / 1024:.2f}KB {'✅' if L0B_usage <= L0B_limit else '❌'}")
    print(f"L0C: {L0C_usage / 1024:.2f}KB / {L0C_limit / 1024:.2f}KB {'✅' if L0C_usage <= L0C_limit else '❌'}")
    
    return valid


# 验证推荐配置
validate_block_config(128, 256, 256, torch.float16)
```

## Autotune 版本

```python
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 64}),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 256, 'BLOCK_K': 32}),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32}),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def matmul_autotune_kernel(
    a_ptr, b_ptr, c_ptr,
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
    
    c = acc.to(tl.float16)
    tl.store(
        c_ptr + rm[:, None] * stride_cm + rn[None, :] * stride_cn,
        c,
        mask=(rm[:, None] < M) & (rn[None, :] < N)
    )


def matmul_autotune(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    M, K = a.shape
    K2, N = b.shape
    
    c = torch.empty((M, N), device=a.device, dtype=a.dtype)
    
    grid = lambda META: (triton.cdiv(M, META['BLOCK_M']) * triton.cdiv(N, META['BLOCK_N']),)
    
    matmul_autotune_kernel[grid](
        a, b, c,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1)
    )
    
    return c
```

## 常见错误

| 错误 | 原因 | 解决方案 |
|------|------|---------|
| L0 溢出 | 分块过大 | 调整 BLOCK_M/N/K 满足 L0 约束 |
| 精度损失 | 累加器类型错误 | 使用 float32 累加器 |
| 结果错误 | 输入不是连续内存 | 使用 `.contiguous()` |
| Grid 超限 | grid 总大小超过 65535 | 使用固定核心数 + 交错循环 |
