# L0 Constraints Calculation

L0 约束计算用于验证 Cube 算子的分块配置是否满足 Ascend NPU 的硬件约束。

## 算子描述

**功能**：验证给定的分块配置是否满足 L0A/L0B/L0C 容量约束

**硬件约束**：
- L0A ≤ 64KB (存储矩阵 A)
- L0B ≤ 64KB (存储矩阵 B)
- L0C ≤ 128KB (存储累加器 C)

## L0 约束计算

### 计算公式

```
L0A_usage = BLOCK_M * BLOCK_K * sizeof(A.dtype)
L0B_usage = BLOCK_K * BLOCK_N * sizeof(B.dtype)
L0C_usage = BLOCK_M * BLOCK_N * sizeof(C.dtype)

约束：
- L0A_usage ≤ 64KB
- L0B_usage ≤ 64KB
- L0C_usage ≤ 128KB
```

### 数据类型大小

| 数据类型 | 字节数 |
|---------|--------|
| float16 | 2 |
| bfloat16 | 2 |
| float32 | 4 |
| int8 | 1 |

## 验证代码

### 验证函数

```python
def validate_block_config(BLOCK_M, BLOCK_N, BLOCK_K, dtype_a, dtype_b, dtype_c):
    """
    验证分块配置是否满足 L0 约束
    
    参数:
        BLOCK_M, BLOCK_N, BLOCK_K: 分块大小
        dtype_a, dtype_b, dtype_c: 数据类型
    
    返回:
        dict: 验证结果和详细信息
    """
    dtype_size = {
        torch.float16: 2,
        torch.bfloat16: 2,
        torch.float32: 4,
        torch.int8: 1,
    }
    
    size_a = dtype_size[dtype_a]
    size_b = dtype_size[dtype_b]
    size_c = dtype_size[dtype_c]
    
    L0A_usage = BLOCK_M * BLOCK_K * size_a
    L0B_usage = BLOCK_K * BLOCK_N * size_b
    L0C_usage = BLOCK_M * BLOCK_N * size_c
    
    L0A_limit = 64 * 1024
    L0B_limit = 64 * 1024
    L0C_limit = 128 * 1024
    
    valid = (
        L0A_usage <= L0A_limit and
        L0B_usage <= L0B_limit and
        L0C_usage <= L0C_limit
    )
    
    return {
        'valid': valid,
        'L0A_usage': L0A_usage,
        'L0B_usage': L0B_usage,
        'L0C_usage': L0C_usage,
        'L0A_limit': L0A_limit,
        'L0B_limit': L0B_limit,
        'L0C_limit': L0C_limit,
    }


# 示例：验证 FP16 配置
result = validate_block_config(
    BLOCK_M=128,
    BLOCK_N=256,
    BLOCK_K=256,
    dtype_a=torch.float16,
    dtype_b=torch.float16,
    dtype_c=torch.float16
)

print(f"Valid: {result['valid']}")
print(f"L0A: {result['L0A_usage'] / 1024:.2f} KB / {result['L0A_limit'] / 1024:.2f} KB")
print(f"L0B: {result['L0B_usage'] / 1024:.2f} KB / {result['L0B_limit'] / 1024:.2f} KB")
print(f"L0C: {result['L0C_usage'] / 1024:.2f} KB / {result['L0C_limit'] / 1024:.2f} KB")
```

### 打印格式化版本

```python
def print_l0_constraints(BLOCK_M, BLOCK_N, BLOCK_K, dtype):
    """
    打印 L0 约束验证结果
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
    
    print(f"Configuration: BLOCK_M={BLOCK_M}, BLOCK_K={BLOCK_K}, BLOCK_N={BLOCK_N}")
    print(f"Dtype: {dtype} ({size} bytes)")
    print()
    print(f"L0A: {L0A_usage / 1024:6.2f} KB / {L0A_limit / 1024:6.2f} KB {'✅' if L0A_usage <= L0A_limit else '❌'}")
    print(f"L0B: {L0B_usage / 1024:6.2f} KB / {L0B_limit / 1024:6.2f} KB {'✅' if L0B_usage <= L0B_limit else '❌'}")
    print(f"L0C: {L0C_usage / 1024:6.2f} KB / {L0C_limit / 1024:6.2f} KB {'✅' if L0C_usage <= L0C_limit else '❌'}")
    print()
    
    valid = (
        L0A_usage <= L0A_limit and
        L0B_usage <= L0B_limit and
        L0C_usage <= L0C_limit
    )
    
    if valid:
        print("✅ Configuration is valid")
    else:
        print("❌ Configuration exceeds L0 constraints")
    
    return valid
```

## 推荐配置

### FP16/BF16 配置

| 配置 | BLOCK_M | BLOCK_K | BLOCK_N | L0A | L0B | L0C | 状态 |
|------|---------|---------|---------|-----|-----|-----|------|
| 推荐 1 | 128 | 256 | 256 | 64KB | 64KB | 64KB | ✅ |
| 推荐 2 | 64 | 256 | 512 | 32KB | 64KB | 64KB | ✅ |
| 推荐 3 | 256 | 128 | 256 | 64KB | 32KB | 64KB | ✅ |

### FP32 配置

| 配置 | BLOCK_M | BLOCK_K | BLOCK_N | L0A | L0B | L0C | 状态 |
|------|---------|---------|---------|-----|-----|-----|------|
| 推荐 1 | 64 | 128 | 128 | 32KB | 32KB | 32KB | ✅ |
| 推荐 2 | 32 | 128 | 256 | 16KB | 32KB | 32KB | ✅ |

## 使用示例

```python
import torch

# 验证推荐配置
print("=== FP16 Configurations ===")
print_l0_constraints(128, 256, 256, torch.float16)
print()
print_l0_constraints(64, 256, 512, torch.float16)
print()
print_l0_constraints(256, 128, 256, torch.float16)

print("\n=== FP32 Configurations ===")
print_l0_constraints(64, 128, 128, torch.float32)
print()
print_l0_constraints(32, 128, 256, torch.float32)
```
