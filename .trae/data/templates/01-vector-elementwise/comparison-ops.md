# Comparison Operations

比较操作是逐元素操作的一种，但 Ascend NPU 有特殊限制。

## ⚠️ 重要注意事项

**Ascend NPU 不支持 int32/int64 的比较操作**，会导致回退到标量计算，严重影响性能。

```python
# ❌ 错误：int64 比较会回退到标量
cols = tl.arange(0, BLOCK_N)  # 默认 int64
mask = cols < N  # 性能极差！

# ✅ 正确：转换为 float32 后比较
cols = tl.arange(0, BLOCK_N)
cols_f32 = cols.to(tl.float32)
mask = cols_f32 < N  # 向量化比较
```

---

## 算子描述

**功能**：逐元素比较两个张量

**常见类型**：
- Greater Than (`>`)
- Less Than (`<`)
- Equal (`==`)
- Not Equal (`!=`)
- Greater or Equal (`>=`)
- Less or Equal (`<=`)

## 硬件约束

| 约束 | 值 | 说明 |
|------|-----|------|
| UB 占用 | ≤ 85KB/循环 | 单次循环内 UB 占用需小于此值 |
| BLOCK_SIZE | 推荐 512-1024 | 太大可能导致 UB 溢出 |

---

## Greater Than

### 核心代码

```python
import triton
import triton.language as tl


@triton.jit
def greater_than_kernel(
    x_ptr,
    y_ptr,
    out_ptr,
    n_elements: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    逐元素比较：x > y
    
    返回布尔张量
    """
    pid = tl.program_id(axis=0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset < n_elements
    
    x = tl.load(x_ptr + offset, mask=mask, other=0.0)
    y = tl.load(y_ptr + offset, mask=mask, other=0.0)
    
    out = x > y
    
    tl.store(out_ptr + offset, out, mask=mask)


def greater_than(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    assert x.shape == y.shape, "输入 shape 必须相同"
    
    n_elements = x.numel()
    BLOCK_SIZE = 1024
    out = torch.empty_like(x, dtype=torch.bool)
    
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    greater_than_kernel[grid](x, y, out, n_elements, BLOCK_SIZE)
    
    return out
```

---

## Less Than

### 核心代码

```python
@triton.jit
def less_than_kernel(
    x_ptr,
    y_ptr,
    out_ptr,
    n_elements: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    逐元素比较：x < y
    """
    pid = tl.program_id(axis=0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset < n_elements
    
    x = tl.load(x_ptr + offset, mask=mask, other=0.0)
    y = tl.load(y_ptr + offset, mask=mask, other=0.0)
    
    out = x < y
    
    tl.store(out_ptr + offset, out, mask=mask)


def less_than(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    assert x.shape == y.shape
    
    n_elements = x.numel()
    BLOCK_SIZE = 1024
    out = torch.empty_like(x, dtype=torch.bool)
    
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    less_than_kernel[grid](x, y, out, n_elements, BLOCK_SIZE)
    
    return out
```

---

## Equal

### 核心代码

```python
@triton.jit
def equal_kernel(
    x_ptr,
    y_ptr,
    out_ptr,
    n_elements: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    逐元素比较：x == y
    """
    pid = tl.program_id(axis=0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset < n_elements
    
    x = tl.load(x_ptr + offset, mask=mask, other=0.0)
    y = tl.load(y_ptr + offset, mask=mask, other=0.0)
    
    out = x == y
    
    tl.store(out_ptr + offset, out, mask=mask)


def equal(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    assert x.shape == y.shape
    
    n_elements = x.numel()
    BLOCK_SIZE = 1024
    out = torch.empty_like(x, dtype=torch.bool)
    
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    equal_kernel[grid](x, y, out, n_elements, BLOCK_SIZE)
    
    return out
```

---

## 与标量比较

### 核心代码

```python
@triton.jit
def greater_than_scalar_kernel(
    x_ptr,
    out_ptr,
    scalar,
    n_elements: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    逐元素比较：x > scalar
    """
    pid = tl.program_id(axis=0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset < n_elements
    
    x = tl.load(x_ptr + offset, mask=mask, other=0.0)
    
    out = x > scalar
    
    tl.store(out_ptr + offset, out, mask=mask)


def greater_than_scalar(x: torch.Tensor, scalar: float) -> torch.Tensor:
    n_elements = x.numel()
    BLOCK_SIZE = 1024
    out = torch.empty_like(x, dtype=torch.bool)
    
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    greater_than_scalar_kernel[grid](x, out, scalar, n_elements, BLOCK_SIZE)
    
    return out
```

---

## 使用示例

```python
import torch

x = torch.randn(1024, device='npu', dtype=torch.float16)
y = torch.randn(1024, device='npu', dtype=torch.float16)

# 张量比较
out_gt = greater_than(x, y)
expected_gt = x > y
assert torch.allclose(out_gt, expected_gt)
print("✅ Greater Than 正确性验证通过")

# 标量比较
out_scalar = greater_than_scalar(x, 0.5)
expected_scalar = x > 0.5
assert torch.allclose(out_scalar, expected_scalar)
print("✅ 标量比较正确性验证通过")
```

---

## 常见错误

| 错误 | 原因 | 解决方案 |
|------|------|---------|
| 性能极差 | 使用 int32/int64 进行比较 | 转换为 float32 后再比较 |
| 结果不一致 | 输入不是连续内存 | 使用 `.contiguous()` 或确保输入连续 |
| UB 溢出 | BLOCK_SIZE 过大 | 减小 BLOCK_SIZE 至 512 或 256 |

---
