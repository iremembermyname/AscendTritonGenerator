# Activation Functions

激活函数是深度学习中的基础组件，属于逐元素操作（element-wise）。

## 算子描述

**功能**：对输入张量应用非线性激活函数

**常见类型**：
- GELU (Gaussian Error Linear Unit)
- SiLU/Swish (Sigmoid Linear Unit)
- ReLU (Rectified Linear Unit)
- Sigmoid

## 硬件约束

| 约束 | 值 | 说明 |
|------|-----|------|
| UB 占用 | ≤ 85KB/循环 | 单次循环内 UB 占用需小于此值 |
| BLOCK_SIZE | 推荐 512-1024 | 太大可能导致 UB 溢出 |

### UB 占用计算

```
UB_usage = (load_input + store_output) * element_size + intermediate
         = 2 * BLOCK_SIZE * 2 bytes (FP16) + intermediate
         = 4KB + intermediate (远小于 85KB 限制)
```

---

## GELU

### 核心代码

```python
import triton
import triton.language as tl


@triton.jit
def gelu_kernel(
    x_ptr,
    y_ptr,
    n_elements: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    GELU 激活函数
    
    GELU(x) = 0.5 * x * (1 + erf(x / sqrt(2)))
            = 0.5 * x * (1 + erf(x / 1.41421356237))
    """
    pid = tl.program_id(axis=0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset < n_elements
    
    x = tl.load(x_ptr + offset, mask=mask, other=0.0).to(tl.float32)
    
    # GELU 计算
    gelu_val = 0.5 * x * (1.0 + tl.erf(x / 1.41421356237))
    
    tl.store(y_ptr + offset, gelu_val.to(y_ptr.dtype.element_ty), mask=mask)


def gelu(x: torch.Tensor) -> torch.Tensor:
    n_elements = x.numel()
    BLOCK_SIZE = 1024
    y = torch.empty_like(x)
    
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    gelu_kernel[grid](x, y, n_elements, BLOCK_SIZE)
    
    return y
```

### 使用示例

```python
import torch

x = torch.randn(1024, device='npu', dtype=torch.float16)

y = gelu(x)

expected = torch.nn.functional.gelu(x)
assert torch.allclose(y, expected, rtol=1e-3, atol=1e-3)
print("✅ GELU 正确性验证通过")
```

---

## SiLU (Swish)

### 核心代码

```python
@triton.jit
def silu_kernel(
    x_ptr,
    y_ptr,
    n_elements: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    SiLU/Swish 激活函数
    
    SiLU(x) = x * sigmoid(x)
    """
    pid = tl.program_id(axis=0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset < n_elements
    
    x = tl.load(x_ptr + offset, mask=mask, other=0.0).to(tl.float32)
    
    # SiLU 计算
    silu_val = x * tl.sigmoid(x)
    
    tl.store(y_ptr + offset, silu_val.to(y_ptr.dtype.element_ty), mask=mask)


def silu(x: torch.Tensor) -> torch.Tensor:
    n_elements = x.numel()
    BLOCK_SIZE = 1024
    y = torch.empty_like(x)
    
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    silu_kernel[grid](x, y, n_elements, BLOCK_SIZE)
    
    return y
```

### 使用示例

```python
import torch

x = torch.randn(1024, device='npu', dtype=torch.float16)

y = silu(x)

expected = torch.nn.functional.silu(x)
assert torch.allclose(y, expected, rtol=1e-3, atol=1e-3)
print("✅ SiLU 正确性验证通过")
```

---

## ReLU

### 核心代码

```python
@triton.jit
def relu_kernel(
    x_ptr,
    y_ptr,
    n_elements: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    ReLU 激活函数
    
    ReLU(x) = max(0, x)
    """
    pid = tl.program_id(axis=0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset < n_elements
    
    x = tl.load(x_ptr + offset, mask=mask, other=0.0)
    
    # ReLU 计算
    relu_val = tl.maximum(x, 0.0)
    
    tl.store(y_ptr + offset, relu_val, mask=mask)


def relu(x: torch.Tensor) -> torch.Tensor:
    n_elements = x.numel()
    BLOCK_SIZE = 1024
    y = torch.empty_like(x)
    
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    relu_kernel[grid](x, y, n_elements, BLOCK_SIZE)
    
    return y
```

---

## Sigmoid

### 核心代码

```python
@triton.jit
def sigmoid_kernel(
    x_ptr,
    y_ptr,
    n_elements: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Sigmoid 激活函数
    
    Sigmoid(x) = 1 / (1 + exp(-x))
    """
    pid = tl.program_id(axis=0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset < n_elements
    
    x = tl.load(x_ptr + offset, mask=mask, other=0.0).to(tl.float32)
    
    # Sigmoid 计算
    sigmoid_val = 1.0 / (1.0 + tl.exp(-x))
    
    tl.store(y_ptr + offset, sigmoid_val.to(y_ptr.dtype.element_ty), mask=mask)


def sigmoid(x: torch.Tensor) -> torch.Tensor:
    n_elements = x.numel()
    BLOCK_SIZE = 1024
    y = torch.empty_like(x)
    
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    sigmoid_kernel[grid](x, y, n_elements, BLOCK_SIZE)
    
    return y
```

---

## 通用模式

### 类型转换

对于需要高精度的激活函数（如 GELU），使用 float32 进行中间计算：

```python
# 加载时转换为 float32
x = tl.load(x_ptr + offset, mask=mask, other=0.0).to(tl.float32)

# 计算
result = compute_function(x)

# 存储时转换回原类型
tl.store(y_ptr + offset, result.to(y_ptr.dtype.element_ty), mask=mask)
```

### 固定核心数启动（Ascend 推荐）

```python
import torch_npu


@triton.jit
def activation_kernel_fixed_core(
    x_ptr,
    y_ptr,
    n_elements: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    CORE_NUM: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    
    for offset in range(pid, n_elements, CORE_NUM * BLOCK_SIZE):
        offsets = offset + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        
        x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        y = 0.5 * x * (1.0 + tl.erf(x / 1.41421356237))
        
        tl.store(y_ptr + offsets, y.to(y_ptr.dtype.element_ty), mask=mask)


def gelu_fixed_core(x: torch.Tensor) -> torch.Tensor:
    try:
        num_cores = torch_npu.npu.npu_config.get_device_limit(0).get("vector_core_num", 40)
    except:
        num_cores = 40
    
    y = torch.empty_like(x)
    n_elements = x.numel()
    BLOCK_SIZE = 512
    
    grid = (num_cores,)
    activation_kernel_fixed_core[grid](x, y, n_elements, BLOCK_SIZE, num_cores)
    
    return y
```

---

## 常见错误

| 错误 | 原因 | 解决方案 |
|------|------|---------|
| 精度损失 | 使用低精度计算 | 使用 float32 进行中间计算 |
| UB 溢出 | BLOCK_SIZE 过大 | 减小 BLOCK_SIZE 至 512 或 256 |
| 结果不一致 | 输入不是连续内存 | 使用 `.contiguous()` 或确保输入连续 |

---