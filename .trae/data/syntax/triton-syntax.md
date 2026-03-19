# Triton 语法参考

本文档提供Triton编程语言的核心语法参考。

---

## 目录

1. [核心概念](#1-核心概念)
2. [标准内核结构](#2-标准内核结构)
3. [数据类型](#3-数据类型)
4. [内存操作](#4-内存操作)
5. [张量操作](#5-张量操作)
6. [数学运算](#6-数学运算)
7. [规约操作](#7-规约操作)
8. [条件操作](#8-条件操作)
9. [原子操作](#9-原子操作)
10. [autotune使用](#10-autotune使用)

---

## 1. 核心概念

### 内核 (Kernel)
- 使用 `@triton.jit` 装饰的 Python 函数
- 每个内核实例处理数据的一个子集，通过程序 ID 区分

### 网格 (Grid) 与块 (Block)
- **网格**: 内核启动时的并行维度配置
- **块**: 每个程序实例处理的数据块大小
- **关系**: `grid_size = ceil(total_elements / block_size)`

### 内存层次
- **全局内存**: 主内存，所有程序可访问，延迟高
- **共享内存**: 块内共享，延迟低，容量有限

---

## 2. 标准内核结构

所有 Triton 内核都遵循相同的五步结构模式：

```python
@triton.jit
def standard_kernel(
    output_ptr, input_ptr, n_elements, 
    BLOCK_SIZE: tl.constexpr,
):
    # 1. 获取程序 ID 和计算偏移
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    
    # 2. 创建边界掩码
    mask = offsets < n_elements
    
    # 3. 加载数据
    data = tl.load(input_ptr + offsets, mask=mask)
    
    # 4. 执行计算
    result = compute_function(data)
    
    # 5. 存储结果
    tl.store(output_ptr + offsets, result, mask=mask)
```

### 内核启动方式

```python
class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input_tensor):
        output_tensor = torch.empty_like(input_tensor)
        BLOCK_SIZE = 1024  
        grid = (triton.cdiv(input_tensor.numel(), BLOCK_SIZE),)
        
        kernel[grid](
            output_tensor, input_tensor, input_tensor.numel(),
            BLOCK_SIZE=BLOCK_SIZE,
        )
        return output_tensor
```

---

## 3. 数据类型

### 基本类型

| 类型 | 说明 | 大小 |
|------|------|------|
| `tl.float16` | 半精度浮点 | 2 bytes |
| `tl.float32` | 单精度浮点 | 4 bytes |
| `tl.bfloat16` | BF16 | 2 bytes |
| `tl.int8` | 8位整数 | 1 byte |
| `tl.int32` | 32位整数 | 4 bytes |
| `tl.int64` | 64位整数 | 8 bytes |

### 类型转换

```python
x_f32 = x.to(tl.float32)
x_bf16 = x.to(tl.bfloat16)
```

### constexpr

编译期常量，用于Block大小、形状等：

```python
@triton.jit
def kernel(
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pass
```

---

## 4. 内存操作

### tl.load

```python
# 基本加载
x = tl.load(ptr + offsets)

# 带mask加载
x = tl.load(ptr + offsets, mask=mask)
```

### tl.store

```python
# 基本存储
tl.store(ptr + offsets, value)

# 带mask存储
tl.store(ptr + offsets, value, mask=mask)
```

### 块指针 (Block Pointer)

```python
# 创建块指针
block_ptr = tl.make_block_ptr(
    base=ptr,
    shape=(M, N),
    strides=(stride_m, stride_n),
    offsets=(0, 0),
    block_shape=(BLOCK_M, BLOCK_N),
    order=(1, 0),
)

# 加载
x = tl.load(block_ptr)

# 存储后推进
block_ptr = tl.advance(block_ptr, (BLOCK_M, 0))
```

---

## 5. 张量操作

### 创建张量

```python
# arange - 创建范围
offsets = tl.arange(0, BLOCK_SIZE)

# zeros - 创建零张量
zeros = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)

# full - 创建填充张量
full = tl.full([BLOCK_M, BLOCK_N], value=1.0, dtype=tl.float32)
```

### 形状操作

```python
# reshape - 改变形状
x = x.reshape(BLOCK_M, BLOCK_N)

# broadcast_to - 广播
x = tl.broadcast_to(x, [BLOCK_M, BLOCK_N])

# transpose - 转置
x = tl.trans(x)  # 或 x.T
```

---

## 6. 数学运算

### 基本运算

```python
y = x + y
y = x - y
y = x * y
y = x / y
y = tl.pow(x, 2)
y = tl.sqrt(x)
y = tl.rsqrt(x)  # 1/sqrt(x)
```

### 三角函数

```python
y = tl.sin(x)
y = tl.cos(x)
y = tl.tan(x)
y = tl.tanh(x)
```

### 指数和对数

```python
y = tl.exp(x)
y = tl.log(x)
y = tl.log2(x)
y = tl.log10(x)
```

### 其他数学函数

```python
y = tl.abs(x)
y = tl.floor(x)
y = tl.ceil(x)
y = tl.round(x)
y = tl.clamp(x, min_val, max_val)
```

---

## 7. 规约操作

```python
# 求和
sum_x = tl.sum(x, axis=0)

# 最大值
max_x = tl.max(x, axis=0)

# 最小值
min_x = tl.min(x, axis=0)

# 均值
mean_x = tl.mean(x, axis=0)
```

---

## 8. 条件操作

### tl.where

```python
y = tl.where(condition, x, y)  # condition为True选x，否则选y
```

### 比较

```python
eq = x == y
ne = x != y
lt = x < y
le = x <= y
gt = x > y
ge = x >= y
```

---

## 9. 原子操作

```python
# 原子加
tl.atomic_add(ptr + offsets, value, mask=mask)

# 原子最大/最小
tl.atomic_max(ptr + offsets, value, mask=mask)
tl.atomic_min(ptr + offsets, value, mask=mask)

# 原子交换
old_value = tl.atomic_xchg(ptr + offsets, value, mask=mask)

# 原子比较交换
old_value = tl.atomic_cas(ptr + offsets, cmp, val, mask=mask)
```

---

## 10. autotune使用

```python
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128}),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 64}),
    ],
    key=['M', 'N', 'K'],  # 当这些参数变化时触发重新autotune
)
@triton.jit
def matmul_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
    BLOCK_SIZE_M: tl.constexpr,  # configs中的参数必须声明为constexpr
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    pass

# 调用时不要传递configs中的参数
grid = lambda meta: (triton.cdiv(M, meta['BLOCK_SIZE_M']) * triton.cdiv(N, meta['BLOCK_SIZE_N']),)
matmul_kernel[grid](a, b, c, M, N, K, ...)
```

**关键要点**：
1. grid必须使用lambda：`grid = lambda meta: (...)`
2. 不要传递configs参数
3. configs参数必须是constexpr
