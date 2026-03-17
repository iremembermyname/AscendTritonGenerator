# Triton语法参考

本文档提供Triton编程语言的核心语法参考，重点关注Ascend NPU上的使用。

---

## 目录

1. [基本概念](#1-基本概念)
   - [Kernel函数](#11-kernel函数)
   - [Program ID](#12-program-id)
   - [Grid配置](#13-grid配置)
2. [数据类型](#2-数据类型)
   - [基本类型](#21-基本类型)
   - [类型转换](#22-类型转换)
   - [constexpr](#23-constexpr)
3. [内存操作](#3-内存操作)
   - [tl.load](#31-tlload)
   - [tl.store](#32-tlstore)
   - [块指针](#33-块指针-block-pointer)
4. [张量操作](#4-张量操作)
5. [数学运算](#5-数学运算)
6. [规约操作](#6-规约操作)
7. [条件操作](#7-条件操作)
8. [循环](#8-循环)
9. [原子操作](#9-原子操作)
10. [Ascend特定注意事项](#10-ascend特定注意事项)
11. [调试技巧](#11-调试技巧)

---

## 1. 基本概念

### 1.1 Kernel函数

Triton kernel使用 `@triton.jit` 装饰器：

```python
import triton
import triton.language as tl

@triton.jit
def my_kernel(
    x_ptr,        # 输入指针
    y_ptr,        # 输出指针
    n_elements,   # 元素数量
    BLOCK_SIZE: tl.constexpr,  # 编译期常量
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = x * 2
    tl.store(y_ptr + offsets, y, mask=mask)
```

### 1.2 Program ID

```python
pid = tl.program_id(axis)  # 获取指定轴的program ID
num_programs = tl.num_programs(axis)  # 获取指定轴的program数量
```

### 1.3 Grid配置

```python
def launch_kernel(x, y, n):
    grid = lambda meta: (triton.cdiv(n, meta['BLOCK_SIZE']),)
    my_kernel[grid](
        x, y, n,
        BLOCK_SIZE=1024,
    )
```

---

## 2. 数据类型

### 2.1 基本类型

| 类型 | 说明 | 大小 |
|------|------|------|
| `tl.float16` | 半精度浮点 | 2 bytes |
| `tl.float32` | 单精度浮点 | 4 bytes |
| `tl.bfloat16` | BF16 | 2 bytes |
| `tl.int8` | 8位整数 | 1 byte |
| `tl.int32` | 32位整数 | 4 bytes |
| `tl.int64` | 64位整数 | 8 bytes |

### 2.2 类型转换

```python
x_f32 = x.to(tl.float32)
x_bf16 = x.to(tl.bfloat16)
```

### 2.3 constexpr

编译期常量，用于Block大小、形状等：

```python
@triton.jit
def kernel(
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    # BLOCK_M 和 BLOCK_N 在编译期确定
    pass
```

---

## 3. 内存操作

### 3.1 tl.load

```python
# 基本加载
x = tl.load(ptr + offsets)

# 带mask加载
x = tl.load(ptr + offsets, mask=mask)

# 带默认值加载（Ascend上避免使用）
# x = tl.load(ptr + offsets, mask=mask, other=0.0)  # 避免！
```

**⚠️ Ascend注意事项**：避免使用 `other` 参数，它会内部调用 `tl.where`，影响流水线性能。

### 3.2 tl.store

```python
# 基本存储
tl.store(ptr + offsets, value)

# 带mask存储
tl.store(ptr + offsets, value, mask=mask)
```

### 3.3 块指针 (Block Pointer)

```python
# 创建块指针
block_ptr = tl.make_block_ptr(
    base=ptr,
    shape=(M, N),
    strides=(stride_m, stride_n),
    offsets=(0, 0),
    block_shape=(BLOCK_M, BLOCK_N),
    order=(1, 0),  # 内存布局顺序
)

# 加载
x = tl.load(block_ptr)

# 存储后推进
block_ptr = tl.advance(block_ptr, (BLOCK_M, 0))
```

---

## 4. 张量操作

### 4.1 创建张量

```python
# arange - 创建范围
offsets = tl.arange(0, BLOCK_SIZE)

# zeros - 创建零张量
zeros = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)

# full - 创建填充张量
full = tl.full([BLOCK_M, BLOCK_N], value=1.0, dtype=tl.float32)
```

### 4.2 形状操作

```python
# reshape - 改变形状
x = x.reshape(BLOCK_M, BLOCK_N)

# broadcast_to - 广播
x = tl.broadcast_to(x, [BLOCK_M, BLOCK_N])

# transpose - 转置
x = tl.trans(x)  # 或 x.T
```

### 4.3 切片操作

```python
# extract_slice - 提取切片
slice_x = tl.extract_slice(x, [0, 0], [BLOCK_M, BLOCK_N])

# insert_slice - 插入切片
tl.insert_slice(x, slice_x, [0, 0])
```

---

## 5. 数学运算

### 5.1 基本运算

```python
# 算术运算
y = x + y
y = x - y
y = x * y
y = x / y

# 幂运算
y = tl.pow(x, 2)
y = tl.sqrt(x)
y = tl.rsqrt(x)  # 1/sqrt(x)
```

### 5.2 三角函数

```python
y = tl.sin(x)
y = tl.cos(x)
y = tl.tan(x)
y = tl.tanh(x)
```

### 5.3 指数和对数

```python
y = tl.exp(x)
y = tl.log(x)
y = tl.log2(x)
y = tl.log10(x)
```

### 5.4 其他数学函数

```python
y = tl.abs(x)
y = tl.floor(x)
y = tl.ceil(x)
y = tl.round(x)
y = tl.clamp(x, min_val, max_val)
```

---

## 6. 规约操作

### 6.1 基本规约

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

### 6.2 规约轴

```python
# 沿轴0规约
sum_0 = tl.sum(x, axis=0)

# 沿轴1规约
sum_1 = tl.sum(x, axis=1)

# 沿所有轴规约
sum_all = tl.sum(x)
```

---

## 7. 条件操作

### 7.1 tl.where

```python
# 条件选择
y = tl.where(condition, x, y)  # condition为True选x，否则选y
```

**⚠️ Ascend注意事项**：`tl.where` 性能较差，尽量避免使用。

### 7.2 比较

```python
# 比较操作
eq = x == y
ne = x != y
lt = x < y
le = x <= y
gt = x > y
ge = x >= y
```

---

## 8. 循环

### 8.1 range循环

```python
@triton.jit
def kernel(...):
    for i in range(num_loops):
        # 循环体
        pass
```

### 8.2 循环展开

```python
# 使用 tl.static_range 控制循环展开
for i in tl.static_range(num_loops):  # 完全展开
    pass
```

---

## 9. 原子操作

### 9.1 原子加

```python
tl.atomic_add(ptr + offsets, value, mask=mask)
```

### 9.2 原子最大/最小

```python
tl.atomic_max(ptr + offsets, value, mask=mask)
tl.atomic_min(ptr + offsets, value, mask=mask)
```

### 9.3 原子交换

```python
old_value = tl.atomic_xchg(ptr + offsets, value, mask=mask)
```

### 9.4 原子比较交换

```python
old_value = tl.atomic_cas(ptr + offsets, cmp, val, mask=mask)
```

---

## 10. Ascend特定注意事项

### 10.1 Block大小限制

```python
# Ascend NPU Block大小限制
MAX_BLOCK_SIZE = 1024
RECOMMENDED_BLOCK_SIZE = 256
```

### 10.2 UB容量限制

```python
# UB容量计算
UB_CAPACITY = 192 * 1024  # 192KB
UB_SAFE_LIMIT = 85 * 1024  # 85KB (考虑Double Buffering)

# 计算单次循环最大处理量
def calculate_max_tokens(hidden_size, dtype_bytes=2):
    return UB_SAFE_LIMIT // (hidden_size * dtype_bytes)
```

### 10.3 内存连续性

```python
# 确保输入张量内存连续
x = x.contiguous()
```

### 10.4 避免的操作

```python
# 避免使用带other的load
# x = tl.load(ptr + offsets, mask=mask, other=0.0)  # 避免！

# 改用
x = tl.load(ptr + offsets, mask=mask)
x = tl.where(mask, x, 0.0)  # 如果必须，手动处理
```

---

## 11. 调试技巧

### 11.1 打印调试

```python
# 在kernel中打印（仅用于调试）
tl.device_print("value:", x)
```

### 11.2 断言

```python
# 编译期断言
tl.static_assert(condition, "message")
```

### 11.3 设备同步

```python
# 同步NPU
torch.npu.synchronize()
```
