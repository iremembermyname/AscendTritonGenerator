# Triton 性能优化指南

本文档提供Triton算子优化的实用技巧，重点关注Ascend NPU平台。

---

## 目录

1. [内存访问优化](#1-内存访问优化)
2. [UB容量优化](#2-ub容量优化)
3. [流水线优化](#3-流水线优化)
4. [分核优化](#4-分核优化)
5. [离散访存优化](#5-离散访存优化)
6. [数据类型优化](#6-数据类型优化)
7. [特定算子优化](#7-特定算子优化)
8. [性能调试](#8-性能调试)

---

## 1. 内存访问优化

### 1.1 连续内存访问

**原则**：尽量访问连续的内存地址，避免离散访问。

```python
# 好：连续访问
offsets = tl.arange(0, BLOCK_SIZE)
x = tl.load(ptr + offsets)

# 差：离散访问
indices = [0, 5, 10, 15, ...]  # 非连续
x = tl.load(ptr + indices)
```

### 1.2 使用Block Pointer

对于规则访问模式，使用block pointer：

```python
block_ptr = tl.make_block_ptr(
    base=ptr,
    shape=(M, N),
    strides=(stride_m, stride_n),
    offsets=(0, 0),
    block_shape=(BLOCK_M, BLOCK_N),
    order=(1, 0),
)
x = tl.load(block_ptr)
block_ptr = tl.advance(block_ptr, (BLOCK_M, 0))
```

### 1.3 连续内存的一维访问优化

```python
# 方案1：转连续后用一维访问（推荐）
if not input_tensor.is_contiguous():
    input_tensor = input_tensor.contiguous()

output_tensor = torch.empty_like(input_tensor)
n_elements = input_tensor.numel()
grid = (triton.cdiv(n_elements, BLOCK_SIZE),)

elementwise_kernel[grid](input_tensor, output_tensor, n_elements, BLOCK_SIZE)
```

---

## 2. UB容量优化

### 2.1 控制UB使用量

**原则**：单次循环UB占用 ≤ 85KB。

```python
# 计算UB占用
def calculate_ub(block_size, num_tensors, dtype_bytes=2):
    return block_size * num_tensors * dtype_bytes / 1024  # KB
```

### 2.2 减少中间变量

```python
# 差：多个中间变量
a = x + y
b = a * z
c = b - w
result = c / v

# 好：合并计算
result = ((x + y) * z - w) / v
```

### 2.3 及时释放变量

```python
# 好：边计算边store
for i in range(num_blocks):
    block = compute_block(i)
    tl.store(output_ptr + offsets, block)  # 及时释放
```

---

## 3. 流水线优化

### 3.1 避免带other的load

```python
# 差：带other的load
x = tl.load(ptr + offsets, mask=mask, other=0.0)

# 好：分离load和where
x = tl.load(ptr + offsets, mask=mask)
x = tl.where(mask, x, 0.0)
```

### 3.2 避免数据依赖

```python
# 差：迭代间有依赖
for i in range(N):
    offset = prev_offset + stride  # 依赖前一次
    x = tl.load(ptr + offset)
    prev_offset = offset

# 好：独立计算偏移
for i in range(N):
    offset = base + i * stride  # 独立计算
    x = tl.load(ptr + offset)
```

---

## 4. 分核优化

### 4.1 负载均衡

```python
# 计算每个核的工作量
num_tokens_per_core = (num_tokens + num_cores - 1) // num_cores

# 分配工作
for core_id in range(num_cores):
    start = core_id * num_tokens_per_core
    end = min(start + num_tokens_per_core, num_tokens)
```

### 4.2 交错循环处理（推荐）

```python
@triton.jit
def row_processing_kernel(
    input_ptr, output_ptr, 
    M, N,
    BLOCK_N: tl.constexpr,
    CORE_NUM: tl.constexpr,
):
    pid = tl.program_id(0)
    
    # 交错处理：每个核心处理 pid, pid+CORE_NUM, pid+2*CORE_NUM, ... 行
    for row_idx in range(pid, M, CORE_NUM):
        row_ptr = input_ptr + row_idx * stride_m
        # 处理当前行
        ...

# 启动方式
grid = (VEC_CORE_NUM,)
row_processing_kernel[grid](...)
```

---

## 5. 离散访存优化

### 5.1 使用tl.gather替代直接离散访问

```python
# 差：直接从GM离散访问
idx = tl.load(idx_ptr + rn)
val = tl.load(x_ptr + idx, mask=mask)  # 离散访问，退化为标量

# 好：先加载到UB，再gather
rm = tl.arange(0, M)
x_shared = tl.load(x_ptr + rm)  # 批量加载到UB
val = tl.gather(x_shared, idx, 0)  # 从UB中gather
```

---

## 6. 数据类型优化

### 6.1 避免int64运算

Ascend矢量运算单元不支持int64，使用int32替代：

```python
# 差：int64运算退化为标量
x = torch.randint(0, 100, (1, vector_len), device='npu', dtype=torch.int64)

# 好：使用int32启用向量化
x = torch.randint(0, 100, (1, vector_len), device='npu', dtype=torch.int32)
```

### 6.2 cmp操作类型转换

cmp操作不支持int32/int64，需转换为float32：

```python
# 差：int64比较退化为标量
cols = tl.arange(0, BLOCK_N)  # int64
xbar = tl.where(cols < N, x - mean, 0.0)

# 好：转换为float32启用向量化
cols = tl.arange(0, BLOCK_N)
cols_cmp = cols.to(tl.float32)
xbar = tl.where(cols_cmp < N, x - mean, 0.0)
```

---

## 7. 特定算子优化

### 7.1 Matmul切分优化

合理的切分是提升matmul算子性能的关键：

| 转置情况 | 分块行宽 | 推荐配置 |
|---------|---------|---------|
| A、B都不转置 | K0和N0 | M0=128, K0=256, N0=256 |
| A不转置，B转置 | 都是K0 | K0=256 |
| A、B都转置 | M0和K0 | M0=256, K0=256, N0=128 |

### 7.2 Flash Attention优化

使用在线Softmax算法：

```python
# 初始化全局统计量
m_i = -float("inf")  # 全局最大值
l_i = 0.0           # 全局exp和
acc = 0.0           # 输出累加器

# 分块处理
for start_n in range(0, seq_len, BLOCK_SIZE):
    scores = tl.load(scores_ptr + start_n, mask=load_mask, other=-float("inf"))
    
    # 更新全局最大值
    m_ij = tl.maximum(m_i, tl.max(scores, 0))
    
    # 计算当前块的exp值
    scores = scores - m_ij
    p = tl.math.exp2(scores)
    
    # 更新全局exp和
    l_ij = tl.sum(p, 0)
    alpha = tl.math.exp2(m_i - m_ij)
    l_i = l_i * alpha + l_ij
    
    # 更新输出累加器
    acc = acc * alpha + p
    m_i = m_ij

# 最终归一化
acc = acc / l_i
```

---

## 8. 性能调试

### 8.1 使用msprof

```bash
msprof op --output=./profile --kernel-name="my_kernel" \
    --warm-up=20 --launch-count=20 python test.py
```

### 8.2 关键指标

| 指标 | 说明 | 优化目标 |
|------|------|---------|
| Task Duration | 总执行时间 | 最小化 |
| MTE Utilization | MTE利用率 | 与Vector并行 |
| Vector Utilization | Vector利用率 | 最大化 |
| UB Usage | UB使用量 | ≤ 85KB |
