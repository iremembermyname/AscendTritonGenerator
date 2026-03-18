# Triton 性能优化指南

本文档提供Triton算子优化的实用技巧，重点关注Ascend NPU平台。

---

## 目录

1. [内存访问优化](#1-内存访问优化)
2. [存储容量优化](#2-存储容量优化)
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

## 2. 存储容量优化

### 2.1 算子类型识别

在优化存储容量前，首先识别算子类型：

| 类型 | 特征 | 存储 | 约束 |
|------|------|------|------|
| Vector | 不使用 tl.dot | UB | ≤ 85KB/循环 |
| Cube | 使用 tl.dot | L0A/L0B/L0C | L0A≤64KB, L0B≤64KB, L0C≤128KB |
| CV 混合 | tl.dot + 向量运算 | UB + L0 系列 | 需特殊处理 |

### 2.2 Vector 算子 UB 优化

**原则**：单次循环 UB 占用 ≤ 85KB。

```python
# 计算 UB 占用
def calculate_ub(block_size, num_tensors, dtype_bytes=2):
    return block_size * num_tensors * dtype_bytes / 1024  # KB
```

**减少中间变量**：

```python
# 差：多个中间变量
a = x + y
b = a * z
c = b - w
result = c / v

# 好：合并计算
result = ((x + y) * z - w) / v
```

**及时释放变量**：

```python
# 好：边计算边store
for i in range(num_blocks):
    block = compute_block(i)
    tl.store(output_ptr + offsets, block)  # 及时释放
```

### 2.3 Cube 算子 L0 优化

**原则**：分块大小需满足 L0A/L0B/L0C 容量约束。

**约束公式**：
- L0A: BLOCK_M × BLOCK_K × sizeof(dtype) ≤ 64KB
- L0B: BLOCK_K × BLOCK_N × sizeof(dtype) ≤ 64KB
- L0C: BLOCK_M × BLOCK_N × sizeof(accumulator_dtype) ≤ 128KB

**推荐配置** (FP16/BF16)：
- 不转置: BLOCK_M=128, BLOCK_K=256, BLOCK_N=256
- B 转置: BLOCK_K=256
- 都转置: BLOCK_M=256, BLOCK_K=256, BLOCK_N=128

**约束验证代码**：

```python
# 验证 Cube 分块约束
def validate_cube_blocks(BLOCK_M, BLOCK_K, BLOCK_N, dtype_size=2, acc_size=4):
    L0A = BLOCK_M * BLOCK_K * dtype_size
    L0B = BLOCK_K * BLOCK_N * dtype_size
    L0C = BLOCK_M * BLOCK_N * acc_size
    
    assert L0A <= 64 * 1024, f"L0A overflow: {L0A} > 64KB"
    assert L0B <= 64 * 1024, f"L0B overflow: {L0B} > 64KB"
    assert L0C <= 128 * 1024, f"L0C overflow: {L0C} > 128KB"
    return True

# 示例
BLOCK_M, BLOCK_K, BLOCK_N = 128, 256, 128
validate_cube_blocks(BLOCK_M, BLOCK_K, BLOCK_N)
# L0A: 128 * 256 * 2 = 64KB ✓
# L0B: 256 * 128 * 2 = 64KB ✓
# L0C: 128 * 128 * 4 = 64KB ✓
```

### 2.4 CV 混合算子优化

CV 混合算子指运算过程中既使用了 AI Core (Cube) 又使用了 Vector Core，需要特殊优化策略。

参考：`triton-ascend/docs/zh/migration_guide/architecture_difference.md`

**autotune 编译选项**：

| 选项 | 说明 | 默认值 |
|------|------|--------|
| `multibuffer` | 开启流水并行数据搬运 | true |
| `limit_auto_multi_buffer_only_for_local_buffer` | CV算子优化项，cube搬出优化 | None |
| `tile_mix_vector_loop` | CV算子优化项，vector切分份数 | None，如 [2,4,8] |
| `tile_mix_cube_loop` | CV算子优化项，cube切分份数 | None，如 [2,4,8] |

```python
# autotune 配置示例
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64, 'multibuffer': True}),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64, 'multibuffer': True, 'tile_mix_vector_loop': [2, 4]}),
    ],
    key=['M', 'N', 'K'],
)
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

### 4.3 核心数选择

| 算子类型 | 核心数获取 | 默认值 |
|---------|-----------|--------|
| Vector | `vector_core_num` | 40-48 |
| Cube | `cube_core_num` | 20-24 |

```python
import torch_npu

# Vector 算子核心数
VEC_CORE_NUM = torch_npu.npu.npu_config.get_device_limit(0).get("vector_core_num", 40)

# Cube 算子核心数
CUBE_CORE_NUM = torch_npu.npu.npu_config.get_device_limit(0).get("cube_core_num", 20)
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

合理的切分是提升 matmul 算子性能的关键，需满足 L0 缓存约束：

| 转置情况 | 分块行宽 | 推荐配置 |
|---------|---------|---------|
| A、B都不转置 | K0和N0 | M0=128, K0=256, N0=256 |
| A不转置，B转置 | 都是K0 | K0=256 |
| A、B都转置 | M0和K0 | M0=256, K0=256, N0=128 |

**L0 约束验证**：
```python
# FP16 示例 (dtype_size = 2 bytes, acc_size = 4 bytes)
BLOCK_M, BLOCK_K, BLOCK_N = 128, 256, 128

# L0A: 128 * 256 * 2 = 64KB ✓
# L0B: 256 * 128 * 2 = 64KB ✓
# L0C: 128 * 128 * 4 = 64KB ✓
```

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
| UB Usage | UB使用量 (Vector算子) | ≤ 85KB |
| L0 Usage | L0使用量 (Cube算子) | L0A/L0B≤64KB, L0C≤128KB |
