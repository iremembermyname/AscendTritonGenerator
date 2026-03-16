# CUDA与Ascend Triton差异对照

本文档详细对比CUDA和Ascend平台上的Triton API差异，帮助代码迁移。

---

## 目录

1. [核心API兼容性](#1-核心api兼容性)
   - [完全兼容的API](#11-完全兼容的api)
   - [部分兼容的API](#12-部分兼容的api)
   - [不支持的API](#13-不支持的api)
2. [硬件限制差异](#2-硬件限制差异)
   - [Block大小限制](#21-block大小限制)
   - [Grid大小限制](#22-grid大小限制)
   - [Shared Memory / UB](#23-shared-memory--ub)
3. [性能特性差异](#3-性能特性差异)
   - [内存访问](#31-内存访问)
   - [计算密度](#32-计算密度)
   - [流水线](#33-流水线)
4. [常见转换模式](#4-常见转换模式)
   - [Block大小调整](#41-block大小调整)
   - [移除不支持的API](#42-移除不支持的api)
   - [替代Warp Shuffle](#43-替代warp-shuffle)
   - [优化内存访问](#44-优化内存访问)
5. [数据类型差异](#5-数据类型差异)
   - [支持的数据类型](#51-支持的数据类型)
   - [精度注意事项](#52-精度注意事项)
6. [转换检查清单](#6-转换检查清单)
   - [必须检查项](#61-必须检查项)
   - [建议优化项](#62-建议优化项)
   - [验证项](#63-验证项)
7. [迁移示例](#7-迁移示例)
   - [Softmax迁移](#71-softmax迁移)
   - [MatMul迁移](#72-matmul迁移)

---

## 1. 核心API兼容性

### 1.1 完全兼容的API

以下API在CUDA和Ascend上行为一致：

| API | 说明 |
|-----|------|
| `tl.load` | 加载数据 |
| `tl.store` | 存储数据 |
| `tl.arange` | 创建范围 |
| `tl.zeros` | 创建零张量 |
| `tl.full` | 创建填充张量 |
| `tl.sum` | 求和规约 |
| `tl.max` | 最大值规约 |
| `tl.min` | 最小值规约 |
| `tl.mean` | 均值规约 |
| `tl.exp` | 指数函数 |
| `tl.log` | 对数函数 |
| `tl.sqrt` | 平方根 |
| `tl.sin` | 正弦函数 |
| `tl.cos` | 余弦函数 |
| `tl.abs` | 绝对值 |
| `tl.where` | 条件选择 |

### 1.2 部分兼容的API

| API | CUDA行为 | Ascend行为 | 注意事项 |
|-----|---------|-----------|---------|
| `tl.dot` | 支持多种精度 | 支持BF16/FP16 | 累加器建议用FP32 |
| `tl.atomic_add` | 高性能 | 性能可能较低 | 批量操作更高效 |
| `tl.atomic_max` | 支持 | 支持 | 同上 |
| `tl.atomic_min` | 支持 | 支持 | 同上 |

### 1.3 不支持的API

| API | CUDA | Ascend | 替代方案 |
|-----|------|--------|---------|
| `tl.debug_barrier` | 支持 | 不支持 | 移除 |
| `tl.device_print` | 支持 | 不支持 | 移除或条件编译 |
| `tl.shuffle` | 支持 | 不支持 | 使用shared memory替代 |
| `tl.experimental.async_copy` | 支持 | 不支持 | 使用同步copy |

---

## 2. 硬件限制差异

### 2.1 Block大小限制

| 平台 | 最大BLOCK_SIZE | 推荐值 |
|------|---------------|--------|
| CUDA | 2048+ | 512-1024 |
| Ascend | 1024 | 256-512 |

```python
# CUDA代码
BLOCK_SIZE = 2048  # 可能工作

# Ascend转换
BLOCK_SIZE = 1024  # 必须 <= 1024
```

### 2.2 Grid大小限制

| 平台 | 最大Grid维度 |
|------|-------------|
| CUDA | 3维 (x, y, z) |
| Ascend | 3维 (x, y, z) |

### 2.3 Shared Memory / UB

| 平台 | 容量 | 管理方式 |
|------|------|---------|
| CUDA | 48KB-164KB | 显式shared memory |
| Ascend | 192KB | 编译器自动管理UB |

```python
# CUDA代码
shared = tl.shared_memory(tl.float32, [BLOCK_SIZE])

# Ascend转换
# 通常不需要显式shared memory
# 编译器会自动将数据放入UB
```

---

## 3. 性能特性差异

### 3.1 内存访问

| 特性 | CUDA | Ascend |
|------|------|--------|
| 连续访问 | 高效 | 高效 |
| 跨步访问 | 较高效 | 效率较低 |
| 随机访问 | 较低效 | 效率很低 |

**建议**：Ascend上更强调连续内存访问。

### 3.2 计算密度

| 特性 | CUDA | Ascend |
|------|------|--------|
| 计算密集型 | 高效 | 高效 |
| 内存密集型 | 较高效 | 需要优化 |

**建议**：Ascend上需要更高的计算密度来隐藏内存延迟。

### 3.3 流水线

| 特性 | CUDA | Ascend |
|------|------|--------|
| 自动流水线 | 较好 | 需要手动优化 |
| Double Buffering | 自动 | 需要控制UB占用 |

---

## 4. 常见转换模式

### 4.1 Block大小调整

```python
# CUDA
@triton.jit
def kernel(..., BLOCK_SIZE: tl.constexpr = 2048):
    ...

# Ascend
@triton.jit
def kernel(..., BLOCK_SIZE: tl.constexpr = 1024):
    ...
```

### 4.2 移除不支持的API

```python
# CUDA
@triton.jit
def kernel(...):
    ...
    tl.debug_barrier()  # CUDA同步
    ...

# Ascend
@triton.jit
def kernel(...):
    ...
    # tl.debug_barrier()  # 移除
    ...
```

### 4.3 替代Warp Shuffle

```python
# CUDA: 使用warp shuffle
@triton.jit
def cuda_kernel(...):
    value = tl.load(ptr + lane_id)
    shuffled = tl.shuffle(value, src_lane=0)
    ...

# Ascend: 使用atomic或reduce替代
@triton.jit
def ascend_kernel(...):
    value = tl.load(ptr + lane_id)
    # 广播第一个lane的值
    broadcast_value = tl.sum(value, axis=0) / num_lanes
    ...
```

### 4.4 优化内存访问

```python
# CUDA: 可能容忍非连续访问
# Ascend: 需要优化为连续访问

# CUDA
for i in range(num_rows):
    row = tl.load(ptr + row_indices[i] * stride + offsets)

# Ascend优化
# 如果row_indices是连续的，直接批量加载
if is_consecutive(row_indices):
    rows = tl.load(ptr + row_indices[:, None] * stride + offsets[None, :])
else:
    # 逐行加载
    for i in range(num_rows):
        row = tl.load(ptr + row_indices[i] * stride + offsets)
```

---

## 5. 数据类型差异

### 5.1 支持的数据类型

| 类型 | CUDA | Ascend |
|------|------|--------|
| FP32 | ✓ | ✓ |
| FP16 | ✓ | ✓ |
| BF16 | ✓ | ✓ |
| INT8 | ✓ | ✓ |
| INT32 | ✓ | ✓ |
| INT64 | ✓ | 部分支持 |

### 5.2 精度注意事项

```python
# 累加操作建议使用FP32
# CUDA和Ascend都适用

@triton.jit
def kernel(...):
    # 使用FP32累加
    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
    
    for k in range(0, K, BLOCK_K):
        a = tl.load(a_ptr + ...).to(tl.float32)
        b = tl.load(b_ptr + ...).to(tl.float32)
        acc += tl.dot(a, b)
    
    # 最后转换回目标精度
    result = acc.to(tl.bfloat16)
    tl.store(out_ptr + ..., result)
```

---

## 6. 转换检查清单

### 6.1 必须检查项

- [ ] BLOCK_SIZE <= 1024
- [ ] 移除 `tl.debug_barrier`
- [ ] 移除 `tl.device_print`
- [ ] 替换 `tl.shuffle`
- [ ] 检查内存访问模式

### 6.2 建议优化项

- [ ] 优化为连续内存访问
- [ ] 控制UB占用 <= 85KB
- [ ] 使用FP32累加
- [ ] 优化分核策略

### 6.3 验证项

- [ ] 编译通过
- [ ] 精度验证
- [ ] 性能测试

---

## 7. 迁移示例

### 7.1 Softmax迁移

**CUDA版本**：
```python
@triton.jit
def softmax_kernel_cuda(
    input_ptr, output_ptr,
    M, N, stride_m,
    BLOCK_SIZE: tl.constexpr = 1024,
):
    row_idx = tl.program_id(0)
    row_start = row_idx * stride_m
    
    # 找最大值
    max_val = float("-inf")
    for i in range(0, N, BLOCK_SIZE):
        offsets = i + tl.arange(0, BLOCK_SIZE)
        mask = offsets < N
        x = tl.load(input_ptr + row_start + offsets, mask=mask)
        max_val = tl.maximum(max_val, tl.max(x, axis=0))
    
    # ... 其余代码
```

**Ascend版本**：
```python
@triton.jit
def softmax_kernel_ascend(
    input_ptr, output_ptr,
    M, N, stride_m,
    BLOCK_SIZE: tl.constexpr = 512,  # 调整为更小的值
):
    row_idx = tl.program_id(0)
    row_start = row_idx * stride_m
    
    # 找最大值
    max_val = float("-inf")
    for i in range(0, N, BLOCK_SIZE):
        offsets = i + tl.arange(0, BLOCK_SIZE)
        mask = offsets < N
        x = tl.load(input_ptr + row_start + offsets, mask=mask)
        max_val = tl.maximum(max_val, tl.max(x, axis=0))
    
    # ... 其余代码（基本相同）
```

### 7.2 MatMul迁移

**CUDA版本**：
```python
@triton.jit
def matmul_kernel_cuda(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    BLOCK_M: tl.constexpr = 128,
    BLOCK_N: tl.constexpr = 128,
    BLOCK_K: tl.constexpr = 32,
):
    # ... CUDA实现
```

**Ascend版本**：
```python
@triton.jit
def matmul_kernel_ascend(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    BLOCK_M: tl.constexpr = 64,   # 减小以控制UB
    BLOCK_N: tl.constexpr = 64,
    BLOCK_K: tl.constexpr = 32,
):
    # ... 相同实现，但注意UB占用
```
