# Ascend扩展API

本文档介绍Ascend NPU上的Triton扩展API和优化技术。

---

## 目录

1. [Ascend硬件架构](#1-ascend硬件架构)
2. [性能优化技术](#2-性能优化技术)
   - [Double Buffering](#21-double-buffering)
   - [多Token并行处理](#22-多token并行处理)
   - [流水线优化](#23-流水线优化)
   - [内存访问优化](#24-内存访问优化)
3. [Ascend特定API](#3-ascend特定api)
4. [常见优化模式](#4-常见优化模式)
   - [Softmax优化](#41-softmax优化)
   - [LayerNorm优化](#42-layernorm优化)
   - [MatMul优化](#43-matmul优化)
5. [性能分析工具](#5-性能分析工具)
6. [常见问题](#6-常见问题)

---

## 1. Ascend硬件架构

### 1.1 计算引擎

Ascend NPU包含三大计算引擎：

| 引擎 | 功能 | 特点 |
|------|------|------|
| Scalar | 标量计算 | 地址计算、循环控制、条件判断 |
| MTE | 数据搬运 | GM ↔ UB 数据传输 |
| Vector | 向量计算 | 算术运算、规约、类型转换 |

**关键**：三个引擎可以流水并行执行，是性能优化的核心。

### 1.2 存储层次

```
Global Memory (GM)
    ↓ tl.load
Unified Buffer (UB) ← Vector计算
    ↓ tl.store
Global Memory (GM)
```

### 1.3 UB容量

| 芯片型号 | UB容量 | Double Buffering可用 | 建议使用量 |
|---------|--------|---------------------|-----------|
| 910B | 192 KB | 96 KB | ~85 KB |

---

## 2. 性能优化技术

### 2.1 Double Buffering

**原理**：将UB分为两个Buffer，实现MTE和Vector的流水并行。

```
时间 →
MTE:    [load_A] [load_B] [load_A] [load_B] ...
Vector:         [calc_A] [calc_B] [calc_A] ...
```

**要求**：单次循环UB占用 ≤ 总容量的一半。

```python
# 正确：控制UB占用
BLOCK_SIZE = 1024  # 每个元素2字节，共2KB
# 多个tensor同时占用时，确保总和 ≤ 85KB
```

### 2.2 多Token并行处理

**原理**：一次循环处理多个Token，减少循环次数，提升效率。

```python
@triton.jit
def kernel(...):
    # 计算单次循环最大处理Token数
    # N = 85 * 1024 // S_token
    N = 8  # 假设计算结果
    
    for i in range(0, num_tokens, N):
        # 批量加载N个Token
        tokens = tl.load(ptr + i * stride + offsets)
        # 批量计算
        result = compute(tokens)
        # 批量存储
        tl.store(out_ptr + i * stride + offsets, result)
```

### 2.3 流水线优化

**目标**：让MTE和Vector并行执行。

```python
# 好的做法：加载和计算交织
for i in range(num_loops):
    x = tl.load(x_ptr + offsets)  # MTE加载
    y = compute(x)                # Vector计算
    tl.store(y_ptr + offsets, y)  # MTE存储

# 避免：破坏流水线
# 不要使用带other的load，它会阻止MTE独立执行
# x = tl.load(ptr, mask=m, other=0.0)  # 避免！
```

### 2.4 内存访问优化

**原则**：连续访问，避免离散访问。

```python
# 好：连续访问
x = tl.load(ptr + tl.arange(0, BLOCK_SIZE))

# 差：离散访问
# indices = [0, 5, 10, 15, ...]  # 非连续
# x = tl.load(ptr + indices)

# 对于离散数据，需要逐行加载
for i in range(num_rows):
    row = tl.load(ptr + i * row_stride + tl.arange(0, row_size))
```

---

## 3. Ascend特定API

### 3.1 获取核数

```python
# 获取Vector核数量
import torch_npu

def get_num_cores():
    return torch_npu.npu.get_device_properties("npu:0").multi_processor_count
```

### 3.2 分核策略

```python
@triton.jit
def kernel(..., NUM_CORES: tl.constexpr):
    pid = tl.program_id(0)
    
    # 负载均衡分核
    num_tokens_per_core = (num_tokens + NUM_CORES - 1) // NUM_CORES
    
    # 当前核处理的Token范围
    start = pid * num_tokens_per_core
    end = min(start + num_tokens_per_core, num_tokens)
    
    for i in range(start, end):
        # 处理Token i
        pass
```

### 3.3 环境变量

```python
# 设置环境变量
import os

# 启用Double Buffering
os.environ["TRITON_ASCEND_ENABLE_DOUBLE_BUFFER"] = "1"

# 设置编译优化级别
os.environ["TRITON_ASCEND_OPT_LEVEL"] = "O3"
```

---

## 4. 常见优化模式

### 4.1 Softmax优化

```python
@triton.jit
def softmax_kernel(
    input_ptr, output_ptr,
    M, N,
    stride_m,
    BLOCK_N: tl.constexpr,
):
    # 每个program处理一行
    row_idx = tl.program_id(0)
    
    # 计算行起始位置
    row_start = row_idx * stride_m
    
    # 分块处理
    max_val = float("-inf")
    sum_exp = 0.0
    
    # 第一遍：找最大值
    for block_start in range(0, N, BLOCK_N):
        offsets = block_start + tl.arange(0, BLOCK_N)
        mask = offsets < N
        x = tl.load(input_ptr + row_start + offsets, mask=mask)
        max_val = tl.maximum(max_val, tl.max(x, axis=0))
    
    # 第二遍：计算exp和求和
    for block_start in range(0, N, BLOCK_N):
        offsets = block_start + tl.arange(0, BLOCK_N)
        mask = offsets < N
        x = tl.load(input_ptr + row_start + offsets, mask=mask)
        exp_x = tl.exp(x - max_val)
        sum_exp += tl.sum(exp_x, axis=0)
    
    # 第三遍：归一化
    for block_start in range(0, N, BLOCK_N):
        offsets = block_start + tl.arange(0, BLOCK_N)
        mask = offsets < N
        x = tl.load(input_ptr + row_start + offsets, mask=mask)
        exp_x = tl.exp(x - max_val)
        out = exp_x / sum_exp
        tl.store(output_ptr + row_start + offsets, out, mask=mask)
```

### 4.2 LayerNorm优化

```python
@triton.jit
def layernorm_kernel(
    x_ptr, y_ptr, weight_ptr, bias_ptr,
    M, N,
    stride_m,
    eps: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    row_idx = tl.program_id(0)
    row_start = row_idx * stride_m
    
    # 计算均值
    sum_x = 0.0
    for block_start in range(0, N, BLOCK_N):
        offsets = block_start + tl.arange(0, BLOCK_N)
        mask = offsets < N
        x = tl.load(x_ptr + row_start + offsets, mask=mask)
        sum_x += tl.sum(x, axis=0)
    mean = sum_x / N
    
    # 计算方差
    sum_var = 0.0
    for block_start in range(0, N, BLOCK_N):
        offsets = block_start + tl.arange(0, BLOCK_N)
        mask = offsets < N
        x = tl.load(x_ptr + row_start + offsets, mask=mask)
        diff = x - mean
        sum_var += tl.sum(diff * diff, axis=0)
    var = sum_var / N
    rstd = tl.rsqrt(var + eps)
    
    # 归一化
    for block_start in range(0, N, BLOCK_N):
        offsets = block_start + tl.arange(0, BLOCK_N)
        mask = offsets < N
        x = tl.load(x_ptr + row_start + offsets, mask=mask)
        w = tl.load(weight_ptr + offsets, mask=mask)
        b = tl.load(bias_ptr + offsets, mask=mask)
        y = (x - mean) * rstd * w + b
        tl.store(y_ptr + row_start + offsets, y, mask=mask)
```

### 4.3 MatMul优化

```python
@triton.jit
def matmul_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    # 分块矩阵乘法
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    # 初始化累加器
    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
    
    # 分块计算
    for k in range(0, K, BLOCK_K):
        # 加载A块
        a_offsets_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        a_offsets_k = k + tl.arange(0, BLOCK_K)
        a_mask_m = a_offsets_m < M
        a_mask_k = a_offsets_k < K
        a = tl.load(a_ptr + a_offsets_m[:, None] * stride_am + 
                    a_offsets_k[None, :] * stride_ak,
                    mask=a_mask_m[:, None] & a_mask_k[None, :])
        
        # 加载B块
        b_offsets_k = k + tl.arange(0, BLOCK_K)
        b_offsets_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        b_mask_k = b_offsets_k < K
        b_mask_n = b_offsets_n < N
        b = tl.load(b_ptr + b_offsets_k[:, None] * stride_bk + 
                    b_offsets_n[None, :] * stride_bn,
                    mask=b_mask_k[:, None] & b_mask_n[None, :])
        
        # 矩阵乘法
        acc += tl.dot(a, b)
    
    # 存储结果
    c_offsets_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    c_offsets_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    c_mask_m = c_offsets_m < M
    c_mask_n = c_offsets_n < N
    tl.store(c_ptr + c_offsets_m[:, None] * stride_cm + 
             c_offsets_n[None, :] * stride_cn,
             acc,
             mask=c_mask_m[:, None] & c_mask_n[None, :])
```

---

## 5. 性能分析工具

### 5.1 msprof

```bash
# 性能分析
msprof op --output=./profile --kernel-name="my_kernel" \
    --warm-up=20 --launch-count=20 python test.py
```

### 5.2 关键指标

| 指标 | 说明 | 优化目标 |
|------|------|---------|
| Task Duration | 总执行时间 | 最小化 |
| MTE Utilization | MTE利用率 | 与Vector并行 |
| Vector Utilization | Vector利用率 | 最大化 |
| UB Usage | UB使用量 | ≤ 85KB |

---

## 6. 常见问题

### 6.1 UB溢出

**症状**：运行时报错或性能严重退化。

**解决**：
1. 检查单次循环UB占用
2. 减小BLOCK_SIZE
3. 减少同时存活的变量

### 6.2 流水线不工作

**症状**：MTE和Vector没有并行执行。

**解决**：
1. 避免使用带other的load
2. 确保循环迭代可独立执行
3. 检查是否有数据依赖

### 6.3 性能不达标

**症状**：执行时间过长。

**解决**：
1. 检查内存访问模式
2. 应用多Token并行处理
3. 优化分核策略
