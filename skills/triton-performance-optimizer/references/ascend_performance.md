# Ascend性能特性

本文档介绍Ascend NPU的性能特性和优化要点。

---

## 1. 硬件架构

### 1.1 计算引擎

Ascend NPU包含三大计算引擎：

| 引擎 | 功能 | 特点 |
|------|------|------|
| Scalar | 标量计算 | 地址计算、循环控制、条件判断 |
| MTE (Memory Transfer Engine) | 数据搬运 | GM ↔ UB 数据传输 |
| Vector | 向量计算 | 算术运算、规约、类型转换 |

**关键**：三个引擎可以流水并行执行，是性能优化的核心。

### 1.2 存储层次

```
┌─────────────────────────────────────┐
│         Global Memory (GM)          │
│           几十GB容量                 │
│         带宽: ~1.2 TB/s             │
└─────────────────┬───────────────────┘
                  │ tl.load / tl.store
                  ▼
┌─────────────────────────────────────┐
│        Unified Buffer (UB)          │
│           192KB容量                  │
│         高带宽片上存储               │
└─────────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────┐
│          Vector Unit                │
│         向量计算单元                 │
└─────────────────────────────────────┘
```

### 1.3 Vector核数量

| 芯片型号 | Vector核数 |
|---------|-----------|
| 910B1 | 108 |
| 910B2 | 108 |
| 910B3 | 108 |
| 910B4 | 108 |

---

## 2. UB容量管理

### 2.1 容量限制

| 芯片型号 | UB总容量 | Double Buffering可用 | 建议使用量 |
|---------|---------|---------------------|-----------|
| 910B | 192 KB | 96 KB | ~85 KB |

### 2.2 Double Buffering机制

**原理**：将UB分为两个Buffer，实现MTE和Vector的流水并行。

```
时间 →
MTE:    [load_A] [load_B] [load_A] [load_B] ...
Vector:         [calc_A] [calc_B] [calc_A] ...
```

**要求**：单次循环UB占用 ≤ 总容量的一半。

### 2.3 UB占用计算

```python
def calculate_ub_usage(
    block_size: int,
    num_load_tensors: int,
    num_store_tensors: int,
    num_intermediate_tensors: int,
    dtype_bytes: int = 2,  # BF16 = 2, FP32 = 4
) -> float:
    """
    计算UB占用（KB）
    
    Args:
        block_size: 每个block的元素数量
        num_load_tensors: load的tensor数量
        num_store_tensors: store的tensor数量
        num_intermediate_tensors: 中间tensor数量
        dtype_bytes: 每个元素的字节数
    
    Returns:
        UB占用（KB）
    """
    total_tensors = num_load_tensors + num_store_tensors + num_intermediate_tensors
    ub_bytes = block_size * total_tensors * dtype_bytes
    return ub_bytes / 1024
```

### 2.4 最大处理量计算

```python
def calculate_max_tokens(
    hidden_size: int,
    dtype_bytes: int = 2,
    ub_limit_kb: float = 85,
) -> int:
    """
    计算单次循环最大处理Token数
    
    Args:
        hidden_size: 隐藏层大小
        dtype_bytes: 数据类型字节数
        ub_limit_kb: UB限制（KB）
    
    Returns:
        最大Token数
    """
    ub_limit_bytes = ub_limit_kb * 1024
    token_size = hidden_size * dtype_bytes
    return ub_limit_bytes // token_size
```

---

## 3. 流水线优化

### 3.1 理想流水线

```
时间 →
Scalar: [addr1] [addr2] [addr3] ...
MTE:    [load1] [load2] [load3] ...
Vector:         [comp1] [comp2] [comp3] ...
MTE:            [store1][store2][store3]...
```

### 3.2 破坏流水线的操作

| 操作 | 影响 | 解决方案 |
|------|------|---------|
| `tl.load` with other | 阻止MTE独立执行 | 分离load和where |
| 数据依赖 | 迭代无法并行 | 独立计算偏移 |
| 大量Scalar计算 | Scalar成为瓶颈 | 预计算、向量化 |
| 二维mask | 占用大量UB | 使用insert_slice |

### 3.3 流水线检查

使用msprof检查流水线状态：

```bash
msprof op --output=./profile python test.py
```

检查项：
- MTE和Vector是否有重叠执行区间
- Scalar是否存在长时间独占执行
- 各引擎的利用率是否均衡

---

## 4. 内存访问优化

### 4.1 内存带宽

| 芯片型号 | 内存带宽 |
|---------|---------|
| 910B | ~1.2 TB/s |

### 4.2 访问模式

| 模式 | 效率 | 说明 |
|------|------|------|
| 连续访问 | 高 | 地址连续，带宽利用率高 |
| 跨步访问 | 中 | 固定步长，效率取决于步长 |
| 随机访问 | 低 | 地址不连续，效率低 |

### 4.3 对齐要求

- 推荐32字节对齐
- 可提升MTE搬运效率

```python
# 确保对齐
aligned_offset = (offset // 16) * 16
```

---

## 5. 分核策略

### 5.1 核数获取

```python
import torch_npu

def get_num_cores():
    props = torch_npu.npu.get_device_properties("npu:0")
    return props.multi_processor_count
```

### 5.2 负载均衡

```python
def calculate_work_distribution(total_work: int, num_cores: int) -> List[Tuple[int, int]]:
    """
    计算各核的工作分布
    
    Returns:
        每个核的(start, end)元组列表
    """
    base_work = total_work // num_cores
    extra = total_work % num_cores
    
    distribution = []
    start = 0
    
    for i in range(num_cores):
        work = base_work + (1 if i < extra else 0)
        end = start + work
        distribution.append((start, end))
        start = end
    
    return distribution
```

### 5.3 Grid配置

```python
# 根据数据量选择grid大小
def calculate_grid(num_elements: int, block_size: int, num_cores: int) -> int:
    num_blocks = triton.cdiv(num_elements, block_size)
    
    # 不超过核数的2倍
    return min(num_blocks, num_cores * 2)
```

---

## 6. 数据类型性能

### 6.1 各数据类型性能

| 数据类型 | 相对性能 | 说明 |
|---------|---------|------|
| BF16 | 1.0x | 基准 |
| FP16 | 1.0x | 与BF16相当 |
| FP32 | 0.5x | 计算量翻倍 |
| INT8 | 2.0x | 计算量减半 |

### 6.2 类型转换开销

- GM → UB：无额外开销
- UB内转换：有开销
- 建议：尽量在GM读取时就确定类型

---

## 7. 性能基准

### 7.1 常见算子性能参考

| 算子 | Shape | 耗时（参考） |
|------|-------|------------|
| Softmax | [4096, 4096] | ~0.5 ms |
| LayerNorm | [4096, 4096] | ~0.3 ms |
| MatMul | [4096, 4096] x [4096, 4096] | ~2 ms |
| Flash Attention | [1, 32, 4096, 128] | ~0.8 ms |

### 7.2 性能指标

| 指标 | 目标值 |
|------|--------|
| MTE利用率 | > 80% |
| Vector利用率 | > 80% |
| 内存带宽利用率 | > 70% |
| UB使用量 | < 85 KB |

---

## 8. 常见性能问题

### 8.1 UB溢出

**症状**：性能严重退化或运行时错误。

**解决**：
1. 减小BLOCK_SIZE
2. 减少中间变量
3. 及时store释放UB

### 8.2 流水线不工作

**症状**：MTE和Vector没有并行执行。

**解决**：
1. 避免带other的load
2. 消除数据依赖
3. 检查循环结构

### 8.3 内存带宽不足

**症状**：计算单元空闲等待数据。

**解决**：
1. 优化内存访问模式
2. 增加计算密度
3. 使用Double Buffering

### 8.4 分核不均

**症状**：部分核空闲，部分核过载。

**解决**：
1. 重新计算工作分布
2. 调整grid配置
3. 使用动态负载均衡
