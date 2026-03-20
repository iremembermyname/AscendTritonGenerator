---
name: msprof-profiling
description: 使用msprof工具对Ascend NPU上的Triton算子进行深度性能分析。当需要详细的硬件性能指标、流水线利用率、存储使用分析时使用。触发场景：用户提到"msprof"、"profiling"、"性能分析"、"流水线分析"、"存储使用"、"利用率"、"性能瓶颈定位"、"详细性能报告"。Use when deep profiling with msprof tool is needed for Triton operators on Ascend NPU.
---

# msprof-profiling

使用msprof工具对Ascend NPU上的Triton算子进行深度性能分析，获取详细的硬件性能指标。

## When to Use

This skill is triggered when:

- 用户要求使用msprof工具进行性能分析
- 需要详细的流水线利用率指标（MTE/Vector/Cube）
- 需要分析存储使用情况（UB/L0A/L0B/L0C）
- 需要定位性能瓶颈
- 用户提到"profiling"、"性能瓶颈"、"利用率"

## 前置检查

**重要**: 在执行性能分析前，必须先确认输入内容包含Triton算子定义。

### 如何识别Triton算子

Triton算子通常具有以下特征：

1. **函数装饰器**: 使用 `@triton.jit` 或 `@triton.autotune`
2. **Kernel函数**: 包含 `tl.load`, `tl.store`, `tl.compute` 等Triton API
3. **编译调用**: 通过 `kernel[grid](...)` 方式启动

### 检查步骤

1. 阅读用户提供的代码或文件
2. 搜索以下Triton特征：
   - `@triton.jit` 装饰器
   - `def kernel_name(...):` - kernel函数定义
   - `tl.load`, `tl.store`, `tl.arange` - Triton API调用
   - `kernel[grid](...)` - Kernel启动模式
3. 如果确认是Triton算子，继续执行性能分析流程
4. 如果不是Triton算子，向用户说明当前skill只支持Triton算子分析

## Workflow

```
输入检查 → 环境配置 → 生成测试代码 → msprof采集 → 指标提取 → 瓶颈分析 → 输出报告
```

---

## Step 1: 环境配置

### 1.1 设置环境变量

```bash
export LD_LIBRARY_PATH=/usr/local/Ascend/driver/lib64/driver:/usr/local/Ascend/driver/lib64/common:/usr/local/Ascend/driver/lib64:$LD_LIBRARY_PATH && source /usr/local/Ascend/ascend-toolkit/set_env.sh
```

### 1.2 确认msprof可用

```bash
msprof --version
```

### 1.3 确认设备可用

```bash
python -c "import torch; print(torch.npu.is_available())"
```

---

## Step 2: 生成测试代码

生成测试脚本 `test_triton_profiling.py`：

**注意**: 本测试脚本不用于时间测量，仅用于触发算子执行。实际性能指标全部从msprof采集结果中获取。

```python
import torch
import triton
import triton.language as tl

# 导入用户的Triton算子
from your_module import triton_kernel

def run_profiling():
    # 根据算子需求设置输入参数
    x = torch.randn(BATCH_SIZE, SEQ_LEN, HIDDEN_SIZE, device='npu', dtype=torch.float16)
    weight = torch.randn(HIDDEN_SIZE, HIDDEN_SIZE, device='npu', dtype=torch.float16)

    # Warmup
    for _ in range(WARM_UP):
        _ = triton_kernel(x, weight)

    # Benchmark
    for _ in range(LAUNCH_COUNT):
        _ = triton_kernel(x, weight)

    print("Profiling completed")

if __name__ == "__main__":
    run_profiling()
```

---

## Step 3: msprof性能采集

### 3.1 执行采集命令

```bash
msprof op --output=./profile_triton --application="python test_triton_profiling.py" \
    --aic-metrics=BasicInfo,PipeUtilization,MemoryUB,MemoryL0 \
    --warm-up=20 --launch-count=20
```

### 3.2 采集参数说明

| 参数 | 说明 | 推荐值 |
|------|------|-------|
| `--warm-up` | 预热次数 | 20 |
| `--launch-count` | 采集次数 | 20 |
| `--aic-metrics` | 采集指标 | BasicInfo,PipeUtilization,MemoryUB,MemoryL0 |

### 3.3 可选采集指标

| 指标类别 | 包含内容 | 用途 |
|---------|---------|------|
| BasicInfo | Task Duration, Op Name | 基本性能数据 |
| PipeUtilization | MTE/Vector/Cube利用率 | 流水线分析 |
| MemoryUB | UB使用量 | Vector算子存储分析 |
| MemoryL0 | L0A/L0B/L0C使用量 | Cube算子存储分析 |

---

## Step 4: 指标提取

### 4.1 从msprof结果提取性能数据

**重要**: 所有性能数据必须从msprof采集结果中提取，不要使用打印时间。

#### 基本性能数据提取

1. 找到msprof输出目录中的 `OpBasicInfo_*.csv` 文件
2. 读取Triton kernel的 `Task Duration(us)` 列
3. 如果实现包含多个kernel，需要求和

```bash
# 示例：提取kernel时间
cat profile_triton/*/OpBasicInfo_*.csv | grep -E "Op Name|Task Duration"
```

#### 流水线利用率提取

从 `PipeUtilization_*.csv` 提取：

```bash
cat profile_triton/*/PipeUtilization_*.csv
```

#### 存储使用提取

```bash
# UB使用量
cat profile_triton/*/MemoryUB_*.csv

# L0使用量
cat profile_triton/*/MemoryL0_*.csv
```

---

## Step 5: 瓶颈分析

### 5.1 核函数配置分析

| 指标 | 数值 | 说明 |
|------|------|------|
| BLOCK_SIZE | ... | 是否合理（<65536） |
| num_warps | ... | 线程配置 |
| num_stages | ... | 软件流水阶段数 |
| grid | (...) | 分核策略 |

### 5.2 存储使用分析

| 存储类型 | 使用量 | 约束 | 状态 |
|---------|--------|------|------|
| UB | ... KB | ≤ 85KB | OK/WARNING |
| L0A | ... KB | ≤ 64KB | OK/WARNING |
| L0B | ... KB | ≤ 64KB | OK/WARNING |
| L0C | ... KB | ≤ 128KB | OK/WARNING |

### 5.3 流水线分析

| 指标 | 数值 | 优化目标 |
|------|------|---------|
| MTE1 Utilization | ...% | 与Cube并行 |
| MTE2 Utilization | ...% | > 80% |
| MTE3 Utilization | ...% | > 80% |
| Vector Utilization | ...% | 最大化 |
| Cube Utilization | ...% | 最大化 |

### 5.4 瓶颈定位

根据指标组合判断瓶颈类型：

| 指标组合 | 瓶颈类型 | 优化方向 |
|---------|---------|---------|
| MTE高，Vector低 | 访存瓶颈 | 多Token并行、减少load次数 |
| Vector高，MTE低 | 计算瓶颈 | 优化计算逻辑 |
| L0溢出 | 存储瓶颈 | 减小Block尺寸 |
| UB > 85KB | UB溢出 | 减少变量、启用Double Buffering |
| 两者都不高 | 流水线问题 | 检查other参数、数据依赖 |

### 5.5 优化建议

根据瓶颈分析，提供具体的优化建议：

1. **访存瓶颈建议**:
   - 增加单次加载数据量
   - 启用Double Buffering重叠计算和加载
   - 合并离散访存为连续访存

2. **计算瓶颈建议**:
   - 增加计算密度
   - 使用更高效的Triton原语
   - 调整num_warps和num_stages

3. **存储瓶颈建议**:
   - 减小BLOCK_M/N/K尺寸
   - 分块处理大数据
   - 优化数据布局

4. **流水线瓶颈建议**:
   - 分离tl.load和tl.where操作
   - 预计算mask
   - 减少Scalar计算

---

## Step 6: 输出报告

```markdown
# msprof性能分析报告

## 基本信息
- 算子名称: <op_name>
- 测试形状: <shape>
- 数据类型: <dtype>
- 测试时间: <timestamp>
- 数据来源: msprof采集

---

## 性能总览

### Kernel执行时间 (来自msprof OpBasicInfo)

| Kernel名称 | Op Type | Task Duration (us) |
|-----------|---------|-------------------|
| <kernel1> | <type> | <time1> |
| <kernel2> | <type> | <time2> |
| **总计** | - | **SUM(所有kernel)** |

---

## 核函数配置 (来自msprof BasicInfo)

| 参数 | 值 | 是否合理 |
|------|-----|---------|
| BLOCK_M | ... | ✓/✗ |
| BLOCK_N | ... | ✓/✗ |
| BLOCK_K | ... | ✓/✗ |
| num_warps | ... | ✓/✗ |
| num_stages | ... | ✓/✗ |

---

## 存储使用 (来自msprof MemoryUB/MemoryL0)

| 存储 | 使用量 | 约束 | 状态 |
|------|--------|------|------|
| UB | ... KB | ≤ 85KB | ✅/⚠️ |
| L0A | ... KB | ≤ 64KB | ✅/⚠️ |
| L0B | ... KB | ≤ 64KB | ✅/⚠️ |
| L0C | ... KB | ≤ 128KB | ✅/⚠️ |

---

## 流水线指标 (来自msprof PipeUtilization)

| 指标 | 数值 | 优化目标 | 状态 |
|------|------|---------|------|
| Task Duration | ... us | 最小化 | ... |
| MTE1 Utilization | ...% | 并行 | ... |
| MTE2 Utilization | ...% | > 80% | ... |
| MTE3 Utilization | ...% | > 80% | ... |
| Vector Utilization | ...% | 最大化 | ... |
| Cube Utilization | ...% | 最大化 | ... |

---

## 瓶颈分析

**主要瓶颈**: <瓶颈类型>

**原因分析**:
1. ...
2. ...

---

## 优化建议

1. ...
2. ...
3. ...
```

---

## 与其他组件的协作

```
verify-precision skill (精度验证通过)
        ↓
msprof-profiling skill (深度性能分析) ← 当前
        ↓
triton-expert agent (根据分析结果优化)
```

**分析完成后**: 建议使用 `triton-expert` agent 根据优化建议改进Triton算子

---

## 与 benchmark-comparison 的区别

| 特性 | msprof-profiling | benchmark-comparison |
|------|-----------------|---------------------|
| 工具 | msprof | time.time() + synchronize |
| 指标 | 详细硬件指标 | 仅时间对比 |
| 用途 | 深度分析、瓶颈定位 | 快速对比、回归测试 |
| 输出 | 完整性能报告 | 简单加速比 |

**选择建议**:
- 需要详细分析瓶颈 → 使用 `msprof-profiling`
- 只需快速对比性能 → 使用 `benchmark-comparison`
