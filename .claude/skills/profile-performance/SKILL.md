---
name: profile-performance
description: Triton算子性能分析流程。当需要分析算子性能瓶颈、解读性能指标时使用。触发场景：优化前分析、用户报告性能问题、用户提到"分析性能"、"性能瓶颈"。Use when analyzing performance bottlenecks or interpreting performance metrics.
---

# Profile Performance

面向Ascend NPU的Triton算子性能分析流程。

## When to Use

This skill is triggered when:

- 用户要求分析算子性能
- 优化前需要了解瓶颈
- 用户报告性能问题
- 用户提到"性能"、"瓶颈"、"耗时"

## Workflow

```
环境配置 → msprof采集 → 指标解读 → 瓶颈定位 → 输出报告
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

---

## Step 2: msprof采集

### 2.1 性能采集命令

```bash
msprof op --output=./profile --kernel-name="<kernel_name>" --warm-up=20 --launch-count=20 python test_perf.py
```

### 2.2 测试代码模板

```python
import torch
import time

def test_performance():
    x = torch.randn(128, 1024, device='npu', dtype=torch.float16)
    
    # Warmup
    for _ in range(20):
        _ = my_operator(x)
    
    # Benchmark
    torch.npu.synchronize()
    start = time.time()
    for _ in range(20):
        _ = my_operator(x)
    torch.npu.synchronize()
    end = time.time()
    
    avg_time_us = (end - start) / 20 * 1e6
    print(f"Average time: {avg_time_us:.2f} us")

if __name__ == "__main__":
    test_performance()
```

---

## Step 3: 指标解读

### 3.1 关键指标

| 指标 | 说明 | 优化目标 |
|------|------|---------|
| Task Duration | 总执行时间 | 最小化 |
| MTE Utilization | MTE利用率 | 与Vector并行 |
| Vector Utilization | Vector利用率 | 最大化 |
| UB Usage | UB使用量 | ≤ 85KB |

### 3.2 指标解读表

| 指标组合 | 瓶颈类型 | 优化方向 |
|---------|---------|---------|
| MTE高，Vector低 | 访存瓶颈 | 多Token并行、减少load次数 |
| Vector高，MTE低 | 计算瓶颈 | 优化计算逻辑 |
| 两者都不高 | 流水线问题 | 检查other参数、数据依赖 |
| UB > 85KB | UB溢出 | 减少变量、调整Block |

---

## Step 4: 瓶颈定位

### 4.1 知识检索

**文件**: `.claude/data/guides/optimization-guide.md`

| 瓶颈类型 | 检索章节 |
|---------|---------|
| 访存瓶颈 | §1 内存访问优化 |
| UB溢出 | §2 UB容量优化 |
| 流水线问题 | §3 流水线优化 |
| 分核问题 | §5 分核优化 |

### 4.2 案例检索

**目录**: `.claude/data/cases/optimization/`

| 案例文件 | 适用场景 |
|---------|---------|
| `matmul_tuning.json` | 矩阵乘法 |
| `discrete_memory_access.json` | 离散访存 |
| `ub_overflow_handling.json` | UB溢出 |

---

## Step 5: 输出报告

```markdown
# 性能分析报告

## 基本信息
- 算子名称: <op_name>
- 测试形状: <shape>
- 数据类型: <dtype>

## 性能指标
| 指标 | 数值 | 评估 |
|------|------|------|
| Task Duration | ... us | ... |
| MTE Utilization | ...% | ... |
| Vector Utilization | ...% | ... |
| UB Usage | ... KB | ... |

## 瓶颈分析
- 主要瓶颈: ...
- 原因: ...

## 优化建议
1. ...
2. ...

## 参考案例
- `data/cases/optimization/xxx.json`
```

---

## 与其他组件的协作

```
verify-precision skill (精度验证通过)
        ↓
profile-performance skill (性能分析) ← 当前
        ↓
triton-expert agent (根据分析结果优化)
```

**分析完成后**：建议使用 `triton-expert` agent 进行优化
