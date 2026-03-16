---
name: triton-performance-optimizer
description: 分析和优化Triton算子性能，专注于Ascend NPU平台。当用户需要提升Triton算子性能、分析性能瓶颈、优化内存访问模式、调整block配置、解决UB溢出问题、或应用Ascend特定优化技术时使用此skill。触发场景包括：(1)用户明确要求"优化性能"、"加速算子"、"提升吞吐量"；(2)精度验证通过后需要性能优化；(3)性能测试不达标；(4)出现UB溢出、流水线不工作等性能问题；(5)用户提到msprof、性能分析、benchmark等关键词。即使用户没有明确说"优化"，只要涉及性能相关话题，都应考虑使用此skill。
---

# Triton Performance Optimizer

分析和优化Triton算子性能，专注于Ascend NPU的性能调优。

## 核心职责

1. **性能分析** - 运行benchmark，识别性能瓶颈
2. **优化建议** - 根据分析结果提供具体优化方案
3. **代码优化** - 生成优化后的Triton代码
4. **性能验证** - 对比优化前后性能，生成报告

## 被调用方式

本skill由主编排器(triton-orchestrator)通过Task工具调用，输入文件位于session目录：

```
{session_dir}/optimization/
├── input.json      # 优化输入
├── optimized.py    # 输出：优化后代码
└── report.md       # 输出：优化报告
```

## 知识检索

执行任务前，按以下顺序检索知识：

### 1. 静态知识（总是加载）

- **`references/ascend_performance.md`** - Ascend硬件架构、UB容量、流水线机制
  - 读取时机：任务开始时，了解硬件约束
  - 关键内容：Vector核数、UB容量限制、Double Buffering机制

- **`references/optimization_tips.md`** - 具体优化技巧和代码模式
  - 读取时机：确定优化方向后，查找对应优化技术
  - 关键内容：内存访问优化、UB容量优化、流水线优化

### 2. 动态知识（按需检索）

- **`.triton-gen/knowledge/cases/optimization/`** - 历史优化案例
  - 检索时机：遇到类似优化场景时
  - 用途：参考成功案例的优化策略

- **`.triton-gen/knowledge/rules/performance_rules.json`** - 性能规则
  - 检索时机：验证优化是否符合规则
  - 用途：确保优化不违反硬件约束

## 输入文件格式

```json
{
  "session_id": "session_xxx",
  "code_file": "../code/output.py",
  "operator": {
    "name": "softmax",
    "inputs": [{"name": "x", "shape": [4096, 4096], "dtype": "bfloat16"}],
    "outputs": [{"name": "y", "shape": [4096, 4096], "dtype": "bfloat16"}]
  },
  "performance_target": {
    "target_time_ms": 0.5,
    "current_time_ms": 2.0
  },
  "constraints": {
    "target_arch": "ascend910b2",
    "max_block_size": 1024
  },
  "has_npu_env": true
}
```

## 输出文件

### optimized.py
优化后的Triton代码，保持与原代码相同的接口。

### report.md
优化报告，包含：
- 优化摘要（前后对比、加速比）
- 应用的优化技术
- 性能指标对比
- 关键修改说明
- 后续建议

## 优化流程

### Phase 1: 性能分析

1. **读取输入文件**：获取代码、算子信息、性能目标
2. **检索硬件知识**：读取 `references/ascend_performance.md`
3. **运行性能测试**（如有NPU环境）：
   - 使用 `scripts/profiler.py` 的 `benchmark_kernel` 函数
   - 获取基准性能数据
4. **分析性能瓶颈**：
   - 如有msprof数据，解析分析
   - 否则基于代码静态分析

### Phase 2: 确定优化方向

根据瓶颈类型选择优化策略：

| 瓶颈类型 | 识别特征 | 优化策略 |
|---------|---------|---------|
| UB溢出 | UB > 85KB 或运行时错误 | 减小BLOCK_SIZE，减少中间变量 |
| MTE瓶颈 | 内存带宽利用率低 | 优化内存访问，合并load |
| Vector瓶颈 | 计算单元利用率低 | 增加计算密度，减少冗余计算 |
| Scalar瓶颈 | 控制流开销大 | 预计算，移出循环 |
| 流水线问题 | MTE/Vector无并行 | 避免带other的load，消除数据依赖 |
| 分核不均 | 部分核空闲 | 优化grid配置 |

### Phase 3: 应用优化

1. **检索优化技巧**：读取 `references/optimization_tips.md` 对应章节
2. **检索历史案例**：从 `.triton-gen/knowledge/cases/optimization/` 检索相似案例
3. **应用优化技术**：
   - 一次应用一个优化
   - 记录每次修改
4. **生成优化后代码**：写入 `optimized.py`

### Phase 4: 验证与报告

1. **验证精度**：确保优化不改变计算结果
2. **验证性能**：运行benchmark对比
3. **生成报告**：写入 `report.md`

## 使用Profiler脚本

`scripts/profiler.py` 提供以下工具函数：

```python
from scripts.profiler import (
    benchmark_kernel,      # 运行性能测试
    estimate_ub_usage,     # 估算UB占用
    calculate_optimal_block_size,  # 计算最优block大小
    analyze_performance,   # 分析性能瓶颈
    generate_performance_report,   # 生成报告
)
```

### 示例：估算UB占用

```python
ub_kb = estimate_ub_usage(
    block_size=1024,
    num_inputs=2,
    num_outputs=1,
    num_intermediates=3,
    dtype_bytes=2  # BF16
)
# 如果 ub_kb > 85，需要减小block_size或减少中间变量
```

### 示例：计算最优block大小

```python
optimal_block = calculate_optimal_block_size(
    ub_limit_kb=85,
    num_inputs=2,
    num_outputs=1,
    num_intermediates=2,
    dtype_bytes=2
)
```

## 关键优化技术速查

### 1. UB容量优化
- 单次循环UB占用 ≤ 85KB
- 减少同时存活的中间变量
- 及时store释放UB空间

### 2. 流水线优化
- 避免带other参数的tl.load
- 分离load和where操作
- 确保循环迭代可独立执行

### 3. 内存访问优化
- 优先连续内存访问
- 合并对同一地址的多次load
- 使用block pointer进行规则访问

### 4. 分核优化
- 均匀分配工作量到各核
- grid大小不超过核数的2倍
- 避免过度分核

详细技术说明见 `references/optimization_tips.md`。

## 重试策略

- **最大优化尝试次数**：3次
- **每次尝试后**：验证精度，确保不引入错误
- **性能未达标时**：
  - 如果已尝试所有可行优化 → 询问用户是否接受当前性能
  - 如果还有优化空间 → 继续尝试
- **同类问题连续出现**：2次后询问用户

## 错误处理

| 错误类型 | 处理方式 |
|---------|---------|
| 精度下降 | 回滚优化，尝试其他策略 |
| 编译失败 | 分析错误，调整优化 |
| 运行时错误 | 检查硬件约束，调整配置 |
| 性能退化 | 回滚优化，分析原因 |

## 参考文档

- `references/ascend_performance.md` - Ascend硬件架构和性能特性
- `references/optimization_tips.md` - 详细优化技巧和代码示例
- `scripts/profiler.py` - 性能分析工具函数
