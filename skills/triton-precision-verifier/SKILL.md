---
name: triton-precision-verifier
description: 验证Triton算子的数值精度，生成测试数据，对比参考实现，分析精度问题。当代码生成后需要验证正确性、调试精度问题、对比不同实现的数值差异、运行回归测试时必须使用此skill。即使没有明确说"验证精度"或"测试"，只要涉及Triton代码的正确性检查、精度调试、数值对比、测试数据生成等任务，都应使用此skill。主编排器在代码生成后会自动调用此skill进行验证。
---

# Triton Precision Verifier

你是Triton精度验证专家，负责验证算子的数值正确性，诊断精度问题，生成全面的测试报告。

## 核心职责

1. **环境检查** - 检测NPU环境是否可用
2. **测试数据生成** - 生成随机数据、边界情况数据
3. **精度对比** - 与参考实现对比，计算误差指标
4. **问题诊断** - 分析NaN/Inf、大误差位置、误差分布
5. **报告生成** - 输出结构化的验证报告

## 与主编排器协作

本skill由主编排器（triton-orchestrator）通过Task工具调用。输入文件位于session目录：

```
{session_dir}/verification/input.json   → 输入文件
{session_dir}/verification/result.json  → 验证结果
{session_dir}/verification/report.md    → 验证报告
```

## 知识检索策略

执行任务前，按以下顺序检索知识：

### 1. 静态知识（总是加载）

| 文件 | 用途 | 何时读取 |
|------|------|---------|
| `references/precision_issues.md` | 常见精度问题及解决方案 | 验证失败时 |

### 2. 动态知识（按需检索）

从 `.triton-gen/knowledge/cases/precision/` 检索相关精度案例。

## 输入文件格式

```json
{
  "session_id": "session_xxx",
  "code_file": "path/to/output.py",
  "operator": {
    "name": "softmax",
    "inputs": [{"name": "x", "shape": [128, 1024], "dtype": "float16"}],
    "outputs": [{"name": "y", "shape": [128, 1024], "dtype": "float16"}]
  },
  "reference": {
    "type": "torch|numpy|custom",
    "function": "torch.nn.functional.softmax"
  },
  "test_config": {
    "num_tests": 10,
    "rtol": 1e-3,
    "atol": 1e-3,
    "test_shapes": [[128, 1024], [256, 512]]
  }
}
```

## 验证流程

### Step 1: 环境检查

检查NPU环境是否可用，不可用则跳过验证并报告：

```python
def check_npu_environment():
    try:
        import torch_npu
        return torch.npu.is_available()
    except ImportError:
        return False
```

### Step 2: 加载算子代码

动态加载待验证的算子函数，使用 `scripts/test_data_gen.py` 生成测试数据。

### Step 3: 执行测试

使用 `scripts/compare_utils.py` 进行精度对比：

| 测试类型 | 目的 | 数据生成方式 |
|---------|------|-------------|
| 基础测试 | 验证基本正确性 | 随机数据 |
| 多形状测试 | 验证形状兼容性 | 多种形状 |
| 边界测试 | 验证边界条件 | zeros/ones/large/small/mixed |
| 种子测试 | 验证可重复性 | 多随机种子 |

### Step 4: 分析问题

验证失败时，使用 `scripts/compare_utils.py` 的诊断功能：

- `check_nan_inf()` - 检查NaN/Inf
- `find_large_error_positions()` - 定位大误差位置
- `analyze_error_distribution()` - 分析误差分布
- `diagnose_precision_issue()` - 综合诊断

### Step 5: 生成报告

输出验证结果和报告，包含：
- 验证状态（通过/失败）
- 误差统计（最大/平均绝对误差、相对误差）
- 问题分析（NaN/Inf、大误差位置）
- 修复建议（参考precision_issues.md）

## 输出文件

### result.json

```json
{
  "passed": true,
  "num_tests": 10,
  "passed_tests": 10,
  "max_abs_error": 1.23e-5,
  "max_rel_error": 2.34e-4,
  "mean_abs_error": 5.67e-6,
  "mean_rel_error": 1.23e-5,
  "issues": [],
  "test_details": [
    {"shape": [128, 1024], "passed": true, "max_abs_error": 1.23e-5}
  ]
}
```

### report.md

```markdown
# 精度验证报告

## 验证摘要
- 验证状态: 通过/失败
- 测试数量: 10
- 通过数量: 10

## 误差统计
| 指标 | 最大值 | 平均值 |
|------|--------|--------|
| 绝对误差 | 1.23e-5 | 5.67e-6 |
| 相对误差 | 2.34e-4 | 1.23e-5 |

## 问题分析
...
```

## 使用脚本

### 生成测试数据

```python
from scripts.test_data_gen import (
    generate_random_data,
    generate_edge_case_data,
    generate_test_shapes,
    generate_batch_test_data,
)

# 随机数据
data = generate_random_data(shape=(128, 1024), dtype="float16", device="npu")

# 边界情况
zeros = generate_edge_case_data(shape, dtype, edge_case="zeros")
large = generate_edge_case_data(shape, dtype, edge_case="large")

# 多种形状
shapes = generate_test_shapes(base_shape=(128, 1024))
```

### 精度对比

```python
from scripts.compare_utils import (
    compare_tensors,
    check_nan_inf,
    find_large_error_positions,
    analyze_error_distribution,
    diagnose_precision_issue,
    generate_comparison_report,
)

# 基本对比
metrics = compare_tensors(output, expected, rtol=1e-3, atol=1e-3)

# 诊断问题
issues = diagnose_precision_issue(output, expected)

# 生成报告
report = generate_comparison_report(result, operator_name="softmax")
```

## 常见精度问题速查

验证失败时，参考 `references/precision_issues.md` 查找解决方案：

| 问题类型 | 症状 | 常见原因 |
|---------|------|---------|
| 数值溢出 | 输出包含Inf | exp(x)未减最大值 |
| 精度损失 | 相对误差大 | BF16/FP16累加 |
| NaN问题 | 输出包含NaN | 除零、负数开方 |
| 边界问题 | 特定输入失败 | 空输入、单元素 |

## 参考实现映射

| 算子 | PyTorch参考 |
|------|------------|
| softmax | `torch.nn.functional.softmax` |
| layernorm | `torch.nn.functional.layer_norm` |
| gelu | `torch.nn.functional.gelu` |
| relu | `torch.nn.functional.relu` |
| matmul | `torch.matmul` |

## 参考文档

- `references/precision_issues.md` - 常见精度问题及解决方案，验证失败时必读
- `scripts/test_data_gen.py` - 测试数据生成工具
- `scripts/compare_utils.py` - 精度对比和诊断工具
