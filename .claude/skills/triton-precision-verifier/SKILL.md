---
name: triton-precision-verifier
description: Triton算子精度验证指导。当需要验证算子正确性、调试精度问题、或对比不同实现时使用此skill。Use when verifying operator correctness, debugging precision issues, or comparing implementations.
---

# Triton Precision Verifier

Triton精度验证技能，提供step-by-step的精度验证指导。

## When to Use

This skill is triggered when:
- User asks "verify the precision of this code"
- User wants to debug numerical issues
- User mentions testing or validation
- After code generation, before optimization

## Knowledge Retrieval

执行任务前，检索相关知识：
1. `@.claude/data/guides/precision-issues.md` - 常见精度问题
2. `@.claude/data/cases/precision/` - 精度问题案例

## Step-by-Step Guide

### Step 1: Environment Check

检查NPU环境是否可用：

```python
def check_npu_environment():
    try:
        import torch_npu
        return torch.npu.is_available()
    except ImportError:
        return False
```

- **有NPU**：执行完整验证流程
- **无NPU**：跳过验证，提示用户

### Step 2: Generate Test Data

| 测试类型 | 目的 | 数据生成方式 |
|---------|------|-------------|
| 基础测试 | 验证基本正确性 | 随机数据 |
| 多形状测试 | 验证形状兼容性 | 多种形状 |
| 边界测试 | 验证边界条件 | zeros/ones/large/small/mixed |
| 种子测试 | 验证可重复性 | 多随机种子 |

### Step 3: Execute Tests

```python
import torch

def verify_operator(kernel_fn, ref_fn, shapes, dtype=torch.float16):
    for shape in shapes:
        x = torch.randn(shape, dtype=dtype, device='npu')
        output = kernel_fn(x)
        expected = ref_fn(x)
        
        # 检查NaN/Inf
        assert not torch.isnan(output).any(), "Output contains NaN"
        assert not torch.isinf(output).any(), "Output contains Inf"
        
        # 计算误差
        max_abs_error = (output - expected).abs().max().item()
        max_rel_error = ((output - expected).abs() / expected.abs().clamp(min=1e-6)).max().item()
        
        print(f"Shape {shape}: max_abs={max_abs_error:.2e}, max_rel={max_rel_error:.2e}")
```

### Step 4: Analyze Issues

验证失败时，进行诊断：

- **NaN问题**：检查除零、负数开方、数值溢出
- **大误差**：检查数值稳定性、中间精度
- **特定形状失败**：检查边界条件处理

### Step 5: Generate Report

```markdown
# 精度验证报告

## 验证摘要
- 验证状态: 通过/失败
- 测试数量: N
- 通过数量: M

## 误差统计
| 指标 | 最大值 | 平均值 |
|------|--------|--------|
| 绝对误差 | X.XXe-XX | X.XXe-XX |
| 相对误差 | X.XXe-XX | X.XXe-XX |

## 问题分析
[问题描述和建议]
```

## Key Requirements

1. **确定性**：相同输入产生相同输出
2. **容差合理**：rtol=1e-3, atol=1e-3 for float16
3. **全面覆盖**：多种形状、边界情况
4. **可重复**：固定随机种子

## Common Mistakes

- ❌ 只测试单一形状
- ❌ 忽略边界情况
- ❌ 容差设置过严
- ❌ 不检查NaN/Inf

## Reference Implementations

| Operator | PyTorch Reference |
|----------|-------------------|
| softmax | `torch.nn.functional.softmax` |
| layernorm | `torch.nn.functional.layer_norm` |
| gelu | `torch.nn.functional.gelu` |
| relu | `torch.nn.functional.relu` |
