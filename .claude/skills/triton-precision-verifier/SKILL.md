---
name: triton-precision-verifier
description: Triton算子精度验证技能。当需要验证算子正确性、调试精度问题、对比不同实现时使用。触发场景包括：代码生成后需要验证、用户报告精度问题、用户提到"验证"、"测试"、"精度"、"正确性"、出现NaN/Inf输出。Use when verifying operator correctness, debugging precision issues, or comparing implementations.
---

# Triton Precision Verifier

面向 Ascend NPU 的 Triton 算子精度验证技能，提供完整的验证流程和问题诊断。

## 工作流程

```
环境检查 → 测试生成 → 执行验证 → 问题诊断 → 输出报告
```

---

## Step 1: 环境检查

### 1.1 检查 NPU 环境

```python
def check_npu_environment():
    try:
        import torch_npu
        if torch.npu.is_available():
            print(f"NPU可用: {torch.npu.get_device_name(0)}")
            return True
        return False
    except ImportError:
        print("torch_npu 未安装")
        return False
```

| 环境状态 | 处理方式 |
|---------|---------|
| NPU可用 | 执行完整验证流程 |
| NPU不可用 | 提示用户配置环境，跳过验证 |

### 1.2 环境配置命令

```bash
export LD_LIBRARY_PATH=/usr/local/Ascend/driver/lib64/driver:/usr/local/Ascend/driver/lib64/common:/usr/local/Ascend/driver/lib64:$LD_LIBRARY_PATH && source /usr/local/Ascend/ascend-toolkit/set_env.sh
```

---

## Step 2: 知识检索

### 2.1 检索精度问题指南

**文件**: `.claude/data/guides/precision-issues.md`

**检索策略**: 根据验证失败的症状检索对应章节

| 症状 | 检索章节 |
|------|---------|
| 输出包含 NaN/Inf | §3 NaN/Inf问题 |
| 相对误差过大 | §2 精度损失 |
| 特定形状失败 | §5 边界条件问题 |
| Softmax异常 | §6 Softmax特有问题 |
| LayerNorm异常 | §7 LayerNorm特有问题 |

### 2.2 检索精度案例

**目录**: `.claude/data/cases/precision/`

**案例文件**:
- `store_alignment_issue.json` - 存储对齐问题

**使用方式**: 根据错误症状匹配案例，参考解决方案

---

## Step 3: 测试生成

### 3.1 测试数据类型

| 测试类型 | 目的 | 数据生成 |
|---------|------|---------|
| 基础测试 | 验证基本正确性 | `torch.randn(shape, dtype=torch.float16)` |
| 多形状测试 | 验证形状兼容性 | 多种形状组合 |
| 边界测试 | 验证边界条件 | zeros/ones/large/small/mixed |
| 种子测试 | 验证可重复性 | 多随机种子 |

### 3.2 测试代码模板

```python
import torch
import pytest

def test_correctness():
    torch.manual_seed(42)
    
    shapes = [
        (128, 1024),
        (256, 512),
        (1, 1024),
        (1024, 1),
        (1024, 1024),
    ]
    
    for shape in shapes:
        x = torch.randn(shape, device='npu', dtype=torch.float16)
        
        output = my_operator(x)
        expected = torch_reference(x)
        
        assert not torch.isnan(output).any(), f"NaN in output for shape {shape}"
        assert not torch.isinf(output).any(), f"Inf in output for shape {shape}"
        
        max_abs = (output - expected).abs().max().item()
        max_rel = ((output - expected).abs() / expected.abs().clamp(min=1e-6)).max().item()
        
        print(f"Shape {shape}: max_abs={max_abs:.2e}, max_rel={max_rel:.2e}")
        
        assert max_abs < 1e-2, f"Absolute error too large: {max_abs}"
        assert max_rel < 1e-2, f"Relative error too large: {max_rel}"

def test_edge_cases():
    test_cases = [
        ("zeros", torch.zeros(128, 1024, device='npu', dtype=torch.float16)),
        ("ones", torch.ones(128, 1024, device='npu', dtype=torch.float16)),
        ("large", torch.full((128, 1024), 1e4, device='npu', dtype=torch.float16)),
        ("small", torch.full((128, 1024), 1e-4, device='npu', dtype=torch.float16)),
    ]
    
    for name, x in test_cases:
        output = my_operator(x)
        expected = torch_reference(x)
        
        assert not torch.isnan(output).any(), f"NaN in {name} test"
        assert not torch.isinf(output).any(), f"Inf in {name} test"
        print(f"{name}: passed")
```

### 3.3 参考实现映射

| 算子 | PyTorch 参考实现 |
|------|-----------------|
| softmax | `torch.nn.functional.softmax(x, dim=-1)` |
| layernorm | `torch.nn.functional.layer_norm(x, [N], weight, bias, eps)` |
| gelu | `torch.nn.functional.gelu(x)` |
| relu | `torch.nn.functional.relu(x)` |
| matmul | `torch.matmul(a, b)` |

---

## Step 4: 执行验证

### 4.1 验证流程

```python
def verify_operator(kernel_fn, ref_fn, shapes, dtype=torch.float16):
    results = []
    
    for shape in shapes:
        x = torch.randn(shape, dtype=dtype, device='npu')
        
        output = kernel_fn(x)
        expected = ref_fn(x.cpu()).npu()
        
        nan_count = torch.isnan(output).sum().item()
        inf_count = torch.isinf(output).sum().item()
        
        max_abs_error = (output - expected).abs().max().item()
        max_rel_error = ((output - expected).abs() / expected.abs().clamp(min=1e-6)).max().item()
        
        results.append({
            'shape': shape,
            'nan_count': nan_count,
            'inf_count': inf_count,
            'max_abs_error': max_abs_error,
            'max_rel_error': max_rel_error,
            'passed': max_abs_error < 1e-2 and max_rel_error < 1e-2 and nan_count == 0
        })
    
    return results
```

### 4.2 容差标准

| 数据类型 | 绝对容差 (atol) | 相对容差 (rtol) |
|---------|----------------|----------------|
| float16 | 1e-2 | 1e-2 |
| bfloat16 | 1e-2 | 1e-2 |
| float32 | 1e-5 | 1e-5 |

---

## Step 5: 问题诊断

### 5.1 诊断流程

```
验证失败 → 症状分类 → 知识检索 → 根因分析 → 解决方案
```

### 5.2 症状诊断表

| 症状 | 可能原因 | 诊断方法 | 解决方案 |
|------|---------|---------|---------|
| NaN输出 | 除零、负数开方、exp溢出 | 检查中间值范围 | 添加保护、减最大值 |
| Inf输出 | exp溢出、除以极小值 | 检查输入范围 | 添加clamp、epsilon |
| 大误差 | BF16累加、类型转换 | 对比float32结果 | 使用float32累加 |
| 特定形状失败 | 边界条件、mask错误 | 检查边界处理 | 修复mask逻辑 |

### 5.3 常见问题解决方案

**NaN问题**:

```python
# 检查除零
result = a / (b + 1e-10)

# 检查负数开方
result = tl.sqrt(tl.maximum(x, 0.0))

# 检查exp溢出
max_x = tl.max(x, axis=0)
exp_x = tl.exp(x - max_x)
```

**精度损失**:

```python
# 使用float32累加
acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
for k in range(0, K, BLOCK_K):
    a = tl.load(...).to(tl.float32)
    b = tl.load(...).to(tl.float32)
    acc += tl.dot(a, b)
result = acc.to(tl.bfloat16)
```

---

## Step 6: 输出报告

```markdown
# 精度验证报告

## 验证摘要

- 验证状态: 通过/失败
- 测试数量: N
- 通过数量: M
- 失败测试: [列表]

## 误差统计

| 形状 | 最大绝对误差 | 最大相对误差 | NaN数量 | Inf数量 |
|------|------------|------------|--------|--------|
| ... | ... | ... | ... | ... |

## 问题分析

[问题描述和根因分析]

## 解决方案

[建议的修复方案]

## 代码修改建议

```python
# 修复后的代码
```
```

---

## 调试技巧

### 分步验证

```python
# 将复杂计算分解为简单步骤
step1 = compute_step1(input)
print(f"Step1: min={step1.min()}, max={step1.max()}, nan={torch.isnan(step1).any()}")

step2 = compute_step2(step1)
print(f"Step2: min={step2.min()}, max={step2.max()}, nan={torch.isnan(step2).any()}")
```

### 对比参考实现

```python
expected = torch_reference(input)
output = triton_operator(input)

diff = torch.abs(output - expected)
print(f"Max diff: {diff.max()}")
print(f"Mean diff: {diff.mean()}")
print(f"Diff > 0.01: {(diff > 0.01).sum()}")
```

---

## 与其他 Skill 的协作

```
triton-code-generator (代码生成)
        ↓
triton-precision-verifier (精度验证) ← 当前
        ↓ 验证通过
triton-performance-optimizer (性能优化)
```

**验证通过后**，建议用户：
- 使用 `triton-performance-optimizer` 进行性能优化

**验证失败时**：
- 根据诊断结果修复代码
- 重新验证直到通过
