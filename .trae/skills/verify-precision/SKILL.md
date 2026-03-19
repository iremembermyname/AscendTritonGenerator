---
name: verify-precision
description: Triton算子精度验证流程。当需要验证算子正确性、调试精度问题时使用。触发场景：代码生成后验证、用户报告精度问题、用户提到"验证精度"、"测试正确性"。Use when verifying operator correctness or debugging precision issues.
---

# Verify Precision

面向Ascend NPU的Triton算子精度验证流程。

## When to Use

This skill is triggered when:

- 用户要求验证算子精度
- 代码生成/转换后需要验证
- 用户报告精度问题（NaN/Inf、误差过大）
- 用户提到"验证"、"测试"、"精度"、"正确性"

## Workflow

```
环境检查 → 测试生成 → 执行验证 → 问题诊断 → 输出报告
```

---

## Step 1: 环境检查

### 1.1 检查NPU环境

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

**文件**: `.trae/data/guides/precision-guide.md`

| 症状 | 检索章节 |
|------|---------|
| 输出包含NaN/Inf | §3 NaN/Inf问题 |
| 相对误差过大 | §2 精度损失 |
| 特定形状失败 | §5 边界条件问题 |

### 2.2 检索精度案例

**目录**: `.trae/data/cases/precision/`

---

## Step 3: 测试生成

### 3.1 测试数据类型

| 测试类型 | 目的 | 数据生成 |
|---------|------|---------|
| 基础测试 | 验证基本正确性 | `torch.randn(shape, dtype=torch.float16)` |
| 多形状测试 | 验证形状兼容性 | 多种形状组合 |
| 边界测试 | 验证边界条件 | zeros/ones/large/small |

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
    ]
    
    for shape in shapes:
        x = torch.randn(shape, device='npu', dtype=torch.float16)
        
        output = my_operator(x)
        expected = torch_reference(x)
        
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()
        
        max_abs = (output - expected).abs().max().item()
        max_rel = ((output - expected).abs() / expected.abs().clamp(min=1e-6)).max().item()
        
        assert max_abs < 1e-2
        assert max_rel < 1e-2
```

### 3.3 参考实现映射

| 算子 | PyTorch参考实现 |
|------|----------------|
| softmax | `torch.nn.functional.softmax(x, dim=-1)` |
| layernorm | `torch.nn.functional.layer_norm(x, [N], weight, bias, eps)` |
| gelu | `torch.nn.functional.gelu(x)` |

---

## Step 4: 执行验证

### 4.1 容差标准

| 数据类型 | 绝对容差(atol) | 相对容差(rtol) |
|---------|---------------|---------------|
| float16 | 1e-3 | 1e-3 |
| bfloat16 | 1e-3 | 1e-3 |
| float32 | 1e-4 | 1e-4 |

---

## Step 5: 问题诊断

### 5.1 症状诊断表

| 症状 | 可能原因 | 解决方案 |
|------|---------|---------|
| NaN输出 | 除零、负数开方、exp溢出 | 添加保护、减最大值 |
| Inf输出 | exp溢出、除以极小值 | 添加clamp、epsilon |
| 大误差 | BF16累加、类型转换 | 使用float32累加 |

### 5.2 常见问题解决方案

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
```

---

## Step 6: 输出报告

```markdown
# 精度验证报告

## 验证摘要
- 验证状态: 通过/失败
- 测试数量: N
- 通过数量: M

## 误差统计
| 形状 | 最大绝对误差 | 最大相对误差 | NaN数量 | Inf数量 |
|------|------------|------------|--------|--------|
| ... | ... | ... | ... | ... |

## 问题分析
[问题描述和根因分析]

## 解决方案
[建议的修复方案]
```

---

## 与其他组件的协作

```
triton-expert agent (代码生成/转换)
        ↓
verify-precision skill (精度验证) ← 当前
        ↓ 验证通过
profile-performance skill (性能分析)
```

**验证通过后**：建议使用 `profile-performance` 分析性能

**验证失败时**：根据诊断结果修复代码，重新验证
