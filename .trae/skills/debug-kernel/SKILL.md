---
name: debug-kernel
description: Triton算子调试流程。当遇到NaN/Inf输出、精度误差、运行时错误时使用。触发场景：用户报告算子问题、用户提到"调试"、"NaN"、"Inf"、"错误"。Use when debugging kernel issues like NaN/Inf, precision errors, or runtime errors.
---

# Debug Kernel

面向Ascend NPU的Triton算子调试流程。

## When to Use

This skill is triggered when:

- 用户报告算子输出NaN/Inf
- 用户报告精度误差过大
- 用户遇到运行时错误
- 用户提到"调试"、"问题"、"错误"

## Workflow

```
问题分类 → 环境配置 → 诊断工具 → 根因定位 → 解决方案
```

---

## Step 1: 问题分类

### 1.1 问题类型识别

| 问题类型 | 症状 | 优先级 |
|---------|------|--------|
| NaN输出 | 输出包含NaN值 | 高 |
| Inf输出 | 输出包含Inf值 | 高 |
| 精度误差 | 输出与预期差异大 | 中 |
| 运行时错误 | 编译或执行失败 | 高 |
| 性能问题 | 执行时间过长 | 低 |

### 1.2 快速诊断问题

```python
def quick_diagnosis(output, expected=None):
    issues = []
    
    if torch.isnan(output).any():
        issues.append("NaN detected in output")
    
    if torch.isinf(output).any():
        issues.append("Inf detected in output")
    
    if expected is not None:
        max_diff = (output - expected).abs().max().item()
        if max_diff > 0.1:
            issues.append(f"Large error: max_diff={max_diff}")
    
    return issues
```

---

## Step 2: 环境配置

### 2.1 调试环境变量

```bash
# 启用详细日志
export TRITON_INTERPRET=1  # 使用解释器模式
```

### 2.2 NPU环境

```bash
export LD_LIBRARY_PATH=/usr/local/Ascend/driver/lib64/driver:/usr/local/Ascend/driver/lib64/common:/usr/local/Ascend/driver/lib64:$LD_LIBRARY_PATH && source /usr/local/Ascend/ascend-toolkit/set_env.sh
```

---

## Step 3: 诊断工具

### 3.1 分步验证

将复杂计算分解为简单步骤：

```python
def debug_step_by_step(x):
    print(f"Input: min={x.min()}, max={x.max()}")
    
    step1 = compute_step1(x)
    print(f"Step1: min={step1.min()}, max={step1.max()}, nan={torch.isnan(step1).any()}")
    
    step2 = compute_step2(step1)
    print(f"Step2: min={step2.min()}, max={step2.max()}, nan={torch.isnan(step2).any()}")
    
    return step2
```

### 3.2 边界条件检查

```python
def check_edge_cases():
    # 零输入
    x = torch.zeros(128, 1024, device='npu')
    output = my_operator(x)
    
    # 极大值
    x = torch.full((128, 1024), 1e4, device='npu')
    output = my_operator(x)
    
    # 极小值
    x = torch.full((128, 1024), 1e-4, device='npu')
    output = my_operator(x)
```

---

## Step 4: 根因定位

### 4.1 知识检索

**文件**: `.trae/data/guides/debugging-guide.md`

| 问题类型 | 检索章节 |
|---------|---------|
| NaN问题 | `debugging-guide.md` §NaN诊断 |
| 编译错误 | `debugging-guide.md` §常见问题排查 |
| 运行时错误 | `debugging-guide.md` §常见问题排查 |

### 4.2 常见根因表

| 问题 | 常见根因 | 定位方法 |
|------|---------|---------|
| NaN | 除零、负数开方、exp溢出 | 检查中间值范围 |
| Inf | exp溢出、除以极小值 | 检查输入范围 |
| 精度误差 | BF16累加、类型转换 | 对比float32结果 |
| 编译错误 | constexpr错误、类型不匹配 | 检查kernel签名 |

---

## Step 5: 解决方案

### 5.1 NaN问题解决

```python
# 问题：除零
result = a / b  # b可能为0

# 解决：添加epsilon
result = a / (b + 1e-10)

# 问题：负数开方
result = tl.sqrt(x)  # x可能为负

# 解决：添加下界
result = tl.sqrt(tl.maximum(x, 0.0))

# 问题：exp溢出
result = tl.exp(x)  # x可能很大

# 解决：减最大值
max_x = tl.max(x, axis=0)
result = tl.exp(x - max_x)
```

### 5.2 精度问题解决

```python
# 问题：BF16累加精度损失
acc = tl.zeros([BLOCK], dtype=tl.bfloat16)

# 解决：使用float32累加
acc = tl.zeros([BLOCK], dtype=tl.float32)
# 最后转回目标类型
result = acc.to(tl.bfloat16)
```

### 5.3 案例检索

**目录**: `.trae/data/cases/precision/`

---

## Step 6: 输出报告

```markdown
# 调试报告

## 问题描述
- 问题类型: NaN/Inf/精度误差/运行时错误
- 症状: ...

## 诊断过程
1. 问题分类: ...
2. 环境检查: ...
3. 诊断工具: ...
4. 根因定位: ...

## 根因分析
- 根本原因: ...
- 代码位置: ...

## 解决方案
```python
# 修复代码
```

## 验证
- 修复后测试结果: ...

## 参考案例
- `data/cases/precision/xxx.json`
```

---

## 与其他组件的协作

```
用户报告问题
        ↓
debug-kernel skill (调试流程) ← 当前
        ↓ 需要深入分析
triton-expert agent (专家诊断和修复)
        ↓
verify-precision skill (验证修复)
```

**调试完成后**：建议使用 `verify-precision` 验证修复效果
