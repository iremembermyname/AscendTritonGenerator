---
name: triton-code-generator
description: Triton算子代码生成技能。当用户需要实现新的Triton kernel、将算子需求转化为代码、或参考模板生成算子时使用。触发场景包括：用户描述算子功能需求、用户要求实现特定算子(如softmax/layernorm/gelu等)、用户需要CUDA到Ascend的代码转换、用户提到"实现"、"生成"、"编写"算子代码。Use when user wants to generate Triton code, implement operators, or convert CUDA code.
---

# Triton Code Generator

面向 Ascend NPU 的 Triton 算子代码生成技能，提供从需求到代码的完整生成流程。

## 工作流程

```
需求分析 → 知识检索 → 代码生成 → 代码检查 → 输出代码
```

---

## Step 1: 需求分析

理解用户需求，确定：

| 分析项 | 说明 |
|-------|------|
| 算子类型 | element-wise / reduction / matmul / attention / custom |
| 输入输出 | 形状、数据类型、内存布局 |
| 计算逻辑 | 数学公式、计算流程 |
| 性能要求 | 是否有特定性能目标 |

---

## Step 2: 知识检索

根据需求，按优先级检索知识库：

### 2.1 检索代码模板

**文件**: `.claude/data/templates/code-templates.md`

**检索策略**:
- 算子类型匹配：根据算子类型查找对应模板
- 模式匹配：element-wise → 向量加法/乘法模板；reduction → 求和/最大值模板

**模板映射表**:

| 算子类型 | 模板位置 | 关键技术点 |
|---------|---------|-----------|
| element-wise | §1 基础模板 | 连续访问、mask处理 |
| reduction | §2 规约模板 | 原子操作、分块规约 |
| softmax | §3 Softmax模板 | 数值稳定性、三遍扫描 |
| layernorm | §4 LayerNorm模板 | 均值方差计算、归一化 |
| matmul | §5 矩阵乘法模板 | 分块计算、累加器 |
| attention | §6 Flash Attention模板 | 在线softmax、分块计算 |

### 2.2 检索语法参考

**文件**: `.claude/data/syntax/triton-syntax.md`

**检索策略**:
- 按需检索：根据代码中使用的 API 查找对应语法
- 重点章节：内存操作(§3)、规约操作(§6)、Ascend注意事项(§10)

### 2.3 检索 Ascend 扩展 API

**文件**: `.claude/data/syntax/ascend-extensions.md`

**检索策略**:
- 优化需求：多Token并行(§2.2)、Double Buffering(§2.1)
- 特定操作：insert_slice(§3.4)、extract_slice(§3.5)、tl.gather(§3.6)

### 2.4 检索硬件约束

**文件**: `.claude/rules/ascend-hardware.md`

**关键约束**:
- UB容量：单次循环 ≤ 85KB
- Block大小：≤ 1024
- 核数：108 个 Vector 核

---

## Step 3: 代码生成

### 3.1 Kernel 结构模板

```python
import torch
import triton
import triton.language as tl

@triton.jit
def kernel_name(
    input_ptrs,
    output_ptrs,
    shape_params,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    x = tl.load(input_ptr + offsets, mask=mask)
    y = compute(x)
    tl.store(output_ptr + offsets, y, mask=mask)
```

### 3.2 Host 函数模板

```python
def operator_name(x: torch.Tensor) -> torch.Tensor:
    output = torch.empty_like(x)
    grid = lambda meta: (triton.cdiv(x.numel(), meta['BLOCK_SIZE']),)
    kernel_name[grid](x, output, x.numel(), BLOCK_SIZE=256)
    return output
```

### 3.3 生成检查清单

生成代码时必须确保：

- [ ] Block大小 ≤ 1024
- [ ] 使用 `tl.constexpr` 声明编译期常量
- [ ] 所有 load 操作带 mask（处理边界）
- [ ] 避免使用带 `other` 参数的 `tl.load`
- [ ] 累加操作使用 `float32`
- [ ] 数值稳定性处理（如 softmax 减最大值）

---

## Step 4: 代码检查

生成代码后，对照规则文件检查：

**文件**: `.claude/rules/triton-code.md`

| 检查项 | 规则 | 修复方法 |
|-------|------|---------|
| Block大小 | ≤ 1024 | 调整 BLOCK_SIZE |
| 数值稳定性 | exp前减最大值 | 添加 `x - tl.max(x)` |
| 精度 | 累加用 float32 | `acc = tl.zeros(..., dtype=tl.float32)` |
| 内存访问 | 连续访问 | 使用 `tl.arange` 生成连续偏移 |

---

## Step 5: 输出代码

输出格式：

```markdown
## 生成的代码

### Kernel 实现

```python
# 代码内容
```

### Host 函数

```python
# 代码内容
```

### 测试代码

```python
# 基础测试代码
```

## 设计说明

- **Block大小**: XXX，原因：...
- **内存访问模式**: XXX
- **数值稳定性处理**: XXX

## 后续建议

1. 运行精度验证（使用 triton-precision-verifier）
2. 性能优化（使用 triton-performance-optimizer）
```

---

## 常见算子生成指南

### Softmax

**模板**: `code-templates.md#softmax`

**关键点**:
1. 三遍扫描：找最大值 → 计算exp和求和 → 归一化
2. 数值稳定性：`exp(x - max_x)`
3. 分块处理大维度

### LayerNorm

**模板**: `code-templates.md#layernorm`

**关键点**:
1. 两遍扫描：计算均值 → 计算方差
2. 使用 `float32` 累加
3. 添加 `eps` 防止除零

### MatMul

**模板**: `code-templates.md#matmul`

**关键点**:
1. 分块计算：BLOCK_M × BLOCK_N × BLOCK_K
2. 使用 `tl.dot` 进行矩阵乘法
3. `float32` 累加器

---

## 常见错误预防

| 错误 | 预防方法 |
|------|---------|
| UB溢出 | 计算 UB 占用：`BLOCK_SIZE × 变量数 × 2字节 ≤ 85KB` |
| 数值溢出 | exp 前减最大值 |
| 精度损失 | 累加使用 float32 |
| 边界错误 | 所有 load/store 带 mask |

---

## 与其他 Skill 的协作

```
triton-code-generator (代码生成)
        ↓
triton-precision-verifier (精度验证)
        ↓
triton-performance-optimizer (性能优化)
```

**生成完成后**，建议用户：
1. 使用 `triton-precision-verifier` 验证正确性
2. 验证通过后，使用 `triton-performance-optimizer` 优化性能
