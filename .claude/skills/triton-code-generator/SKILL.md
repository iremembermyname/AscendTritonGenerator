---
name: triton-code-generator
description: Triton算子代码生成指导。当用户需要生成Triton kernel、实现高性能算子、或修复代码错误时使用此skill。Use when user wants to generate Triton code, implement operators, or fix code errors.
---

# Triton Code Generator

Triton代码生成技能，提供step-by-step的代码生成指导。

## When to Use

This skill is triggered when:
- User asks "how do I implement [operator]?"
- User wants to generate Triton kernel code
- User needs to fix code errors
- User mentions operator implementation

## Knowledge Retrieval

执行任务前，检索相关知识：
1. `@.claude/data/syntax/triton-syntax.md` - Triton语法参考
2. `@.claude/data/syntax/ascend-extensions.md` - Ascend扩展API
3. `@.claude/data/templates/code-templates.md` - 代码模板库

## Step-by-Step Guide

### Step 1: Analyze Requirements

1. Understand operator semantics
2. Determine computation pattern (element-wise, reduction, etc.)
3. Identify memory access pattern
4. Check hardware constraints

### Step 2: Design Kernel

遵循以下结构：

```python
@triton.jit
def kernel_name(
    输入指针,
    输出指针,
    形状参数,
    BLOCK_SIZE: tl.constexpr,
):
    # 1. 获取program ID
    pid = tl.program_id(axis=0)
    # 2. 计算偏移量
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    # 3. 加载数据（带mask）
    mask = offsets < n_elements
    x = tl.load(input_ptr + offsets, mask=mask)
    # 4. 执行计算
    y = compute(x)
    # 5. 存储结果
    tl.store(output_ptr + offsets, y, mask=mask)
```

### Step 3: Apply Ascend Optimizations

**关键约束**：
- UB单次循环占用 ≤ 85KB
- Block大小 ≤ 1024
- 避免使用 `tl.load(..., other=value)` 影响流水线

**优化技术**：
- Double Buffering：MTE与Vector并行
- 多Token并行处理：减少循环次数
- 连续内存访问：避免离散访问

### Step 4: Generate Host Function

```python
def operator_name(x: torch.Tensor) -> torch.Tensor:
    output = torch.empty_like(x)
    grid = lambda meta: (triton.cdiv(x.shape[0], meta['BLOCK_SIZE']),)
    kernel_name[grid](x, output, x.shape[0], BLOCK_SIZE=256)
    return output
```

### Step 5: Generate Test Code

包含正确性测试和性能测试。

## Key Requirements

1. **可读性**：清晰的命名、适当的注释、合理的结构
2. **性能**：遵循Ascend优化原则、合理的Block大小
3. **健壮性**：边界条件处理、数值稳定性、错误处理

## Common Mistakes

- ❌ Block大小超过1024
- ❌ UB使用超过85KB
- ❌ 使用带other参数的tl.load
- ❌ 忽略边界条件处理
- ❌ 数值不稳定（如exp溢出）

## Reference Implementations

| Operator | Template | Description |
|----------|----------|-------------|
| softmax | code-templates.md#softmax | Numerical stability pattern |
| layernorm | code-templates.md#layernorm | Reduction pattern |
| gelu | code-templates.md#gelu | Element-wise pattern |
| matmul | code-templates.md#matmul | Tiling pattern |
