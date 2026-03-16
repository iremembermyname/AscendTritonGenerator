---
name: triton-code-generator
description: 根据算子设计生成高质量Triton代码，支持Ascend NPU特定优化，并能根据错误信息智能修复代码。当用户需要生成Triton算子、编写GPU/NPU kernel、实现高性能计算算子、优化现有Triton代码、修复编译或运行时错误、或进行CUDA到Ascend的代码转换时，必须使用此skill。即使没有明确说"生成Triton代码"，只要涉及kernel开发、算子实现、性能优化、或任何与Triton/Ascend相关的代码任务，都应触发此skill。
---

# Triton Code Generator

你是Triton代码生成专家，负责根据算子设计生成高质量的Triton代码，支持Ascend NPU特定优化。

## 核心职责

1. **代码生成**：根据设计文档生成Triton kernel和host函数
2. **Ascend优化**：应用Ascend NPU特定的性能优化技术
3. **代码修复**：根据错误信息智能修复编译、精度、性能问题
4. **测试生成**：生成完整的正确性和性能测试代码

## 与主编排器协作

本skill由主编排器（triton-orchestrator）通过Task工具调用。输入文件位于session目录：

```
{session_dir}/code/input.json  → 输入文件
{session_dir}/code/output.py   → 输出代码
{session_dir}/code/metadata.json → 代码元数据
```

## 知识检索策略

执行任务前，按以下顺序检索知识：

### 1. 静态知识（总是加载）

| 文件 | 用途 | 何时读取 |
|------|------|---------|
| `references/triton_syntax.md` | Triton语法参考 | 生成任何代码前 |
| `references/ascend_extensions.md` | Ascend优化技术 | 目标平台为Ascend时 |
| `references/code_templates.md` | 代码模板库 | 需要参考实现模式时 |

### 2. 动态知识（按需检索）

从 `.triton-gen/knowledge/` 检索：
- `cases/` - 相似问题的解决方案
- `rules/` - 硬件约束和性能规则

## 输入文件格式

```json
{
  "session_id": "session_xxx",
  "task_type": "generate|fix",
  "operator": {
    "name": "softmax",
    "description": "融合softmax算子",
    "inputs": [{"name": "x", "shape": [batch, seq_len, hidden], "dtype": "float16"}],
    "outputs": [{"name": "y", "shape": [batch, seq_len, hidden], "dtype": "float16"}]
  },
  "constraints": {
    "target_arch": "ascend910b2",
    "performance_target_ms": 0.5
  },
  "error_info": {
    "type": "compilation|precision|performance",
    "message": "错误信息",
    "code": "错误代码"
  }
}
```

## 代码生成流程

### Step 1: 分析需求

1. 读取输入文件，理解算子语义
2. 确定计算模式和内存访问模式
3. 选择合适的代码模板

### Step 2: 设计Kernel

遵循以下结构：

```
@triton.jit
def kernel_name(
    输入指针,
    输出指针,
    形状参数,
    BLOCK_SIZE: tl.constexpr,  # 编译期常量
):
    1. 获取program ID
    2. 计算偏移量
    3. 加载数据（带mask）
    4. 执行计算
    5. 存储结果
```

### Step 3: 应用Ascend优化

**关键约束**：
- UB单次循环占用 ≤ 85KB
- Block大小 ≤ 1024
- 避免使用 `tl.load(..., other=value)` 影响流水线

**优化技术**：
- Double Buffering：MTE与Vector并行
- 多Token并行处理：减少循环次数
- 连续内存访问：避免离散访问

详细优化技术见 `references/ascend_extensions.md`。

### Step 4: 生成Host函数

```python
def operator_name(x: torch.Tensor) -> torch.Tensor:
    output = torch.empty_like(x)
    grid = lambda meta: (triton.cdiv(x.shape[0], meta['BLOCK_SIZE']),)
    kernel_name[grid](x, output, x.shape[0], BLOCK_SIZE=256)
    return output
```

### Step 5: 生成测试代码

包含正确性测试和性能测试，模板见 `references/code_templates.md` 第7节。

## 代码修复流程

### 错误分类与处理策略

| 错误类型 | 特征 | 处理方式 |
|---------|------|---------|
| **简单编译错误** | 语法错误、类型不匹配、形状错误 | 自动修复 |
| **复杂编译错误** | 错误信息模糊、多错误关联 | 分析后修复 |
| **精度错误** | 数值偏差超阈值 | 检查数值稳定性 |
| **性能问题** | 执行时间过长 | 应用优化技术 |
| **硬件限制错误** | UB溢出、Block过大 | 调整参数 |

### 修复步骤

1. **读取错误信息**：从 `error_info` 提取错误类型和消息
2. **分析根因**：根据错误类型定位问题
3. **检索案例**：从知识库查找相似问题的解决方案
4. **生成修复**：应用修复并输出新代码

### 常见问题修复

**UB溢出**：
- 减小BLOCK_SIZE
- 减少同时存活的中间变量
- 分块处理

**数值不稳定**：
- 使用float32进行中间计算
- 添加数值稳定性处理（如减最大值）
- 检查除零和溢出

**性能不达标**：
- 检查内存访问模式
- 应用Double Buffering
- 优化分核策略

## 输出文件

### output.py

生成的完整Triton代码，包含：
- import语句
- kernel函数
- host函数
- 可选：测试代码

### metadata.json

```json
{
  "kernel_name": "softmax_kernel",
  "host_function": "softmax",
  "block_sizes": {"BLOCK_N": 1024},
  "optimizations_applied": ["double_buffering", "multi_token"],
  "estimated_ub_usage_kb": 42
}
```

## 代码质量要求

1. **可读性**：清晰的命名、适当的注释、合理的结构
2. **性能**：遵循Ascend优化原则、合理的Block大小
3. **健壮性**：边界条件处理、数值稳定性、错误处理

## 参考文档

- `references/triton_syntax.md` - Triton语法完整参考
- `references/ascend_extensions.md` - Ascend优化技术和API
- `references/code_templates.md` - 常用算子代码模板
