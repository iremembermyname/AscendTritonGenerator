---
name: add-knowledge
description: 将新知识添加到知识库。用户调用 /add-knowledge 来添加文档、案例或规则。Invoke with /add-knowledge.
allowed-tools: Read, Grep, Glob, Write, Edit
---

# Add Knowledge

将新知识添加到Triton算子生成系统的知识库。

## Arguments

`$ARGUMENTS`

- 无参数：交互式添加知识
- `--type doc|case|rule`：指定知识类型
- `--file path`：从文件读取知识

## Workflow

### Step 1: Analyze Knowledge Content

1. **确定知识类型**：
   - **文档类**：语法参考、API文档、编程指南
   - **案例类**：成功/失败案例、优化案例、转换案例
   - **规则类**：约束条件、性能规则、最佳实践

2. **提取关键信息**：
   - 标题、描述、标签
   - 问题、解决方案
   - 代码示例

### Step 2: Validate Knowledge

1. **质量检查**：
   - 内容完整、描述清晰
   - 代码示例可运行
   - 无重复内容

2. **冲突检查**：
   - 与现有知识不冲突
   - 版本兼容性

### Step 3: Write to Knowledge Base

根据知识类型写入对应位置：

| 类型 | 目标位置 | 格式 |
|------|---------|------|
| 语法文档 | `.claude/data/syntax/` | Markdown |
| 模板文档 | `.claude/data/templates/` | Markdown |
| 指南文档 | `.claude/data/guides/` | Markdown |
| 精度案例 | `.claude/data/cases/precision/` | JSON |
| 优化案例 | `.claude/data/cases/optimization/` | JSON |
| 转换案例 | `.claude/data/cases/conversion/` | JSON |
| 规则 | `.claude/rules/` | Markdown |

### Step 4: Confirm Addition

输出确认信息：

```markdown
## 知识添加成功

- 类型: [类型]
- 位置: [文件路径]
- 标签: [标签列表]

请验证知识是否正确添加。
```

## Case Format

```json
{
  "case_id": "[type]_[name]_[number]",
  "case_type": "precision|optimization|conversion",
  "title": "[简短标题]",
  "problem": {
    "description": "[问题描述]",
    "symptoms": ["症状1", "症状2"]
  },
  "solution": {
    "description": "[解决方案描述]",
    "code_before": "[修复前代码]",
    "code_after": "[修复后代码]"
  },
  "metadata": {
    "tags": ["标签1", "标签2"],
    "created_at": "YYYY-MM-DD"
  }
}
```

## Examples

### Example 1: Add a precision case

```
/add-knowledge --type case

请提供案例信息：
- 标题: Softmax数值溢出问题
- 类型: precision
- 问题描述: 大输入值导致exp溢出
- 解决方案: 减最大值防止溢出
- 代码示例: [提供代码]
```

### Example 2: Add from file

```
/add-knowledge --file ./my_knowledge.md
```

## Error Handling

| 错误 | 处理方式 |
|------|---------|
| 知识类型不明确 | 询问用户确认 |
| 内容不完整 | 提示补充缺失部分 |
| 重复知识 | 提示已存在，询问是否更新 |
| 格式错误 | 提供正确格式示例 |
