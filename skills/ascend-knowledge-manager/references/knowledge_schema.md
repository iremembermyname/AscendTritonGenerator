# 知识格式规范

本文档定义了Triton算子生成系统知识库的知识格式规范。

---

## 1. 案例类知识格式

### 1.1 文件命名

```
{descriptive_name}_{YYYYMMDD}.json
```

示例：`store_alignment_issue_20260316.json`

### 1.2 JSON结构

```json
{
  "case_id": "store_alignment_issue_20260316",
  "case_type": "precision_issue | optimization | conversion | compilation_error | runtime_error",
  "title": "简短描述性问题标题",
  "problem": {
    "description": "问题的详细描述",
    "symptoms": ["症状1", "症状2"],
    "affected_apis": ["tl.store", "tl.load"],
    "error_message": "错误信息（如有）",
    "context": {
      "input_shape": [128, 1024],
      "dtype": "bfloat16",
      "hardware": "ascend910b2"
    }
  },
  "solution": {
    "description": "解决方案的详细描述",
    "code_before": "修复前的代码片段",
    "code_after": "修复后的代码片段",
    "steps": ["步骤1", "步骤2"]
  },
  "metadata": {
    "tags": ["store", "alignment", "performance", "ascend"],
    "severity": "high | medium | low",
    "created_at": "2026-03-16",
    "updated_at": "2026-03-16",
    "author": "作者",
    "usage_count": 0,
    "success_rate": 1.0
  }
}
```

### 1.3 字段说明

| 字段 | 必需 | 类型 | 说明 |
|------|------|------|------|
| case_id | 是 | string | 唯一标识符 |
| case_type | 是 | string | 案例类型 |
| title | 是 | string | 标题 |
| problem | 是 | object | 问题描述 |
| solution | 是 | object | 解决方案 |
| metadata | 否 | object | 元数据 |

---

## 2. 规则类知识格式

### 2.1 文件命名

```
{descriptive_name}.json
```

示例：`ascend_block_size_limit.json`

### 2.2 JSON结构

```json
{
  "rule_id": "ascend_block_size_limit",
  "rule_type": "hardware_constraint | performance_rule | best_practice | api_limitation",
  "title": "规则标题",
  "description": "规则的详细描述",
  "constraints": {
    "max_block_size": 1024,
    "recommended_block_size": 256
  },
  "rationale": "规则存在的原因",
  "examples": {
    "good": "符合规则的代码示例",
    "bad": "违反规则的代码示例"
  },
  "related_apis": ["tl.program_id", "tl.arange"],
  "tags": ["block", "hardware", "ascend"],
  "created_at": "2026-03-16",
  "updated_at": "2026-03-16",
  "source": "来源文档或经验"
}
```

### 2.3 字段说明

| 字段 | 必需 | 类型 | 说明 |
|------|------|------|------|
| rule_id | 是 | string | 唯一标识符 |
| rule_type | 是 | string | 规则类型 |
| title | 是 | string | 标题 |
| description | 是 | string | 描述 |
| constraints | 否 | object | 约束条件 |
| rationale | 否 | string | 原因说明 |
| examples | 否 | object | 示例代码 |
| tags | 否 | array | 标签列表 |

---

## 3. 文档类知识格式

### 3.1 文件命名

```
{descriptive_name}.md
```

示例：`custom_optimization.md`

### 3.2 Markdown结构

```markdown
# 文档标题

## 概述

文档的简要概述。

## 详细内容

### 子章节1

内容...

### 子章节2

内容...

## 示例

代码示例...

## 参考

- 参考1
- 参考2
```

---

## 4. 索引文件格式

### 4.1 主索引文件

位置：`knowledge/index.json`

```json
{
  "keywords": {
    "softmax": ["case_001", "rule_001"],
    "precision": ["case_002", "case_003"]
  },
  "tags": {
    "ascend": ["case_001", "rule_001"],
    "performance": ["case_002", "rule_002"]
  },
  "types": {
    "case:precision_issue": ["case_001", "case_002"],
    "case:optimization": ["case_003"],
    "rule:hardware_constraint": ["rule_001"]
  },
  "entries": {
    "case_001": {
      "id": "case_001",
      "type": "case",
      "subtype": "precision_issue",
      "title": "...",
      "tags": [...],
      "keywords": [...],
      "created_at": "...",
      "usage_count": 0
    }
  },
  "updated_at": "2026-03-16T10:00:00"
}
```

---

## 5. 更新日志格式

位置：`knowledge/update_log.json`

```json
{
  "updates": [
    {
      "timestamp": "2026-03-16T10:00:00",
      "action": "add|update|delete",
      "knowledge_id": "case_001",
      "knowledge_type": "case",
      "description": "添加了新案例",
      "changes": {
        "field": "old_value -> new_value"
      }
    }
  ]
}
```

---

## 6. 标签规范

### 6.1 推荐标签

| 类别 | 标签 |
|------|------|
| 平台 | ascend, cuda, npu, gpu |
| 操作 | load, store, compute, reduce |
| 问题类型 | precision, performance, compilation, runtime |
| API | tl.load, tl.store, tl.dot, tl.sum |
| 数据类型 | bf16, fp16, fp32, int8 |
| 硬件 | ub, gm, vector, mte |

### 6.2 标签命名规范

- 使用小写字母
- 使用下划线分隔单词
- 简洁明了
- 避免冗余

---

## 7. ID命名规范

### 7.1 案例ID

格式：`{descriptive_name}_{YYYYMMDD}`

示例：
- `store_alignment_issue_20260316`
- `softmax_precision_error_20260316`
- `matmul_optimization_20260316`

### 7.2 规则ID

格式：`{category}_{descriptive_name}`

示例：
- `hardware_block_size_limit`
- `performance_ub_optimization`
- `api_load_constraint`

---

## 8. 版本控制

### 8.1 版本字段

```json
{
  "version": "1.0.0",
  "changelog": [
    {
      "version": "1.0.0",
      "date": "2026-03-16",
      "changes": ["初始版本"]
    }
  ]
}
```

### 8.2 版本号规则

- 主版本号：重大变更
- 次版本号：功能增加
- 修订号：问题修复
