---
name: ascend-knowledge-manager
description: 管理Triton算子生成系统的知识库，支持知识吸收、分类、索引和检索。当用户提到"添加知识"、"记录问题"、"保存案例"、"知识库"、"检索知识"、"查找案例"、"发现新问题需要记录"时必须使用此skill。当其他skill需要检索相关知识时，主编排器也会调用此skill。即使用户没有明确说"知识管理"，只要涉及知识积累、案例记录、问题解决方案保存等，都应使用此skill。
---

# Ascend Knowledge Manager

管理Triton算子生成系统的知识库，支持知识的全生命周期管理。

## 核心职责

1. **知识吸收** - 接收并分析新知识，判断类型和存储位置
2. **知识验证** - 检查知识质量、重复和冲突
3. **知识存储** - 选择合适的存储方式和目录
4. **知识检索** - 提供关键词、标签、类型等多维度检索
5. **知识提升** - 识别高频使用的动态知识，建议提升为静态知识

## 知识库结构

```
.triton-gen/knowledge/
├── index.json           # 主索引文件
├── update_log.json      # 更新日志
├── documents/           # 文档类知识（Markdown）
│   └── {descriptive_name}.md
├── cases/               # 案例类知识（JSON）
│   ├── precision/       # 精度问题案例
│   ├── optimization/    # 优化案例
│   ├── conversion/      # 转换案例
│   ├── compilation/     # 编译错误案例
│   └── runtime/         # 运行时错误案例
└── rules/               # 规则类知识（JSON）
    ├── hardware_constraints.json
    └── performance_rules.json
```

## 工作流程

### 添加知识流程

1. **分析知识类型**
   - 判断是案例类、规则类还是文档类
   - 确定具体的子类型（如precision_issue、optimization等）

2. **验证知识质量**
   - 使用 `scripts/knowledge_validator.py` 验证格式
   - 检查必需字段是否完整
   - 检查是否存在重复或相似知识

3. **存储知识**
   - 按照命名规范生成文件名
   - 存储到对应目录
   - 更新索引文件

4. **记录更新日志**
   - 记录操作时间、类型、描述

### 检索知识流程

1. **确定检索策略**
   - 关键词检索：根据问题描述提取关键词
   - 标签检索：根据特定标签过滤
   - 类型检索：根据知识类型过滤

2. **执行检索**
   - 使用 `scripts/knowledge_indexer.py` 进行检索
   - 优先返回高相关度的结果

3. **返回结果**
   - 返回匹配的知识条目列表
   - 包含知识ID、标题、摘要、相关度

## 知识类型快速参考

| 类型 | 子类型 | 存储位置 | 文件格式 |
|------|--------|----------|----------|
| 案例 | precision_issue | cases/precision/ | JSON |
| 案例 | optimization | cases/optimization/ | JSON |
| 案例 | conversion | cases/conversion/ | JSON |
| 案例 | compilation_error | cases/compilation/ | JSON |
| 案例 | runtime_error | cases/runtime/ | JSON |
| 规则 | hardware_constraint | rules/ | JSON |
| 规则 | performance_rule | rules/ | JSON |
| 规则 | best_practice | rules/ | JSON |
| 规则 | api_limitation | rules/ | JSON |
| 文档 | - | documents/ | Markdown |

## 输入文件格式

### 添加知识

```json
{
  "action": "add",
  "knowledge_base_path": ".triton-gen/knowledge",
  "knowledge": {
    "type": "case",
    "subtype": "precision_issue",
    "content": {
      "case_id": "store_alignment_issue_20260316",
      "title": "tl.store地址对齐问题",
      "problem": {
        "description": "地址未对齐导致性能下降50%",
        "symptoms": ["性能下降", "延迟增加"],
        "affected_apis": ["tl.store"]
      },
      "solution": {
        "description": "确保地址16字节对齐",
        "code_before": "tl.store(ptr + offset, value)",
        "code_after": "aligned_offset = (offset // 16) * 16\ntl.store(ptr + aligned_offset, value)"
      },
      "metadata": {
        "tags": ["store", "alignment", "performance", "ascend"],
        "severity": "high"
      }
    }
  }
}
```

### 检索知识

```json
{
  "action": "search",
  "knowledge_base_path": ".triton-gen/knowledge",
  "query": {
    "keywords": ["softmax", "precision"],
    "type": "case",
    "subtype": "precision_issue",
    "tags": ["ascend"],
    "limit": 10
  }
}
```

### 获取知识详情

```json
{
  "action": "get",
  "knowledge_base_path": ".triton-gen/knowledge",
  "knowledge_id": "store_alignment_issue_20260316"
}
```

## 输出文件格式

### 操作结果

```json
{
  "success": true,
  "action": "add|search|get",
  "result": {
    "knowledge_id": "store_alignment_issue_20260316",
    "message": "知识添加成功"
  },
  "validation": {
    "valid": true,
    "warnings": []
  }
}
```

### 检索结果

```json
{
  "success": true,
  "action": "search",
  "results": [
    {
      "id": "store_alignment_issue_20260316",
      "type": "case",
      "subtype": "precision_issue",
      "title": "tl.store地址对齐问题",
      "relevance": 0.95,
      "summary": "地址未对齐导致性能下降50%..."
    }
  ],
  "total": 1
}
```

## 知识提升机制

当动态知识满足以下条件时，建议提升为静态知识：

### 提升条件

1. **使用频率**：usage_count > 5
2. **适用范围**：适用于多种场景
3. **问题重要性**：解决了常见或关键问题

### 提升流程

1. 检查知识的usage_count
2. 生成提升建议报告
3. 在输出中提示用户
4. 用户确认后，迁移到对应skill的references/目录

### 提升建议输出

```json
{
  "promotion_suggestions": [
    {
      "knowledge_id": "store_alignment_issue_20260316",
      "current_usage_count": 8,
      "suggested_target": "triton-code-generator/references/",
      "reason": "高频使用的性能优化案例",
      "priority": "high"
    }
  ]
}
```

## 使用脚本

### 验证知识

```bash
python scripts/knowledge_validator.py --input <knowledge_file> --type case|rule|document
```

### 重建索引

```bash
python scripts/knowledge_indexer.py --knowledge-base .triton-gen/knowledge --rebuild
```

## 知识检索

执行任务前，检索相关知识：
1. **静态知识**（总是加载）：
   - `references/knowledge_schema.md` - 知识格式规范
   - `references/knowledge_types.md` - 知识类型定义
2. **动态知识**（按需检索）：
   - `.triton-gen/knowledge/cases/` - 相关案例
   - `.triton-gen/knowledge/rules/` - 相关规则

## 参考文档

- [knowledge_schema.md](file:///d:/项目/trae/triton_gen/.triton-gen/skills/ascend-knowledge-manager/references/knowledge_schema.md) - 知识格式规范，添加知识前必读
- [knowledge_types.md](file:///d:/项目/trae/triton_gen/.triton-gen/skills/ascend-knowledge-manager/references/knowledge_types.md) - 知识类型定义，确定知识类型时参考
