# 知识类型定义

本文档定义了Triton算子生成系统知识库的知识类型。

---

## 1. 知识类型概述

知识库包含三大类知识：

| 类型 | 存储方式 | 更新频率 | 用途 |
|------|---------|---------|------|
| 案例类 | JSON文件 | 运行时积累 | 记录问题和解决方案 |
| 规则类 | JSON文件 | 手动维护 | 定义约束和最佳实践 |
| 文档类 | Markdown文件 | 手动维护 | 提供详细参考文档 |

---

## 2. 案例类知识

### 2.1 精度问题案例 (precision_issue)

记录精度相关的案例。

```json
{
  "case_type": "precision_issue",
  "problem": {
    "description": "精度问题描述",
    "symptoms": ["NaN", "Inf", "大误差"],
    "affected_apis": ["tl.exp", "tl.sum"]
  },
  "solution": {
    "description": "解决方案",
    "code_before": "...",
    "code_after": "..."
  }
}
```

**常见精度问题**：
- 数值溢出/下溢
- 精度损失
- NaN/Inf问题
- 类型转换问题

### 2.2 优化案例 (optimization)

记录性能优化案例。

```json
{
  "case_type": "optimization",
  "problem": {
    "description": "性能问题描述",
    "symptoms": ["性能不达标", "延迟高"],
    "baseline_performance": "2.0 ms",
    "target_performance": "0.5 ms"
  },
  "solution": {
    "description": "优化方案",
    "techniques": ["多Token并行", "流水线优化"],
    "code_before": "...",
    "code_after": "...",
    "performance_after": "0.4 ms"
  }
}
```

**常见优化技术**：
- UB容量优化
- 多Token并行处理
- 流水线优化
- 内存访问优化
- 分核优化

### 2.3 转换案例 (conversion)

记录CUDA到Ascend转换案例。

```json
{
  "case_type": "conversion",
  "problem": {
    "description": "转换问题描述",
    "cuda_code": "...",
    "issue": "不支持的API"
  },
  "solution": {
    "description": "转换方案",
    "ascend_code": "...",
    "changes": ["移除tl.debug_barrier", "调整BLOCK_SIZE"]
  }
}
```

**常见转换问题**：
- 不支持的API
- 硬件限制差异
- 性能特性差异

### 2.4 编译错误案例 (compilation_error)

记录编译错误案例。

```json
{
  "case_type": "compilation_error",
  "problem": {
    "description": "编译错误描述",
    "error_message": "...",
    "error_code": "..."
  },
  "solution": {
    "description": "解决方案",
    "code_before": "...",
    "code_after": "..."
  }
}
```

### 2.5 运行时错误案例 (runtime_error)

记录运行时错误案例。

```json
{
  "case_type": "runtime_error",
  "problem": {
    "description": "运行时错误描述",
    "error_message": "...",
    "stack_trace": "..."
  },
  "solution": {
    "description": "解决方案",
    "code_before": "...",
    "code_after": "..."
  }
}
```

---

## 3. 规则类知识

### 3.1 硬件约束规则 (hardware_constraint)

定义硬件限制。

```json
{
  "rule_type": "hardware_constraint",
  "title": "Ascend Block大小限制",
  "constraints": {
    "max_block_size": 1024,
    "recommended_block_size": 256
  },
  "rationale": "硬件资源限制"
}
```

**常见硬件约束**：
- Block大小限制
- UB容量限制
- 核数限制
- 内存带宽限制

### 3.2 性能规则 (performance_rule)

定义性能优化规则。

```json
{
  "rule_type": "performance_rule",
  "title": "UB容量优化规则",
  "description": "单次循环UB占用应 <= 85KB",
  "rationale": "确保Double Buffering正常工作"
}
```

**常见性能规则**：
- UB容量优化
- 流水线优化
- 内存访问优化
- 分核策略

### 3.3 最佳实践规则 (best_practice)

定义最佳实践。

```json
{
  "rule_type": "best_practice",
  "title": "累加操作使用FP32",
  "description": "累加操作应使用float32精度",
  "examples": {
    "good": "acc = tl.zeros([M, N], dtype=tl.float32)",
    "bad": "acc = tl.zeros([M, N], dtype=tl.bfloat16)"
  }
}
```

### 3.4 API限制规则 (api_limitation)

定义API使用限制。

```json
{
  "rule_type": "api_limitation",
  "title": "tl.load使用限制",
  "description": "避免使用带other参数的load",
  "rationale": "会阻止MTE独立执行，影响流水线性能"
}
```

---

## 4. 文档类知识

### 4.1 API文档

提供API详细说明。

```markdown
# tl.load API文档

## 功能
从全局内存加载数据到UB。

## 语法
tl.load(ptr, mask=None, other=None, ...)

## 参数
- ptr: 数据指针
- mask: 可选的掩码
- other: 默认值（Ascend上避免使用）

## 示例
...
```

### 4.2 编程指南

提供编程指导。

```markdown
# Ascend Triton编程指南

## 概述
...

## 内存管理
...

## 性能优化
...
```

### 4.3 教程文档

提供学习教程。

```markdown
# Softmax算子开发教程

## 目标
实现一个高性能的Softmax算子。

## 步骤
1. 分析需求
2. 设计kernel
3. 实现代码
4. 验证精度
5. 优化性能
```

---

## 5. 知识关系

### 5.1 关联关系

知识之间可以存在关联：

```json
{
  "related_knowledge": [
    "case_001",
    "rule_001"
  ]
}
```

### 5.2 依赖关系

知识之间可以存在依赖：

```json
{
  "depends_on": [
    "rule_hardware_constraint_001"
  ]
}
```

---

## 6. 知识生命周期

### 6.1 创建

1. 分析新知识
2. 验证知识质量
3. 分配唯一ID
4. 存储到知识库
5. 更新索引

### 6.2 使用

1. 检索知识
2. 增加使用计数
3. 记录使用结果

### 6.3 更新

1. 验证更新内容
2. 更新知识内容
3. 更新索引
4. 记录更新日志

### 6.4 提升

1. 检查提升条件
2. 建议提升
3. 用户确认
4. 迁移到静态知识

### 6.5 归档

1. 标记为归档状态
2. 移动到归档目录
3. 更新索引
