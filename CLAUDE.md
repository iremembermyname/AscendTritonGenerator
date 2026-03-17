# CLAUDE.md - Triton Operator Generation System

## WHAT: Project Overview

Triton算子生成系统，支持Ascend NPU平台的算子开发、验证和优化。

**技术栈**：Triton, PyTorch, Ascend NPU

**核心目录**：
- `.claude/agents/` - 智能体（复杂任务处理）
- `.claude/skills/` - 技能（操作指导）
- `.claude/commands/` - 命令（快捷动作）
- `.claude/rules/` - 规则（代码约束）
- `.claude/data/` - 知识库

## WHY: Purpose

帮助开发者高效生成、验证和优化Ascend Triton算子。

## HOW: Core Commands

- `/add-knowledge` - 添加新知识到知识库

## Boundaries

### Constraints

- 目标平台：Ascend NPU（910B2）
- Block大小：≤1024
- UB使用：≤85KB/循环

### Always Do

- 生成代码后验证精度
- 遵循Ascend硬件约束
- 记录成功/失败案例

### Ask First

- 架构决策
- 性能目标设定
- 知识库更新

### Never Do

- 忽略硬件约束
- 跳过精度验证
- 删除知识库内容

## Progressive Disclosure: Task Routing

| 任务类型 | 路由目标 | 触发条件 |
|---------|---------|---------|
| 算子生成 | planner agent | "生成/创建算子" |
| 代码优化 | triton-performance-optimizer skill | "优化性能" |
| 精度验证 | triton-precision-verifier skill | "验证精度" |
| CUDA转换 | cuda-to-ascend-converter agent | "转换CUDA" |
| 添加知识 | /add-knowledge command | "添加知识" |

## Extended Configuration

- **Agents**: `.claude/agents/` - 复杂任务处理
- **Skills**: `.claude/skills/` - 操作指导
- **Commands**: `.claude/commands/` - 快捷动作
- **Rules**: `.claude/rules/` - 代码约束
- **Knowledge**: `.claude/data/` - 统一知识库
