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

- 目标平台：Ascend NPU

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
| 算子生成 | triton-expert agent | "生成/创建/实现算子" |
| CUDA转换 | triton-expert agent | "转换CUDA/迁移代码" |
| 性能优化 | triton-expert agent | "优化性能/加速" |
| 精度验证 | /verify-precision | "验证精度/测试正确性" |
| 性能分析 | /profile-performance | "分析性能/性能瓶颈" |
| 问题调试 | /debug-kernel | "调试/NaN/Inf/错误" |
| 添加知识 | /add-knowledge command | "添加知识" |

## Extended Configuration

### Agents

| Agent | Purpose | Activation Trigger |
|-------|---------|-------------------|
| `planner` | 任务规划 | 多步骤任务、架构决策 |
| `triton-expert` | 算子开发专家 | 算子生成、CUDA转换、性能优化 |

### Skills

| Skill | Purpose | Invocation |
|-------|---------|------------|
| `verify-precision` | 精度验证流程 | `/verify-precision` |
| `profile-performance` | 性能分析流程 | `/profile-performance` |
| `debug-kernel` | 算子调试流程 | `/debug-kernel` |

### Commands

| Command | Purpose |
|---------|---------|
| `/add-knowledge` | 添加新知识到知识库 |

### Rules

| Rule | Purpose |
|------|---------|
| `ascend-hardware.md` | Ascend硬件约束 |
| `triton-code.md` | Triton代码规范 |

### Knowledge

| Directory | Content |
|-----------|---------|
| `data/templates/` | 代码模板 |
| `data/syntax/` | 语法参考 |
| `data/guides/` | 指南文档 |
| `data/cases/` | 案例库 |
