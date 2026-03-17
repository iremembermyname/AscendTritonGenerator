# AscendTritonGenerator 设计理念

> 基于 Claude Agents 组织模式最佳实践的深度解析

## 一、概述与设计理念

### 1.1 项目背景

在 AI 辅助编程领域，一个核心挑战是：**如何让 AI 在特定领域（如 Ascend NPU 开发）保持高效、准确、可维护**？

传统的做法是将所有项目信息塞进一个巨大的配置文件中，期望"信息越全，AI 表现越好"。然而实践证明，这种做法恰恰相反——Context 太长时，AI 在处理具体任务时反而抓不住重点。

AscendTritonGenerator 通过实践，沉淀出一套成熟的解决方案：**分层架构 + Progressive Disclosure（渐进式信息披露）**。

### 1.2 核心设计理念

#### 1.2.1 Progressive Disclosure（渐进式信息披露）

信息按性质分层，按需加载：

```
Layer 1: CLAUDE.md（入口文件）
    ↓ 包含项目简介、技术栈、目录结构、基本命令、底线约束
    ↓ 以及"路由表"指引 AI 去找深层信息
    
Layer 2: Rules（全局约束）
    ↓ 按路径自动匹配加载
    ↓ 如 ascend-hardware.md 在改动 Triton 代码时自动激活
    
Layer 3: Agents/Skills/Commands（按需激活）
    ↓ 根据任务类型动态加载
    ↓ 处理 CUDA 转换时 cuda-to-ascend-converter 被激活
```

**关键原则**：
- 入口文件保持精简
- 细节知识被拆分到各层
- 每一层都有明确的激活条件

#### 1.2.2 分层解耦

| 层级 | 类型 | 职责 | 激活方式 |
|------|------|------|----------|
| **Agents** | 智能体 | 领域专家，执行复杂任务 | 自动/手动触发 |
| **Skills** | 技能 | 引导式开发流程 | 任务类型匹配 |
| **Commands** | 命令 | 用户直接调用的动作 | 用户调用 `/command-name` |
| **Rules** | 规则 | 代码质量标准 | 路径匹配自动加载 |

**设计要点**：
- Agents 是"会思考的专家"，有独立推理能力
- Skills 是"操作手册"，提供 step-by-step 指导
- Commands 是"快捷动作"，执行特定操作
- Rules 是"约束条件"，自动应用于相关文件

#### 1.2.3 智能决策模式

简单问题自动处理，复杂问题询问用户：

```
任务复杂度判断：
├── 简单任务（单文件、明确实现）→ 自动执行
├── 中等任务（多文件、需要规划）→ 先规划再执行
└── 复杂任务（架构决策、模糊需求）→ 询问用户确认
```

---

## 二、整体架构设计

### 2.1 四层架构图

```
┌─────────────────────────────────────────────────────────────────┐
│                        CLAUDE.md                                 │
│  （入口文件：项目概述 + 路由表 + 底线约束）                          │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                        settings.json                             │
│  （全局配置：权限等）                                              │
└─────────────────────────────────────────────────────────────────┘
                              │
        ┌─────────────────────┼─────────────────────┐
        │                     │                     │
        ▼                     ▼                     ▼
┌───────────────┐    ┌───────────────┐    ┌───────────────┐
│    Rules/     │    │   Agents/     │    │  Commands/    │
│   （规则层）   │    │  （智能体层）  │    │  （命令层）    │
│               │    │               │    │               │
│ 路径匹配加载   │    │ 自动/手动触发  │    │ 用户直接调用   │
│ ascend-hw     │    │ planner       │    │ add-knowledge │
│ triton-code   │    │ cuda-converter│    │               │
└───────────────┘    └───────────────┘    └───────────────┘
                              │
                              ▼
                     ┌───────────────┐
                     │   Skills/     │
                     │  （技能层）    │
                     │               │
                     │ 任务类型匹配   │
                     │ code-gen      │
                     │ perf-opt      │
                     │ precision     │
                     └───────────────┘
                              │
                              ▼
                     ┌───────────────┐
                     │    Data/      │
                     │  （知识库）    │
                     │               │
                     │ syntax/       │
                     │ templates/    │
                     │ guides/       │
                     │ cases/        │
                     └───────────────┘
```

### 2.2 各层职责边界

| 层级 | 核心职责 | 输入 | 输出 | 工具权限 |
|------|----------|------|------|----------|
| **CLAUDE.md** | 项目入口、路由导航 | 用户请求 | 指向具体资源 | 只读 |
| **Rules** | 代码质量约束 | 文件路径 | 约束规则 | 只读 |
| **Agents** | 执行复杂任务 | 任务描述 | 执行结果 | 可配置 |
| **Skills** | 引导开发流程 | 用户需求 | 指导文档 | 只读 |
| **Commands** | 执行特定动作 | 用户参数 | 动作结果 | 可配置 |

---

## 三、组件详解

### 3.1 Agents（智能体）

#### 3.1.1 概念与定位

Agents 是"会思考的领域专家"，具有以下特点：

- **独立推理能力**：能够分析问题、制定计划、执行任务
- **领域专精**：每个 agent 专注于特定领域
- **工具权限可控**：可配置可用的工具集
- **模型可选**：可根据任务复杂度选择不同模型

#### 3.1.2 当前 Agents

| Agent | 功能 | 触发方式 | 模型 |
|-------|------|----------|------|
| planner | 任务规划专家 | PROACTIVE | opus |
| cuda-to-ascend-converter | CUDA 转换专家 | 被动触发 | sonnet |

#### 3.1.3 触发机制设计

**主动触发（PROACTIVE）**：
```yaml
description: Triton算子开发的任务规划专家。当用户需要生成新算子、优化现有代码、转换CUDA代码、或进行其他涉及多步骤的复杂任务时，主动使用此agent进行规划。
```

- AI 自动判断是否需要激活
- 适用于：多文件修改、新功能设计、架构决策

**被动触发**：
```yaml
description: CUDA Triton到Ascend Triton的转换专家。仅当处理CUDA代码转换、GPU到NPU迁移、或Ascend兼容性问题时使用此agent。
```

- 仅在特定上下文出现时激活
- 适用于：特定领域问题

### 3.2 Skills（技能）

#### 3.2.1 概念与定位

Skills 是"引导式开发流程"，具有以下特点：

- **Step-by-step 指导**：提供详细的操作步骤
- **可复用流程**：标准化重复性任务
- **任务类型匹配**：根据任务自动激活
- **只读性质**：不直接修改代码，只提供指导

#### 3.2.2 当前 Skills

| Skill | 功能 | 触发场景 |
|-------|------|---------|
| triton-code-generator | 代码生成指导 | 生成 Triton kernel、实现算子 |
| triton-performance-optimizer | 性能优化指导 | 性能调优、瓶颈分析 |
| triton-precision-verifier | 精度验证指导 | 正确性验证、精度问题调试 |

#### 3.2.3 与 Agents 的区别

| 特性 | Agents | Skills |
|------|--------|--------|
| **定位** | 领域专家 | 操作手册 |
| **触发方式** | 自动/手动 | 任务类型匹配 |
| **执行方式** | 独立推理执行 | 提供指导步骤 |
| **工具权限** | 可配置 | 只读 |
| **适用场景** | 复杂任务 | 标准化流程 |

### 3.3 Commands（命令）

#### 3.3.1 概念与定位

Commands 是"用户直接调用的动作"，具有以下特点：

- **用户主动调用**：通过 `/command-name` 触发
- **执行特定操作**：有明确的输入输出
- **可配置工具权限**：可执行文件操作等
- **参数支持**：支持命令行参数

#### 3.3.2 当前 Commands

| Command | 功能 | 调用方式 |
|---------|------|----------|
| add-knowledge | 添加知识到知识库 | `/add-knowledge` |

### 3.4 Rules（规则）

#### 3.4.1 概念与定位

Rules 是"代码质量标准"，具有以下特点：

- **路径匹配加载**：根据文件路径自动激活
- **约束性规则**：定义必须遵守的规范
- **只读性质**：不执行操作，只提供约束
- **项目级标准**：全局适用的质量标准

#### 3.4.2 当前 Rules

| Rule | 匹配路径 | 功能 |
|------|----------|------|
| ascend-hardware | `**/ascend/**/*.py`, `**/*ascend*.py` | Ascend 硬件约束 |
| triton-code | `**/*.py`, `**/triton/**/*.py` | Triton 代码规范 |

### 3.5 Data（知识库）

#### 3.5.1 概念与定位

知识库是系统的"记忆"，存储各类参考信息：

- **语法参考**：Triton 语法、Ascend 扩展 API
- **代码模板**：常用算子的参考实现
- **指南文档**：优化技巧、问题排查
- **案例库**：成功/失败案例记录

#### 3.5.2 目录结构

```
data/
├── syntax/                    # 语法参考
│   ├── triton-syntax.md       # Triton 核心语法
│   └── ascend-extensions.md   # Ascend 扩展 API
├── templates/                 # 代码模板
│   └── code-templates.md      # 常用算子模板
├── guides/                    # 指南文档
│   ├── optimization-tips.md   # 性能优化技巧
│   ├── precision-issues.md    # 精度问题指南
│   └── troubleshooting.md     # 问题排查指南
└── cases/                     # 案例库
    ├── conversion/            # CUDA 转换案例
    ├── optimization/          # 性能优化案例
    └── precision/             # 精度问题案例
```

---

## 四、协作机制与触发策略

### 4.1 任务路由表

| 任务类型 | 路由目标 | 触发条件 |
|---------|---------|---------|
| 算子生成 | planner agent → triton-code-generator skill | "生成/创建算子" |
| 代码优化 | triton-performance-optimizer skill | "优化性能" |
| 精度验证 | triton-precision-verifier skill | "验证精度" |
| CUDA 转换 | cuda-to-ascend-converter agent | "转换 CUDA" |
| 添加知识 | /add-knowledge command | "添加知识" |

### 4.2 工具选择策略

| 任务类型 | 推荐工具集 |
|----------|------------|
| 规划/审查 | Read, Grep, Glob, Task |
| 执行操作 | Read, Grep, Glob, Task, Write, Edit |
| 研究探索 | Read, Grep, Glob |

### 4.3 模型选择策略

| 复杂度 | 模型 | 适用场景 |
|--------|------|----------|
| **高** | Opus | 架构设计、复杂规划、跨文件分析 |
| **中** | Sonnet | 代码审查、中等复杂度任务 |
| **低** | Haiku | 格式检查、简单任务 |

---

## 五、维护与更新流程

### 5.1 MAINTAINER GUIDE 设计

每个组件文件末尾应包含 MAINTAINER GUIDE：

```markdown
______________________________________________________________________

<!--
================================================================================
                            MAINTAINER GUIDE
================================================================================

Location: .claude/agents/planner.md
Activation: Automatic (PROACTIVE) when complex tasks detected

## Design Philosophy

- **Read-Only Agent**: Never modify code directly; only research and produce plans
- **Tools**: Read, Grep, Glob, Task (intentionally limited)
- **Model**: Opus (deep reasoning for architectural decisions)
- **Proactive**: Auto-activates for multi-file changes, new features, architectural decisions

## How to Update

### Updating Plan Output Format
1. Add to the markdown template in "Phase 3: Plan Output"
2. Document when the section is required

================================================================================
-->
```

### 5.2 更新触发条件

| 触发条件 | 更新内容 | 更新位置 |
|----------|----------|----------|
| API 变更 | 函数签名、示例代码 | SKILL.md、Agent 内容 |
| 新增功能 | 新场景、新模式 | Common Usage Patterns |
| Bug 修复 | Common Pitfalls 表格 | Rules 文件 |
| 工作流变更 | Workflow 步骤 | Commands 文件 |
| 新模型支持 | Model Configuration | 相关 Agent/Command |

---

## 六、总结与最佳实践建议

### 6.1 核心原则

1. **Progressive Disclosure**：信息分层，按需加载
2. **分层解耦**：Agents/Skills/Commands/Rules 各司其职
3. **智能决策**：简单自动，复杂询问
4. **知识积累**：成功/失败案例持续沉淀

### 6.2 编写建议

#### Description 编写

- 明确"做什么"和"何时使用"
- 使用关键词：`PROACTIVELY`、`only when`、`Use when`
- 保持简洁，避免冗余

#### 内容组织

- 使用表格呈现结构化信息
- 提供代码示例
- 包含常见问题和解决方案
- 指向相关资源

### 6.3 常见陷阱

| 陷阱 | 问题 | 解决方案 |
|------|------|----------|
| **过度触发** | Agent 频繁激活，干扰正常工作 | 使用 `only when` 限制触发条件 |
| **Context 过长** | 信息过多，AI 抓不住重点 | 分层设计，按需加载 |
| **职责不清** | Agent/Skill/Command 边界模糊 | 明确各层职责，避免重叠 |
| **缺乏维护** | 文档过时，与代码不同步 | 定期更新，包含 MAINTAINER GUIDE |

### 6.4 进阶建议

1. **建立索引**：在 CLAUDE.md 中维护完整的路由表
2. **定期审查**：检查 agents/skills/commands/rules 是否过时
3. **收集反馈**：记录 AI 使用过程中的问题，持续优化
4. **版本管理**：使用 Git 管理配置文件，便于回滚和追踪
