# AscendTritonGenerator 设计理念

> 基于 Claude Agents 组织模式最佳实践的深度解析

## 一、核心设计原则

### 1.1 Agent vs Skill 的本质区别

**Agent = 领域专家**
- 加载相关知识，处理没有固定流程的复杂任务
- 需要专家判断和决策
- 可以通过Task工具调用其他agent或skill
- 例如：代码生成、性能优化、代码转换

**Skill = 固化流程**
- 有明确的、可枚举的步骤
- 步骤顺序相对固定
- 输出格式可预期
- 例如：验证精度、运行性能分析、调试流程

### 1.2 判断标准

| 任务性质 | 应该是 | 原因 |
|---------|--------|------|
| 需要判断算子类型、设计kernel结构 | Agent | 没有固定步骤，需要专家判断 |
| 需要分析瓶颈、选择优化技术 | Agent | 根据具体情况做决策 |
| 环境检查→测试生成→执行验证→输出报告 | Skill | 有明确的步骤序列 |
| 配置环境→运行工具→解读指标 | Skill | 有标准工具和解读步骤 |

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
│   （规则层）   │    │  （专家层）    │    │  （命令层）    │
│               │    │               │    │               │
│ 路径匹配加载   │    │ 领域专家判断   │    │ 用户直接调用   │
│ ascend-hw     │    │ planner       │    │ add-knowledge │
│ triton-code   │    │ triton-expert │    │               │
└───────────────┘    └───────────────┘    └───────────────┘
                              │
                              ▼
                     ┌───────────────┐
                     │   Skills/     │
                     │  （流程层）    │
                     │               │
                     │ 固化操作流程   │
                     │ verify-prec   │
                     │ profile-perf  │
                     │ debug-kernel  │
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
| **Agents** | 领域专家判断 | 任务描述 | 执行结果 | 可配置 |
| **Skills** | 固化操作流程 | 用户需求 | 流程报告 | 只读 |
| **Commands** | 执行特定动作 | 用户参数 | 动作结果 | 可配置 |

---

## 三、组件详解

### 3.1 Agents（领域专家）

#### 3.1.1 概念与定位

Agents 是"会思考的领域专家"，具有以下特点：

- **独立推理能力**：能够分析问题、制定计划、执行任务
- **领域专精**：每个 agent 专注于特定领域
- **工具权限可控**：可配置可用的工具集
- **模型可选**：可根据任务复杂度选择不同模型
- **可调用其他组件**：通过Task工具调用其他agent或skill

#### 3.1.2 当前 Agents

| Agent | 功能 | 触发方式 | 模型 |
|-------|------|----------|------|
| planner | 任务规划专家 | PROACTIVE | opus |
| triton-expert | 算子开发专家（生成/转换/优化） | 自动/手动 | sonnet |

#### 3.1.3 triton-expert 设计说明

triton-expert 是统一的算子开发专家，涵盖：

| 能力 | 说明 |
|------|------|
| 算子生成 | 根据需求生成 Triton kernel 代码 |
| CUDA转换 | 将 CUDA Triton 代码转换为 Ascend 兼容版本 |
| 性能优化 | 分析和优化算子性能 |

**为什么合并为一个专家？**
- 核心技能相同：都是生成和优化 Triton 代码
- 知识高度重叠：都需要 Ascend 平台知识
- 工作流连续：转换后通常需要优化

### 3.2 Skills（固化流程）

#### 3.2.1 概念与定位

Skills 是"固化操作流程"，具有以下特点：

- **Step-by-step 指导**：提供详细的操作步骤
- **步骤固定**：有明确的执行顺序
- **输出可预期**：结果格式标准化
- **只读性质**：不直接修改代码，只提供指导

#### 3.2.2 当前 Skills

| Skill | 功能 | 步骤流程 |
|-------|------|---------|
| verify-precision | 精度验证 | 环境检查→测试生成→执行验证→问题诊断→输出报告 |
| profile-performance | 性能分析 | 环境配置→msprof采集→指标解读→瓶颈定位→输出报告 |
| debug-kernel | 算子调试 | 问题分类→环境配置→诊断工具→根因定位→解决方案 |

#### 3.2.3 与 Agents 的区别

| 特性 | Agents | Skills |
|------|--------|--------|
| **定位** | 领域专家 | 操作流程 |
| **任务性质** | 需要判断的复杂任务 | 有固定步骤的操作 |
| **触发方式** | 自动/手动 | 用户调用或Agent调用 |
| **执行方式** | 独立推理执行 | 按步骤执行 |
| **工具权限** | 可配置（可写） | 只读 |
| **适用场景** | 代码生成、优化决策 | 验证、分析、调试 |

### 3.3 Commands（命令）

Commands 是"用户直接调用的动作"：

- **用户主动调用**：通过 `/command-name` 触发
- **执行特定操作**：有明确的输入输出
- **可配置工具权限**：可执行文件操作等

| Command | 功能 | 调用方式 |
|---------|------|----------|
| add-knowledge | 添加知识到知识库 | `/add-knowledge` |

### 3.4 Rules（规则）

Rules 是"代码质量标准"：

- **路径匹配加载**：根据文件路径自动激活
- **约束性规则**：定义必须遵守的规范
- **只读性质**：不执行操作，只提供约束

| Rule | 匹配路径 | 功能 |
|------|----------|------|
| ascend-hardware | `**/ascend/**/*.py`, `**/*ascend*.py` | Ascend 硬件约束 |
| triton-code | `**/*.py`, `**/triton/**/*.py` | Triton 代码规范 |

### 3.5 Data（知识库）

知识库是系统的"记忆"，是独立资源：

- **Agent 和 Skill 都可以引用**
- **存储各类参考信息**

```
data/
├── syntax/                    # 语法参考
│   ├── triton-syntax.md       # Triton 核心语法
│   └── ascend-extensions.md   # Ascend 扩展 API
├── templates/                 # 代码模板
│   └── code-templates.md      # 常用算子模板
├── guides/                    # 指南文档
│   ├── optimization-guide.md  # 性能优化指南
│   ├── precision-guide.md     # 精度问题指南
│   └── debugging-guide.md     # 调试指南（含troubleshooting）
└── cases/                     # 案例库
    ├── conversion/            # CUDA 转换案例
    ├── optimization/          # 性能优化案例
    └── precision/             # 精度问题案例
```

---

## 四、协作机制与工作流

### 4.1 Agent 调用 Skill

Agent 可以通过 Task 工具调用 Skill：

```
triton-expert agent (生成代码)
        ↓ 调用
verify-precision skill (验证精度)
        ↓ 返回结果
triton-expert agent (根据结果决定下一步)
```

### 4.2 典型工作流

#### 新算子开发

```
用户需求
    ↓
planner agent (分析任务，制定计划)
    ↓ 调用
triton-expert agent (生成代码)
    ↓ 调用
verify-precision skill (验证精度)
    ↓ 调用
profile-performance skill (分析性能)
    ↓
triton-expert agent (根据分析结果优化)
```

#### CUDA转换

```
CUDA代码
    ↓
triton-expert agent (转换代码)
    ↓ 调用
verify-precision skill (验证精度)
```

#### 问题调试

```
问题报告
    ↓
debug-kernel skill (执行调试流程)
    ↓ 如果需要深入分析
triton-expert agent (专家诊断和修复)
    ↓ 调用
verify-precision skill (验证修复)
```

### 4.3 任务路由表

| 任务类型 | 路由目标 | 触发条件 |
|---------|---------|---------|
| 算子生成 | triton-expert agent | "生成/创建/实现算子" |
| CUDA转换 | triton-expert agent | "转换CUDA/迁移代码" |
| 性能优化 | triton-expert agent | "优化性能/加速" |
| 精度验证 | /verify-precision | "验证精度/测试正确性" |
| 性能分析 | /profile-performance | "分析性能/性能瓶颈" |
| 问题调试 | /debug-kernel | "调试/NaN/Inf/错误" |
| 添加知识 | /add-knowledge command | "添加知识" |

---

## 五、维护与更新

### 5.1 MAINTAINER GUIDE 设计

每个组件文件末尾应包含 MAINTAINER GUIDE：

```markdown
______________________________________________________________________

<!--
================================================================================
                            MAINTAINER GUIDE
================================================================================

Location: .claude/agents/triton-expert.md
Activation: When operator generation, CUDA conversion, or performance optimization detected

## Design Philosophy

- **Full-Stack Expert**: Handles generation, conversion, and optimization
- **Knowledge-Driven**: Actively references templates, cases, and guides
- **Model**: Sonnet (balance between capability and cost)

## How to Update

### When Adding New Operator Templates
1. Add to `data/templates/code-templates.md`
2. Reference in "算子生成" section

================================================================================
-->
```

### 5.2 更新触发条件

| 触发条件 | 更新内容 | 更新位置 |
|----------|----------|----------|
| API 变更 | 函数签名、示例代码 | Agent/Skill 内容 |
| 新增功能 | 新场景、新模式 | Common Patterns |
| Bug 修复 | Common Pitfalls 表格 | Rules 文件 |
| 工作流变更 | Workflow 步骤 | Skills 文件 |

---

## 六、总结

### 6.1 核心原则

1. **Agent是专家，不是执行器**：Agent负责需要判断的复杂任务
2. **Skill是流程，不是知识库**：Skill提供步骤指南，知识库是独立资源
3. **Agent可以调用Skill**：专家可以使用固化流程完成子任务
4. **知识库是共享资源**：Agent和Skill都可以引用知识库

### 6.2 常见陷阱

| 陷阱 | 问题 | 解决方案 |
|------|------|----------|
| **Skill被误用为Agent** | 代码生成/优化被设计为Skill | 这些需要专家判断，应该是Agent |
| **职责不清** | Agent/Skill边界模糊 | 明确：需要判断=Agent，固定步骤=Skill |
| **知识库绑定错误** | 知识库只被Skill引用 | 知识库是共享资源，Agent和Skill都可以引用 |
| **过度拆分Agent** | 多个Agent职责重叠 | 合并为统一的领域专家 |
