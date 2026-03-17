# AscendTritonGenerator

面向 Ascend NPU 的 Triton 算子智能生成系统，支持算子开发、验证和优化的全流程。

## 功能特性

- **智能算子生成** - 领域专家生成高性能 Triton kernel
- **CUDA 到 Ascend 转换** - 自动将 CUDA Triton 代码迁移到 Ascend 平台
- **精度验证** - 提供完整的精度验证流程
- **性能分析与优化** - 针对 Ascend NPU 架构的深度优化

## 技术栈

- Triton
- PyTorch
- Ascend NPU (910B2)

## 系统架构

```
.claude/
├── agents/          # 领域专家 - 处理需要判断的复杂任务
│   ├── planner.md
│   └── triton-expert.md
├── skills/          # 固化流程 - 有明确步骤的操作指南
│   ├── verify-precision/
│   ├── profile-performance/
│   └── debug-kernel/
├── commands/        # 命令 - 快捷动作
│   └── add-knowledge.md
├── rules/           # 规则 - 代码约束
│   ├── ascend-hardware.md
│   └── triton-code.md
└── data/            # 知识库
    ├── syntax/
    ├── templates/
    ├── guides/
    └── cases/
```

## Agents（领域专家）

Agent 是"会思考的领域专家"，负责处理没有固定流程的复杂任务。

| Agent | 功能 | 触发场景 |
|-------|------|---------|
| planner | 任务规划专家 | 多步骤任务、架构决策 |
| triton-expert | 算子开发专家 | 算子生成、CUDA转换、性能优化 |

## Skills（固化流程）

Skill 是"有明确步骤的操作指南"，执行固定的操作流程。

| Skill | 功能 | 调用方式 |
|-------|------|---------|
| verify-precision | 精度验证流程 | `/verify-precision` |
| profile-performance | 性能分析流程 | `/profile-performance` |
| debug-kernel | 算子调试流程 | `/debug-kernel` |

## 设计理念

本项目遵循 **Agent = 领域专家，Skill = 固化流程** 的设计原则：

- **Agent**：加载相关知识，处理需要专家判断的复杂任务，可通过Task工具调用其他agent或skill
- **Skill**：提供有明确步骤的操作指南，执行固定的操作流程

详细信息请参阅 [设计理念文档](docs/design-principles.md)。

核心原则：
- **信息分层**：入口文件精简，细节按需加载
- **职责清晰**：Agent负责复杂决策，Skill负责固定流程
- **知识共享**：知识库是独立资源，Agent和Skill都可以引用

## 快速开始

### 环境要求

- Ascend NPU 环境 (910B2)
- Python 3.8+
- PyTorch + torch_npu
- Triton

### 环境配置

```bash
export LD_LIBRARY_PATH=/usr/local/Ascend/driver/lib64/driver:/usr/local/Ascend/driver/lib64/common:/usr/local/Ascend/driver/lib64:$LD_LIBRARY_PATH && source /usr/local/Ascend/ascend-toolkit/set_env.sh
```

## 典型工作流

### 新算子开发

```
用户需求 → planner agent (规划)
         → triton-expert agent (生成代码)
         → /verify-precision (验证精度)
         → /profile-performance (分析性能)
         → triton-expert agent (优化代码)
```

### CUDA转换

```
CUDA代码 → triton-expert agent (转换)
         → /verify-precision (验证精度)
```

### 问题调试

```
问题报告 → /debug-kernel (调试流程)
         → triton-expert agent (专家诊断，如需要)
         → /verify-precision (验证修复)
```

## 核心命令

### /add-knowledge

将新知识添加到知识库：

```bash
# 交互式添加
/add-knowledge

# 指定类型
/add-knowledge --type case

# 从文件读取
/add-knowledge --file ./my_knowledge.md
```

知识类型与存储位置：

| 类型 | 目标位置 | 说明 |
|------|---------|------|
| 语法文档 | `.claude/data/syntax/` | Triton 语法、Ascend 扩展 API |
| 模板文档 | `.claude/data/templates/` | 代码模板 |
| 指南文档 | `.claude/data/guides/` | 优化技巧、问题排查 |
| 精度案例 | `.claude/data/cases/precision/` | 精度问题案例 |
| 优化案例 | `.claude/data/cases/optimization/` | 性能优化案例 |
| 转换案例 | `.claude/data/cases/conversion/` | CUDA 转换案例 |

## 许可证

MIT License
