# AscendTritonGenerator

面向 Ascend NPU 的 Triton 算子智能生成系统，支持算子开发、验证和优化的全流程。

## 功能特性

- **智能算子生成** - 基于 step-by-step 指导生成高性能 Triton kernel
- **CUDA 到 Ascend 转换** - 自动将 CUDA Triton 代码迁移到 Ascend 平台
- **精度验证** - 提供完整的精度验证流程和问题诊断
- **性能优化** - 针对 Ascend NPU 架构的深度优化指导

## 技术栈

- Triton
- PyTorch
- Ascend NPU (910B2)

## 系统架构

```
.claude/
├── agents/          # 智能体 - 复杂任务处理
│   ├── cuda-to-ascend-converter.md
│   └── planner.md
├── skills/          # 技能 - 操作指导
│   ├── triton-code-generator/
│   ├── triton-performance-optimizer/
│   └── triton-precision-verifier/
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

### Agents（智能体）

| Agent | 功能 | 触发场景 |
|-------|------|---------|
| planner | 任务规划专家 | 多步骤任务、新功能设计、架构决策 |
| cuda-to-ascend-converter | CUDA 转换专家 | CUDA 代码转换、GPU 到 NPU 迁移 |

### Skills（技能）

| Skill | 功能 | 触发场景 |
|-------|------|---------|
| triton-code-generator | 代码生成指导 | 生成 Triton kernel、实现算子 |
| triton-performance-optimizer | 性能优化指导 | 性能调优、瓶颈分析 |
| triton-precision-verifier | 精度验证指导 | 正确性验证、精度问题调试 |

## 设计理念

本项目采用 **分层架构 + Progressive Disclosure（渐进式信息披露）** 的设计理念，详细信息请参阅 [设计理念文档](docs/design-principles.md)。

核心原则：
- **信息分层**：入口文件精简，细节按需加载
- **分层解耦**：Agents/Skills/Commands/Rules 各司其职
- **智能决策**：简单自动处理，复杂询问用户

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
