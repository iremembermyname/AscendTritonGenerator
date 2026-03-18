# Cube 算子存储约束优化设计

## 概述

本设计文档描述如何修改 AscendTritonGenerator 项目，使其能够正确区分 Vector 算子和 Cube 算子的存储约束，避免在生成 Cube 类算子时错误地计算 UB 占用情况。

## 问题背景

### 当前问题

AscendTritonGenerator 项目中的 `triton-expert.md` agent 在工作时，即使生成 Cube 上的算子（如 matmul、attention），也会去计算 UB 的占用情况。但实际上：

- **Cube 引擎**使用 L0A、L0B、L0C、L1 cache 进行矩阵计算
- **UB** 是 Vector 单元的存储，用于向量运算

这导致了错误的优化建议和约束计算。

### 硬件架构差异

| 计算单元 | 存储层级 | 容量 | 用途 |
|---------|---------|------|------|
| Vector (VEC) | UB | 192KB | 向量运算（element-wise, softmax, layernorm） |
| Cube (CUBE) | L0A | 64KB | 左矩阵 A (m0×k0) |
| Cube (CUBE) | L0B | 64KB | 右矩阵 B (k0×n0) |
| Cube (CUBE) | L0C | 128KB | 结果矩阵 C (m0×n0) |
| Cube (CUBE) | L1 | 1MB | Cube 通用缓存 |

## 设计方案

### 1. 硬件约束文档改造

**文件**: `.claude/rules/ascend-hardware.md`

**修改内容**:
- 添加完整的存储系统架构说明
- 区分 Vector 算子和 Cube 算子的存储约束
- 添加 Cube 分块约束公式
- 添加算子类型识别指南

**新增内容**:

```markdown
## 存储系统架构

Ascend NPU 存储系统分为两大类：

### Vector 算子存储约束

Vector 算子使用 UB (Unified Buffer) 进行向量运算。

| 存储层级 | 容量 | 约束 |
|---------|------|------|
| UB | 192KB | 单次循环建议 ≤ 85KB（启用 Double Buffering） |

**适用算子**: element-wise, softmax, layernorm, reduce, activation 等

### Cube 算子存储约束

Cube 算子使用 L0 系列缓存进行矩阵计算。

| 存储层级 | 容量 | 用途 | 约束公式 |
|---------|------|------|---------|
| L0A | 64KB | 左矩阵 A | m0 × k0 × sizeof(dtype) ≤ 64KB |
| L0B | 64KB | 右矩阵 B | k0 × n0 × sizeof(dtype) ≤ 64KB |
| L0C | 128KB | 结果矩阵 C | m0 × n0 × sizeof(dtype) ≤ 128KB |
| L1 | 1MB | 通用缓存 | 用于数据复用 |

**适用算子**: matmul, attention, tl.dot 相关算子

### 算子类型识别

| 类型 | 特征 | 存储 | 核心数 |
|------|------|------|--------|
| Vector | 不使用 tl.dot | UB | VEC_CORE_NUM (40-48) |
| Cube | 使用 tl.dot | L0A/L0B/L0C/L1 | CUBE_CORE_NUM (20-24) |
| CV 混合 | tl.dot + 向量运算 | UB + L0 系列 | 需特殊处理 |

### Cube 分块推荐配置

| 转置情况 | 分块行宽 | 推荐配置 (FP16/BF16) |
|---------|---------|---------------------|
| A、B 都不转置 | K0 和 N0 | M0=128, K0=256, N0=256 |
| A 不转置，B 转置 | 都是 K0 | K0=256 |
| A、B 都转置 | M0 和 K0 | M0=256, K0=256, N0=128 |
```

### 2. Agent 逻辑改造

**文件**: `.claude/agents/triton-expert.md`

**修改内容**:
- 添加算子类型识别能力
- 根据算子类型选择正确的存储约束
- 更新知识库引用表

**新增章节**:

```markdown
## 算子类型识别与约束选择

### 类型识别规则

1. **Vector 类算子**
   - 特征: 不使用 `tl.dot`，纯向量运算
   - 存储: UB (192KB)
   - 核心数: `torch_npu.npu.npu_config.get_device_limit(0).get("vector_core_num", 40)`
   - 约束: 单次循环 UB 占用 ≤ 85KB

2. **Cube 类算子**
   - 特征: 使用 `tl.dot` 进行矩阵乘
   - 存储: L0A/L0B/L0C/L1
   - 核心数: `torch_npu.npu.npu_config.get_device_limit(0).get("cube_core_num", 20)`
   - 约束: L0A ≤ 64KB, L0B ≤ 64KB, L0C ≤ 128KB

3. **CV 混合算子**
   - 特征: 同时使用 `tl.dot` 和向量运算
   - 存储: 同时使用 UB 和 L0 系列缓存
   - 需要特殊优化策略（参考编译选项 multibuffer）

### 约束计算公式

**Vector 算子 UB 计算**:
```
S_token = max(S_load, S_compute, S_store) + S_static
N = 85 * 1024 // S_token
```

**Cube 算子 L0 计算**:
```
L0A_usage = BLOCK_M * BLOCK_K * sizeof(A.dtype)
L0B_usage = BLOCK_K * BLOCK_N * sizeof(B.dtype)
L0C_usage = BLOCK_M * BLOCK_N * sizeof(C.dtype)

约束: L0A ≤ 64KB, L0B ≤ 64KB, L0C ≤ 128KB
```

### 知识库引用更新

| 算子类型 | 知识库 | 路径 |
|---------|--------|------|
| Vector | 硬件约束 | `rules/ascend-hardware.md` (UB 章节) |
| Cube | 硬件约束 | `rules/ascend-hardware.md` (L0 章节) |
| Cube | 优化案例 | `data/cases/optimization/matmul_tuning.json` |
```

### 3. 优化指南改造

**文件**: `.claude/data/guides/optimization-guide.md`

**修改内容**:
- 将 "UB 容量优化" 改为 "存储容量优化"
- 区分 Vector 和 Cube 算子的优化策略
- 添加 Cube 分块优化章节

**修改后的章节结构**:

```markdown
## 2. 存储容量优化

### 2.1 Vector 算子 UB 优化

**原则**: 单次循环 UB 占用 ≤ 85KB。

### 2.2 Cube 算子 L0 优化

**原则**: 分块大小需满足 L0A/L0B/L0C 容量约束。

**约束公式**:
- L0A: BLOCK_M × BLOCK_K × sizeof(dtype) ≤ 64KB
- L0B: BLOCK_K × BLOCK_N × sizeof(dtype) ≤ 64KB
- L0C: BLOCK_M × BLOCK_N × sizeof(accumulator_dtype) ≤ 128KB

**推荐配置** (FP16/BF16):
- 不转置: BLOCK_M=128, BLOCK_K=256, BLOCK_N=256
- B 转置: BLOCK_K=256
- 都转置: BLOCK_M=256, BLOCK_K=256, BLOCK_N=128
```

### 4. 案例更新

**文件**: `.claude/data/cases/optimization/matmul_tuning.json`

**修改内容**:
- 修正问题描述，移除 UB 相关描述
- 添加 L0 约束计算说明
- 更新优化技术描述

**更新后的内容**:

```json
{
  "case_id": "matmul_tuning_20260316",
  "case_type": "optimization",
  "title": "MatMul算子性能优化",
  "problem": {
    "description": "MatMul算子性能不达标，需要优化分块策略以满足L0缓存约束",
    "symptoms": ["性能低于预期", "内存带宽利用率低"],
    "affected_apis": ["tl.dot", "tl.load", "tl.store"],
    "context": {
      "hardware": "ascend910b2",
      "shape": [4096, 4096],
      "dtype": "bfloat16",
      "baseline_time_ms": 5.0,
      "target_time_ms": 2.0
    }
  },
  "solution": {
    "description": "通过调整分块大小满足L0A/L0B/L0C约束，优化L1缓存复用",
    "techniques": [
      "调整BLOCK_M/BLOCK_N/BLOCK_K以满足L0A/L0B/L0C约束",
      "使用FP32累加器提高精度",
      "优化L1缓存复用"
    ],
    "code_before": "BLOCK_M: tl.constexpr = 128\nBLOCK_N: tl.constexpr = 128\nBLOCK_K: tl.constexpr = 32",
    "code_after": "BLOCK_M: tl.constexpr = 64\nBLOCK_N: tl.constexpr = 64\nBLOCK_K: tl.constexpr = 32\n\nacc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)\n\n// L0A: 64*32*2B = 4KB ≤ 64KB ✓\n// L0B: 32*64*2B = 4KB ≤ 64KB ✓\n// L0C: 64*64*4B = 16KB ≤ 128KB ✓",
    "steps": [
      "计算分块大小对应的L0缓存占用",
      "确保L0A/L0B/L0C均满足容量约束",
      "使用FP32累加器提高精度",
      "优化load/store顺序以利用L1缓存"
    ],
    "performance_after": "1.8 ms",
    "l0_constraints": {
      "L0A_formula": "BLOCK_M × BLOCK_K × sizeof(dtype)",
      "L0B_formula": "BLOCK_K × BLOCK_N × sizeof(dtype)",
      "L0C_formula": "BLOCK_M × BLOCK_N × sizeof(accumulator)",
      "example_calculation": {
        "L0A": "64 × 32 × 2B = 4KB",
        "L0B": "32 × 64 × 2B = 4KB",
        "L0C": "64 × 64 × 4B = 16KB"
      }
    }
  },
  "metadata": {
    "tags": ["matmul", "optimization", "performance", "ascend", "cube", "l0-cache"],
    "severity": "medium",
    "created_at": "2026-03-16",
    "updated_at": "2026-03-18",
    "usage_count": 0
  }
}
```

## 实现文件清单

| 文件 | 操作 | 说明 |
|------|------|------|
| `.claude/rules/ascend-hardware.md` | 修改 | 添加 Cube 存储约束 |
| `.claude/agents/triton-expert.md` | 修改 | 添加算子类型识别 |
| `.claude/data/guides/optimization-guide.md` | 修改 | 区分存储优化策略 |
| `.claude/data/cases/optimization/matmul_tuning.json` | 修改 | 修正 L0 约束描述 |

## 验证方法

1. 检查修改后的文档是否正确区分 Vector/Cube 算子
2. 验证 agent 是否能正确识别算子类型
3. 测试生成的 matmul 代码是否使用正确的约束

## 参考资料

- `akg/aikg/python/ai_kernel_generator/resources/docs/hardware/Ascend910B1.md`
- `akg/aikg/python/ai_kernel_generator/resources/docs/hardware/Ascend910B4.md`
- `triton-ascend/docs/zh/migration_guide/architecture_difference.md`
- `triton-ascend/docs/zh/examples/05_matrix_multiplication_example.md`
