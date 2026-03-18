# Cube 存储约束优化实现计划

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 修改 AscendTritonGenerator 项目，使其正确区分 Vector 和 Cube 算子的存储约束。

**Architecture:** 通过修改硬件约束文档、agent 逻辑、优化指南和案例文件，建立完整的 Cube/Vector 分类体系。

**Tech Stack:** Markdown 文档, JSON 配置, Triton kernel 开发

---

## Task 1: 修改硬件约束文档

**Files:**
- Modify: `d:\项目\trae\triton_gen\AscendTritonGenerator\.claude\rules\ascend-hardware.md`

**Step 1: 备份原文件**

```bash
cp "d:\项目\trae\triton_gen\AscendTritonGenerator\.claude\rules\ascend-hardware.md" "d:\项目\trae\triton_gen\AscendTritonGenerator\.claude\rules\ascend-hardware.md.bak"
```

**Step 2: 重写硬件约束文档**

将文件内容替换为：

```markdown
---
paths:
  - "**/ascend/**/*.py"
  - "**/*ascend*.py"
---

# Ascend 硬件约束

Ascend NPU硬件约束规则，自动应用于Ascend相关代码。

---

## 1. 整体架构

昇腾 NPU 包含两大计算引擎：

### Vector 引擎
- **Scalar**：标量计算（地址计算、循环控制、条件判断）
- **MTE（Memory Transfer Engine）**：数据搬运（GM ↔ UB）
- **Vector**：向量计算（算术运算、规约、类型转换等）

三者可以流水并行执行，是性能优化的关键。

### Cube 引擎
- **CUBE**：矩阵计算单元（tl.dot 操作）
- 使用独立的 L0A/L0B/L0C/L1 存储系统

---

## 2. 核数约束

| 型号 | AI Core | VEC | CUBE | L2 Cache | GM |
|------|---------|-----|------|----------|-----|
| 910B1 | 24 | 48 | 24 | 192MB | 64GB |
| 910B2 | 24 | 48 | 24 | 192MB | 64GB |
| 910B3 | 20 | 40 | 20 | 192MB | 64GB |
| 910B4 | 20 | 40 | 20 | 96MB | 32GB |

**分核原则**：
- 负载均衡：将输入数据尽量均匀分配给各核
- 逻辑一致：每个 program ID 对应的 kernel 处理逻辑相同
- 避免过度细分：合理控制 grid 大小

---

## 3. 存储系统架构

### 3.1 Vector 算子存储约束

Vector 算子使用 UB (Unified Buffer) 进行向量运算。

| 存储层级 | 容量 | 共享范围 | 对齐 | 说明 |
|---------|------|---------|------|------|
| UB | 192KB | 单VEC | 256B | 向量运算缓存 |

**硬性约束**：单次循环UB占用必须 ≤ 85KB（为启用Double Buffering）

**UB占用计算公式**：
```
S_token = max(S_token_load, S_token_compute, S_token_store) + S_static

其中：
S_token_load = Σ(load_tensor_i × bytes_per_element_i)
S_token_compute = Σ(load_tensor_i × bytes_per_element_i) + Σ(intermediate_tensor_j × bytes_per_element_j)
S_token_store = Σ(store_tensor_k × bytes_per_element_k)
S_static = 循环体外加载到UB的权重等静态变量（较小时可忽略）

N = 85 * 1024 // S_token  （使用整数除法）
```

**适用算子**: element-wise, softmax, layernorm, reduce, activation 等

### 3.2 Cube 算子存储约束

Cube 算子使用 L0 系列缓存进行矩阵计算。

| 存储层级 | 容量 | 共享范围 | 对齐 | 用途 |
|---------|------|---------|------|------|
| L1 Buffer | 1MB | 单AI Core | 256B | Cube通用缓存 |
| L0A | 64KB | 单Cube | 256B | 左矩阵A (m0×k0) |
| L0B | 64KB | 单Cube | 256B | 右矩阵B (k0×n0) |
| L0C | 128KB | 单Cube | 256B | 结果矩阵C (m0×n0)，支持累加 |

**Cube 分块约束公式**：
```
L0A约束: m0 × k0 × sizeof(A.dtype) ≤ 64KB
L0B约束: k0 × n0 × sizeof(B.dtype) ≤ 64KB
L0C约束: m0 × n0 × sizeof(C.dtype) ≤ 128KB
```

**适用算子**: matmul, attention, tl.dot 相关算子

### 3.3 数据通路

| 通路 | 方向 | 说明 |
|------|------|------|
| MTE1 | L1 → L0A/L0B | Cube 数据加载 |
| MTE2 | GM → UB/L1/L0A/L0B | 全局内存加载 |
| MTE3 | UB → GM, L1 → L2 | 数据写回 |
| FixP | L0C → L1/GM | Cube 结果输出（可随路类型转换） |

---

## 4. 算子类型识别

| 类型 | 特征 | 存储 | 核心数获取方式 |
|------|------|------|---------------|
| Vector | 不使用 tl.dot | UB | `vector_core_num` (40-48) |
| Cube | 使用 tl.dot | L0A/L0B/L0C/L1 | `cube_core_num` (20-24) |
| CV 混合 | tl.dot + 向量运算 | UB + L0 系列 | 需特殊处理 |

**核心数获取代码**：
```python
import torch_npu

# Vector 算子核心数
VEC_CORE_NUM = torch_npu.npu.npu_config.get_device_limit(0).get("vector_core_num", 40)

# Cube 算子核心数
CUBE_CORE_NUM = torch_npu.npu.npu_config.get_device_limit(0).get("cube_core_num", 20)
```

---

## 5. Cube 分块推荐配置

### 5.1 分块策略

合理的切分是提升 Cube 算子性能的关键。行宽为 512B 的整数倍，且单次行数尽量大。

| 转置情况 | 分块行宽 | 推荐配置 (FP16/BF16) |
|---------|---------|---------------------|
| A、B 都不转置 | K0 和 N0 | M0=128, K0=256, N0=256 |
| A 不转置，B 转置 | 都是 K0 | K0=256 |
| A、B 都转置 | M0 和 K0 | M0=256, K0=256, N0=128 |
| A 转置，B 不转置 | M0 和 N0 | 需根据实际情况调整 |

### 5.2 分块约束验证示例

```python
# FP16/BF16 示例 (dtype_size = 2 bytes)
BLOCK_M = 128
BLOCK_K = 256
BLOCK_N = 256

# L0A: 左矩阵 A (BLOCK_M × BLOCK_K)
L0A_usage = BLOCK_M * BLOCK_K * 2  # 128 * 256 * 2 = 65536 bytes = 64KB ✓

# L0B: 右矩阵 B (BLOCK_K × BLOCK_N)
L0B_usage = BLOCK_K * BLOCK_N * 2  # 256 * 256 * 2 = 131072 bytes = 128KB ✗

# 需要调整 BLOCK_N
BLOCK_N = 128  # 调整后
L0B_usage = BLOCK_K * BLOCK_N * 2  # 256 * 128 * 2 = 65536 bytes = 64KB ✓

# L0C: 结果矩阵 (BLOCK_M × BLOCK_N)，使用 FP32 累加
L0C_usage = BLOCK_M * BLOCK_N * 4  # 128 * 128 * 4 = 65536 bytes = 64KB ✓
```

---

## 6. Block大小限制

- BLOCK_SIZE 必须 < 65536
- 线程块所占内存必须符合硬件限制
- 若shape过大，可对循环进行多次切分

---

## 7. 流水并行约束

| 操作 | 影响 | 替代方案 |
|------|------|---------|
| `tl.load` with mask | MTE 等待 Vector 生成 mask | mask 预计算 |
| `tl.load` with other | 内部调用 tl.where，阻止 load 并行 | 去掉 other，手动 tl.where |
| 大量 Scalar 计算 | Scalar 流水成为瓶颈 | 预计算、tl.arange 索引 |

**错误示例**：
```python
x = tl.load(ptr + offsets, mask=mask, other=0.0)  # 影响流水线
```

**正确示例**：
```python
x = tl.load(ptr + offsets, mask=mask)
x = tl.where(mask, x, 0.0)  # 分离操作
```

---

## 8. 数据对齐与连续性

- 传入 Triton kernel 的所有 tensor 必须内存连续
- 尽量保证加载的数据起始地址对齐（256 Bytes）
- `tl.load` 应加载连续的多行数据

---

## Common Pitfalls

| Issue | Cause | Fix |
|-------|-------|-----|
| UB溢出 (Vector算子) | Block过大或中间变量多 | 减小BLOCK_SIZE或减少变量 |
| L0溢出 (Cube算子) | 分块超过L0A/L0B/L0C容量 | 调整BLOCK_M/N/K满足约束 |
| 流水线不工作 | 带other的load | 分离load和where |
| 分核不均 | grid配置不当 | 调整grid大小，使用front_core/tail_core策略 |
| 性能差 | 内存访问不连续 | 优化访问模式 |
| 算子类型识别错误 | 混淆Vector和Cube约束 | 检查是否使用tl.dot |
```

**Step 3: 验证文件修改**

```bash
cat "d:\项目\trae\triton_gen\AscendTritonGenerator\.claude\rules\ascend-hardware.md" | head -50
```

Expected: 文件内容已更新，包含 Cube 存储约束章节

---

## Task 2: 修改 Agent 逻辑

**Files:**
- Modify: `d:\项目\trae\triton_gen\AscendTritonGenerator\.claude\agents\triton-expert.md`

**Step 1: 读取当前文件内容**

```bash
cat "d:\项目\trae\triton_gen\AscendTritonGenerator\.claude\agents\triton-expert.md"
```

**Step 2: 在 "Core Capabilities" 章节前添加算子类型识别章节**

在 `## Core Capabilities` 之前插入：

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

---
```

**Step 3: 更新知识库引用表**

将原有的知识库引用表更新为：

```markdown
### 1. 算子生成

根据用户需求生成Triton kernel代码：

**流程**：
1. 分析需求（算子类型、输入输出、计算逻辑）
2. **识别算子类型**（Vector/Cube/CV混合）
3. 检索知识库（templates、syntax、cases）
4. **根据算子类型选择正确的存储约束**
5. 生成kernel和host函数
6. 生成测试代码

**知识库引用**：
| 知识库 | 路径 | 用途 |
|--------|------|------|
| 代码模板 | `data/templates/code-templates.md` | 参考实现模式 |
| Triton语法 | `data/syntax/triton-syntax.md` | API参考 |
| Ascend扩展 | `data/syntax/ascend-extensions.md` | 平台特定API |
| 硬件约束 | `rules/ascend-hardware.md` | UB/L0存储限制 |
```

**Step 4: 更新 Hardware Constraints 章节**

将原有的 Hardware Constraints 章节更新为：

```markdown
## Hardware Constraints

参考: `rules/ascend-hardware.md`

### Vector 算子约束

| 约束 | 限制 | 影响 |
|------|------|------|
| UB容量 | ≤ 85KB/循环 | 控制BLOCK_SIZE和变量数 |
| Block大小 | < 65536 | 最大元素数 |
| AI Core | 20-24（物理核） | Vector-only算子的Grid规划 |

### Cube 算子约束

| 约束 | 限制 | 影响 |
|------|------|------|
| L0A容量 | ≤ 64KB | BLOCK_M × BLOCK_K × dtype_size |
| L0B容量 | ≤ 64KB | BLOCK_K × BLOCK_N × dtype_size |
| L0C容量 | ≤ 128KB | BLOCK_M × BLOCK_N × accumulator_size |
| L1容量 | 1MB | 数据复用和缓存 |
| Cube Core | 20-24 | tl.dot算子的Grid规划 |
```

---

## Task 3: 修改优化指南

**Files:**
- Modify: `d:\项目\trae\triton_gen\AscendTritonGenerator\.claude\data\guides\optimization-guide.md`

**Step 1: 修改 "UB容量优化" 章节标题和内容**

将 `## 2. UB容量优化` 改为 `## 2. 存储容量优化`，并更新内容：

```markdown
## 2. 存储容量优化

### 2.1 Vector 算子 UB 优化

**原则**：单次循环 UB 占用 ≤ 85KB。

```python
# 计算 UB 占用
def calculate_ub(block_size, num_tensors, dtype_bytes=2):
    return block_size * num_tensors * dtype_bytes / 1024  # KB
```

### 2.2 Cube 算子 L0 优化

**原则**：分块大小需满足 L0A/L0B/L0C 容量约束。

**约束公式**：
- L0A: BLOCK_M × BLOCK_K × sizeof(dtype) ≤ 64KB
- L0B: BLOCK_K × BLOCK_N × sizeof(dtype) ≤ 64KB
- L0C: BLOCK_M × BLOCK_N × sizeof(accumulator_dtype) ≤ 128KB

**推荐配置** (FP16/BF16)：
- 不转置: BLOCK_M=128, BLOCK_K=256, BLOCK_N=256
- B 转置: BLOCK_K=256
- 都转置: BLOCK_M=256, BLOCK_K=256, BLOCK_N=128

```python
# 验证 Cube 分块约束
def validate_cube_blocks(BLOCK_M, BLOCK_K, BLOCK_N, dtype_size=2, acc_size=4):
    L0A = BLOCK_M * BLOCK_K * dtype_size
    L0B = BLOCK_K * BLOCK_N * dtype_size
    L0C = BLOCK_M * BLOCK_N * acc_size
    
    assert L0A <= 64 * 1024, f"L0A overflow: {L0A} > 64KB"
    assert L0B <= 64 * 1024, f"L0B overflow: {L0B} > 64KB"
    assert L0C <= 128 * 1024, f"L0C overflow: {L0C} > 128KB"
    return True
```

### 2.3 减少中间变量

```python
# 差：多个中间变量
a = x + y
b = a * z
c = b - w
result = c / v

# 好：合并计算
result = ((x + y) * z - w) / v
```

### 2.4 及时释放变量

```python
# 好：边计算边store
for i in range(num_blocks):
    block = compute_block(i)
    tl.store(output_ptr + offsets, block)  # 及时释放
```
```

**Step 2: 更新 "特定算子优化" 章节**

将 `### 7.1 Matmul切分优化` 更新为：

```markdown
### 7.1 Matmul切分优化

合理的切分是提升 matmul 算子性能的关键，需满足 L0 缓存约束：

| 转置情况 | 分块行宽 | 推荐配置 |
|---------|---------|---------|
| A、B都不转置 | K0和N0 | M0=128, K0=256, N0=256 |
| A不转置，B转置 | 都是K0 | K0=256 |
| A、B都转置 | M0和K0 | M0=256, K0=256, N0=128 |

**L0 约束验证**：
```python
# FP16 示例 (dtype_size = 2 bytes, acc_size = 4 bytes)
BLOCK_M, BLOCK_K, BLOCK_N = 128, 256, 128

# L0A: 128 * 256 * 2 = 64KB ✓
# L0B: 256 * 128 * 2 = 64KB ✓
# L0C: 128 * 128 * 4 = 64KB ✓
```
```

---

## Task 4: 更新 Matmul 优化案例

**Files:**
- Modify: `d:\项目\trae\triton_gen\AscendTritonGenerator\.claude\data\cases\optimization\matmul_tuning.json`

**Step 1: 重写 JSON 文件**

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
    "code_after": "BLOCK_M: tl.constexpr = 64\nBLOCK_N: tl.constexpr = 64\nBLOCK_K: tl.constexpr = 32\n\nacc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)\n\n# L0A: 64*32*2B = 4KB <= 64KB\n# L0B: 32*64*2B = 4KB <= 64KB\n# L0C: 64*64*4B = 16KB <= 128KB",
    "steps": [
      "计算分块大小对应的L0缓存占用",
      "确保L0A/L0B/L0C均满足容量约束",
      "使用FP32累加器提高精度",
      "优化load/store顺序以利用L1缓存"
    ],
    "performance_after": "1.8 ms",
    "l0_constraints": {
      "L0A_formula": "BLOCK_M * BLOCK_K * sizeof(dtype)",
      "L0B_formula": "BLOCK_K * BLOCK_N * sizeof(dtype)",
      "L0C_formula": "BLOCK_M * BLOCK_N * sizeof(accumulator)",
      "example_calculation": {
        "L0A": "64 * 32 * 2B = 4KB",
        "L0B": "32 * 64 * 2B = 4KB",
        "L0C": "64 * 64 * 4B = 16KB"
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

---

## Task 5: 验证修改

**Step 1: 检查所有修改的文件**

```bash
ls -la "d:\项目\trae\triton_gen\AscendTritonGenerator\.claude\rules\ascend-hardware.md"
ls -la "d:\项目\trae\triton_gen\AscendTritonGenerator\.claude\agents\triton-expert.md"
ls -la "d:\项目\trae\triton_gen\AscendTritonGenerator\.claude\data\guides\optimization-guide.md"
ls -la "d:\项目\trae\triton_gen\AscendTritonGenerator\.claude\data\cases\optimization\matmul_tuning.json"
```

**Step 2: 验证 JSON 文件格式**

```bash
python -c "import json; json.load(open(r'd:\项目\trae\triton_gen\AscendTritonGenerator\.claude\data\cases\optimization\matmul_tuning.json'))"
```

Expected: 无错误输出

**Step 3: 提交修改**

```bash
cd "d:\项目\trae\triton_gen\AscendTritonGenerator"
git add .claude/rules/ascend-hardware.md
git add .claude/agents/triton-expert.md
git add .claude/data/guides/optimization-guide.md
git add .claude/data/cases/optimization/matmul_tuning.json
git add docs/plans/2026-03-18-cube-storage-constraints-design.md
git commit -m "feat: 区分 Vector 和 Cube 算子的存储约束

- 更新 ascend-hardware.md，添加 Cube 存储约束（L0A/L0B/L0C/L1）
- 更新 triton-expert.md，添加算子类型识别能力
- 更新 optimization-guide.md，区分存储优化策略
- 更新 matmul_tuning.json，修正 L0 约束描述
- 添加设计文档"
```

---

## Summary

| Task | File | Description |
|------|------|-------------|
| 1 | `ascend-hardware.md` | 添加 Cube 存储约束章节 |
| 2 | `triton-expert.md` | 添加算子类型识别能力 |
| 3 | `optimization-guide.md` | 区分存储优化策略 |
| 4 | `matmul_tuning.json` | 修正 L0 约束描述 |
| 5 | - | 验证并提交修改 |
