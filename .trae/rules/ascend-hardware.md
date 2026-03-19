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
