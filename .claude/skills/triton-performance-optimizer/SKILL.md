---
name: triton-performance-optimizer
description: Triton算子性能优化技能。当需要提升算子性能、分析性能瓶颈、应用Ascend优化技术时使用。触发场景包括：精度验证通过后需要优化、用户报告性能问题、用户提到"优化"、"性能"、"加速"、"耗时"、用户指定性能提升目标。Use when optimizing performance, analyzing bottlenecks, or applying Ascend optimizations.
---

# Triton Performance Optimizer

面向 Ascend NPU 的 Triton 算子深度性能优化技能，致力于实现用户要求的性能提升目标。

## 核心目标

将指定的 Triton 算子性能提升至少 **x 倍**（用户要求的性能提升），在满足要求的基础上，性能越高越好。

## 工作原则

- **正确性优先**：每次修改后都必须进行正确性验证和性能测量
- **目标导向**：性能提升未达到目标前，持续优化，不停止迭代
- **迭代优化**：可以反复修改、测试、迭代，直至达成目标
- **单算子模式**：禁止使用入图方式提升性能

## 工作流程

```
基线测量 → 瓶颈分析 → 知识检索 → 优化实施 → 验证迭代 → 输出报告
```

---

## Step 0: 环境配置

```bash
export LD_LIBRARY_PATH=/usr/local/Ascend/driver/lib64/driver:/usr/local/Ascend/driver/lib64/common:/usr/local/Ascend/driver/lib64:$LD_LIBRARY_PATH && source /usr/local/Ascend/ascend-toolkit/set_env.sh
```

---

## Step 1: 基线性能测量

### 1.1 算子分析

深入分析算子：
- 输入参数、数据类型、Shape 范围
- 功能逻辑、计算流程及输出结果

### 1.2 功能测试

```bash
python -m pytest test_<op_name>.py
```

### 1.3 性能测试

```bash
msprof op --output=<output_path> --kernel-name="<op_name>_kernel" --warm-up=20 --launch-count=20 python test_<op_name>_perf.py
```

**关键指标**: `Task Duration(us)` - 记录为基线性能数据

---

## Step 2: 知识检索

### 2.1 检索优化技巧指南

**文件**: `.claude/data/guides/optimization-tips.md`

**按优化目标检索**:

| 优化目标 | 检索章节 | 关键技术 |
|---------|---------|---------|
| 访存瓶颈 | §1 内存访问优化 | 连续访问、合并load、Block Pointer |
| UB溢出 | §2 UB容量优化 | 控制使用量、减少中间变量 |
| 流水线问题 | §3 流水线优化 | 避免other参数、加载计算交织 |
| 分核问题 | §5 分核优化 | 负载均衡、front_core/tail_core |
| 离散访存 | §7 离散访存优化 | tl.gather、高维离散低维连续 |
| Load顺序 | §8 Load顺序优化 | 提前无依赖Load |

### 2.2 检索优化案例

**目录**: `.claude/data/cases/optimization/`

**案例文件及适用场景**:

| 案例文件 | 适用场景 | 关键优化点 |
|---------|---------|-----------|
| `matmul_tuning.json` | 矩阵乘法优化 | 分块大小调整、FP32累加 |
| `discrete_memory_access.json` | 离散内存访问 | 访问模式优化 |
| `dtype_optimization.json` | 数据类型优化 | 类型转换优化 |
| `load_order_optimization.json` | Load顺序优化 | 依赖关系分析 |
| `tiling_strategy.json` | 分块策略 | Block大小调整 |
| `ub_overflow_handling.json` | UB溢出处理 | UB占用计算 |
| `high_dim_discrete_access.json` | 高维离散访问 | insert_slice/extract_slice |

**使用方式**:
1. 根据性能瓶颈症状匹配案例
2. 参考案例中的 `code_before` 和 `code_after`
3. 应用相同的优化技术

### 2.3 检索硬件约束

**文件**: `.claude/rules/ascend-hardware.md`

**关键约束**:
- UB容量：192KB，安全使用 ≤85KB
- Block大小：≤1024
- 核数：108个Vector核
- 流水线：MTE/Vector/Scalar并行

### 2.4 检索问题排查指南

**文件**: `.claude/data/guides/troubleshooting.md`

**按问题类型检索**:

| 问题类型 | 检索章节 |
|---------|---------|
| 流水异常 | §1 流水异常 |
| UB溢出 | §3 UB溢出 |
| 编译错误 | §4-5 if分支/constexpr |
| 性能退化 | §6 性能退化排查 |

### 2.5 检索 Ascend 扩展 API

**文件**: `.claude/data/syntax/ascend-extensions.md`

**优化相关 API**:
- Double Buffering (§2.1)
- 多Token并行处理 (§2.2)
- insert_slice (§3.4)
- extract_slice (§3.5)
- tl.gather (§3.6)
- care_padding (§3.7)

---

## Step 3: 瓶颈分析

### 3.1 使用 msprof 分析

```bash
msprof op --output=./profile --kernel-name="my_kernel" --warm-up=20 --launch-count=20 python test.py
```

### 3.2 关键指标分析

| 指标 | 说明 | 优化目标 |
|------|------|---------|
| Task Duration | 总执行时间 | 最小化 |
| MTE Utilization | MTE利用率 | 与Vector并行 |
| Vector Utilization | Vector利用率 | 最大化 |
| UB Usage | UB使用量 | ≤85KB |

### 3.3 瓶颈诊断表

| 症状 | 可能瓶颈 | 优化方向 |
|------|---------|---------|
| MTE利用率高，Vector低 | 访存瓶颈 | 多Token并行、减少load次数 |
| Vector利用率高，MTE低 | 计算瓶颈 | 优化计算逻辑 |
| 两者都不高 | 流水线问题 | 检查other参数、数据依赖 |
| UB Usage > 85KB | UB溢出 | 减少中间变量、调整Block大小 |

---

## Step 4: 优化实施

### 4.1 优化技术优先级

按优先级顺序应用优化：

**优先级1: 多Token并行处理**

```python
# 计算 N 值（使用整数除法！）
N = 85 * 1024 // S_token

for i in range(0, num_tokens, N):
    tokens = tl.load(ptr + i * stride + offsets)
    result = compute(tokens)
    tl.store(out_ptr + i * stride + offsets, result)
```

**优先级2: 消除带other的tl.load**

```python
# 错误
x = tl.load(ptr + offsets, mask=mask, other=0.0)

# 正确
x = tl.load(ptr + offsets, mask=mask)
x = tl.where(mask, x, 0.0)
```

**优先级3: 加载计算交织**

```python
for i in range(num_blocks):
    x = tl.load(x_ptr + offsets)
    y = compute(x)
    tl.store(y_ptr + offsets, y)
```

**优先级4: 分核优化**

```python
core_num = get_vectorcore_num()
front_core_num = num_tokens % core_num if num_tokens % core_num != 0 else core_num
num_tokens_each_front_core = (num_tokens + core_num - 1) // core_num
```

### 4.2 UB占用计算

**公式**:
```
S_token = max(S_load, S_compute, S_store) + S_static
N = 85 * 1024 // S_token  （使用整数除法！）
```

**示例**:
```python
# add_kernel: BLOCK_SIZE个元素，BF16类型
S_load = BLOCK_SIZE * 2 + BLOCK_SIZE * 2 = 4 * BLOCK_SIZE bytes
S_compute = BLOCK_SIZE * 2 + BLOCK_SIZE * 2 + BLOCK_SIZE * 2 = 6 * BLOCK_SIZE bytes
S_store = BLOCK_SIZE * 2 bytes
S_token = max(4, 6, 2) * BLOCK_SIZE = 6 * BLOCK_SIZE bytes
N = 85 * 1024 // (6 * BLOCK_SIZE)
```

### 4.3 常见优化模式

**离散访存优化**:
```python
# 使用 tl.gather 替代直接离散访问
x_shared = tl.load(x_ptr + tl.arange(0, M))
val = tl.gather(x_shared, idx, 0)
```

**Load顺序优化**:
```python
# 将无依赖的load提前
for i in range(HEAD_NUM):
    b_A = tl.load(p_A)  # 提前，可与上一轮store B并行
    idx_B = tl.load(p_B_index)
    b_B = tl.load(B + idx_B)
```

---

## Step 5: 验证迭代

### 5.1 每次优化后验证

```bash
# 1. 正确性验证
python -m pytest test_<op_name>.py

# 2. 性能测试
msprof op --output=<path> --kernel-name="<op_name>_kernel" --warm-up=20 --launch-count=20 python test_<op_name>_perf.py
```

### 5.2 迭代优化流程

```
应用优化 → 正确性验证 → 性能测试 → 达到目标？
                                        ↓是
                                    输出报告
                                        ↓否
                                    继续优化
```

### 5.3 性能退化排查

如果优化后性能变差：

1. 检查UB是否溢出
2. 检查流水线是否被破坏
3. 检查N值计算是否正确
4. 逐步回退定位问题

---

## Step 6: 输出报告

```markdown
## 优化结果报告

### 算子信息

- 算子名称: <op_name>
- 源文件: <file_path>

### 性能对比

| 指标 | 基线 | 优化后 | 提升 |
|------|------|--------|------|
| 耗时 (us) | ... | ... | ...x |
| MTE利用率 | ...% | ...% | - |
| Vector利用率 | ...% | ...% | - |

### 优化技术清单

1. [已应用] 多Token并行处理: N = ...
2. [已应用] 消除带other的tl.load
3. [已应用] 加载计算交织
4. ...

### 关键修改说明

- 修改点1: ...
- 修改点2: ...

### 代码对比

**优化前**:
```python
# code_before
```

**优化后**:
```python
# code_after
```
```

---

## 常见错误预防

| 错误 | 预防方法 |
|------|---------|
| 使用 `tl.cdiv` 计算 N | 必须使用 `//` 整数除法 |
| 带 other 的 tl.load | 分离 load 和 where |
| 过度分核 | grid大小接近物理核数 |
| 忽略边界条件 | 所有load/store带mask |

---

## 与其他 Skill 的协作

```
triton-code-generator (代码生成)
        ↓
triton-precision-verifier (精度验证)
        ↓ 验证通过
triton-performance-optimizer (性能优化) ← 当前
```

**优化完成后**:
- 输出优化报告
- 如需进一步优化，继续迭代
