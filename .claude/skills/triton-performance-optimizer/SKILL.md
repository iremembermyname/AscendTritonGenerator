---
name: triton-performance-optimizer
description: Triton算子性能优化指导。当需要提升算子性能、分析性能瓶颈、或应用Ascend优化技术时使用此skill。Use when optimizing performance, analyzing bottlenecks, or applying Ascend optimizations.
---

# Triton Performance Optimizer

昇腾（Ascend）NPU 上 Triton 算子深度性能优化技能，致力于实现用户要求的性能提升目标。

**核心目标**：将指定的 Triton 算子性能提升至少 **x 倍**（用户要求的性能提升），在满足要求的基础上，性能越高越好，追求极致性能。

**工作模式**：单算子优化模式。**禁止使用入图方式**来提升性能（模型侧会通过整网入图或 Piecewise 方式进行图优化，这里只关注单算子的独立优化）。

**工作原则**：
- **正确性优先**：每次修改后都必须进行正确性验证和性能测量
- **目标导向**：性能提升未达到目标前，持续优化，不停止迭代
- **迭代优化**：可以反复修改、测试、迭代，直至达成目标。修改 Triton 算子源代码前，务必备份，以便需要时恢复。
- **精准修改**：追求"手术级"的精准修改，避免引入新问题。

## When to Use

This skill is triggered when:
- User asks "optimize this code"
- User mentions performance, speed, latency
- After precision verification passes
- Performance target not met
- User mentions "昇腾NPU上Vector类Triton算子性能优化"

## Knowledge Retrieval

执行任务前，检索相关知识：
1. `@.claude/data/guides/optimization-tips.md` - 优化技巧
2. `@.claude/data/cases/optimization/` - 优化案例
3. `@.claude/rules/ascend-hardware.md` - 硬件约束
4. `@.claude/data/guides/troubleshooting.md` - 问题排查

## 工作流程

### Step 0: 环境配置

在昇腾 NPU 环境中，执行以下命令完成环境配置：

```bash
export LD_LIBRARY_PATH=/usr/local/Ascend/driver/lib64/driver:/usr/local/Ascend/driver/lib64/common:/usr/local/Ascend/driver/lib64:$LD_LIBRARY_PATH && source /usr/local/Ascend/ascend-toolkit/set_env.sh
```

### Step 1: 基线性能验证

1. **深入分析算子**：输入参数、数据类型、Shape 范围、功能逻辑、计算流程及输出结果
2. **运行功能测试**：`python -m pytest test_<op_name>.py`，验证算子的正确性和精度
3. **执行性能测试**：
   ```bash
   msprof op --output=<用户指定的路径> --kernel-name="<op_name>_kernel" --warm-up=20 --launch-count=20 python test_<op_name>_perf.py
   ```
   输出中的 `Task Duration(us)` 即为当前算子的耗时，记录为基线性能数据。

### Step 2: 深度性能优化

根据基线分析结果，对 `<op_name>.py` 算子进行针对性优化，确保性能提升至少 **x 倍**。

需运行以下测试：
- **性能测试**：`msprof op --output=<用户指定的路径> --kernel-name="<op_name>_kernel" --warm-up=20 --launch-count=20 python test_<op_name>_perf.py`
- **正确性验证**：`python -m pytest test_<op_name>.py`

### Step 3: 迭代调优

参考以下文档进行迭代调优：
- `@.claude/rules/ascend-hardware.md` - 硬件约束详解
- `@.claude/data/guides/troubleshooting.md` - 问题排查指南

---

## 性能优化参考

### 1. 多Token并行处理（首要优化点）

Ascend NPU 在架构上访存能力相对较弱，而计算能力较强，因此在设计时需要尽可能减少频繁的内存访问。**首要的关键优化点是批量处理多个 Tokens**，必须优先思考和调试，从而避免因逐个加载而产生的大量访存开销。

一次循环里能处理的**最大 Token 数 N**，由 Kernel 内 **UB 可用容量**决定：

**设：**
- 单 Kernel 内 UB 总容量为 **192 KB**
- 为留安全余量，仅使用 170 KB 的 **50%**（为确保启用 Double Buffering），即 **85 KB**
- 单个 Token 在 Kernel 内同时占用的 UB 空间峰值为 $S_{\text{token}}$

则需满足：$N \times S_{\text{token}} \le 85 \times 1024$

因此：$N \le \frac{85 \times 1024}{S_{\text{token}}}$

**计算最大处理量时应使用整数除法（//）而非 `tl.cdiv`，否则易引发 UB 溢出问题。**

### 2. UB占用计算公式

统计 kernel 循环体内所有变量的 UB 同时占用的峰值：

**需要计入的变量**：
1. `tl.load` 加载的所有 tensor
2. 计算过程中产生的中间 tensor
3. `tl.store` 暂存的输出 tensor
4. 类型转换后的变量（bf16 → float32 大小翻倍）

**计算公式**：
```
S_token = max(S_token_load, S_token_compute, S_token_store) + S_static

其中：
S_token_load = Σ(load_tensor_i × bytes_per_element_i)
S_token_compute = Σ(load_tensor_i × bytes_per_element_i) + Σ(intermediate_tensor_j × bytes_per_element_j)
S_token_store = Σ(store_tensor_k × bytes_per_element_k)
S_static = 循环体外加载到UB的权重等静态变量

N = 85 * 1024 // S_token  （使用整数除法）
```

**示例**：以 add_kernel 为例，BLOCK_SIZE 个元素，BF16 类型：
```
S_token_load = BLOCK_SIZE × 2 + BLOCK_SIZE × 2 = 4 × BLOCK_SIZE bytes
S_token_compute = BLOCK_SIZE × 2 + BLOCK_SIZE × 2 + BLOCK_SIZE × 2 = 6 × BLOCK_SIZE bytes
S_token_store = BLOCK_SIZE × 2 bytes
S_token = max(4, 6, 2) × BLOCK_SIZE = 6 × BLOCK_SIZE bytes
N = 85 × 1024 // (6 × BLOCK_SIZE)
```

### 3. 掩码（mask）与尾块处理

每次核函数加载和存储 tensor 时都需使用 `mask` 来处理不需要计算的尾块。经过 mask 处理后，每个核上的 tensor 形状保持一致。

### 4. 减少kernel内Scalar运算

将与 pid 或循环变量无关的计算移至辅助函数或循环外部；能合并的计算尽量合并，减少冗余操作。

### 5. 非连续地址访问

对于 `index_select` 这类涉及非连续地址访问的操作，只能通过循环逐行读取数据；否则会引入大量标量（Scalar）计算（计算二维 mask），严重影响性能。

### 6. 加载与计算交织

当需要多次从同一全局内存地址加载数据并进行计算时，需采用"加载一次、计算一次"的方式，而不是全部加载完再统一计算。前者可有效隐藏访存延迟。

### 7. 多写入流优化

若存在多个写入流，建议边计算边写入数据。写入流通常不会相互冲突，计算完提前写入可以增大并行的可能。

### 8. 使用tl.arange生成索引

使用 `tl.arange` 可以高效地生成二维 tensor 的索引，避免直接从全局内存中读取离散行数据所带来的大量 Scalar 计算。

### 9. 避免tl.where

尽量避免使用 `tl.where`，因其主要适用于离散数据处理，性能较差。当访问内存规则连续时，用 `tl.insert_slice` 代替。

### 10. 规约操作优化

执行规约操作时，优先选择最大的维度进行规约，有助于提升性能。

### 11. kernel入参声明

对于同一模型调用期间保持不变的参数，推荐声明为 `tl.constexpr` 编译期常量；对于可能变化的参数（如 `batch_size`、`seq_len` 等），则应使用普通动态参数传入。

---

## 分核策略

### 负载均衡分核

```python
core_num = get_vectorcore_num()
num_tokens = qkv.shape[0]

# front_core 处理多一个 Token，tail_core 处理少一个
front_core_num = core_num
if num_tokens % core_num != 0:
    front_core_num = num_tokens % core_num

num_tokens_each_front_core = (num_tokens + core_num - 1) // core_num

tail_core_num = 0
if num_tokens > core_num:
    tail_core_num = core_num - front_core_num

num_tokens_each_tail_core = num_tokens // core_num
```

**关键**：front_core 和 tail_core 的工作量差异不应超过 1 个 Token，以保持负载均衡。

---

## 需遵循的规则和约束

### 单算子模式

单个算子只关注单算子模式下的基础功能和性能，**禁止使用入图方式提升性能**。

### tl.load 与 mask 使用要求

- 尽量合并相同 load、计算和 store 操作
- 避免在 `tl.load` 中使用 other 参数，因为其内部会触发 `tl.where`，导致 load 后无法与其他 load 并行
- 推荐替代方案：先执行无掩码的 `tl.load`，再通过 `tl.where` 与 mask 组合实现掩码逻辑；当访问内存规则连续时，用 `tl.insert_slice` 代替

### 分支与编译约束

在 kernel 内部的 `if-else` 分支中，同名变量的 Shape 必须一致，否则会导致编译错误。

### 数据搬运注意事项

- 保证 tl.load 加载的是连续的多行数据；若数据分布离散，需逐行加载
- 传递给 Triton 算子的 tensor 必须是内存连续的，必要时可通过 `.contiguous()` 方法确保
- 避免复用 `tl.load` 和 `tl.store` 的变量名

---

## 结果报告

性能优化目标达成后，需输出标准化报告：

```markdown
## 优化结果报告

### 算子信息

- 算子名称：<op_name>
- 源文件：<file_path>

### 性能对比

| 基线耗时 (us) | 优化后耗时 (us) | 加速比 |
|-------------|---------------|-------|
| ... | ... | ...x |

### 优化技术清单

1. [已应用] 多个 Token 并行处理：N = ...
2. [已应用] 消除带 other 的 tl.load
3. ...

### 关键修改说明

- 修改点 1：...
- 修改点 2：...
```

---

## Common Mistakes

- ❌ 优化后不验证精度
- ❌ 一次应用多个优化
- ❌ 忽略硬件约束
- ❌ 使用 `tl.cdiv` 计算 N 值（应用 `//`）
- ❌ 带其他参数的 `tl.load`
- ❌ 过度优化简单算子

## Optimization Patterns

| Pattern | When to Use | Example |
|---------|-------------|---------|
| Double Buffering | MTE瓶颈 | Pipeline load and compute |
| 多Token并行 | 访存瓶颈 | 批量处理N个Token |
| Block Pointer | 规则访问 | `tl.make_block_ptr` |
| 加载计算交织 | 多次load | load一次compute一次 |
