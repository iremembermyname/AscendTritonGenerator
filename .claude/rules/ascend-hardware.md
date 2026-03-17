---
paths:
  - "**/ascend/**/*.py"
  - "**/*ascend*.py"
---

# Ascend Hardware Constraints

Ascend NPU硬件约束规则，自动应用于Ascend相关代码。

---

## 1. 整体架构

昇腾 NPU 的计算核心包含三大引擎：
- **Scalar**：标量计算（地址计算、循环控制、条件判断）
- **MTE（Memory Transfer Engine）**：数据搬运（GM ↔ UB）
- **Vector**：向量计算（算术运算、规约、类型转换等）

三者可以流水并行执行，是性能优化的关键。当任何一个引擎的操作阻塞了其他引擎，流水并行就被破坏。

---

## 2. UB容量限制

### 2.1 基本容量

| 芯片型号 | UB容量 | Double Buffering可用 | 建议使用量 |
|---------|--------|---------------------|-----------|
| 910B | 192 KB | 96 KB | ~85 KB |

### 2.2 硬性约束

- UB总容量：192KB
- 安全使用限制：≤85KB
- **单次循环UB占用必须 ≤ 85KB**（为启用Double Buffering）

### 2.3 Double Buffering 机制

**原理**：将 UB 分为两个 Buffer（A 和 B）：
- 当 Vector 在 Buffer A 中计算时，MTE 同时将下一批数据搬入 Buffer B
- 当 Vector 切换到 Buffer B 计算时，MTE 将 Buffer A 的结果搬出并加载新数据
- 如此交替，实现 MTE 与 Vector 的流水并行

**硬性约束**：单次循环的 UB 占用必须 ≤ 总容量的一半，否则 Double Buffering 无法工作。

```
单次循环 UB 占用 ≤ 192 KB // 2 = 96 KB
预留临时变量 → 建议 ≤ 85 KB
```

### 2.4 UB占用计算公式

统计 kernel 循环体内所有变量的 UB 同时占用的峰值：

**需要计入的变量**：
1. `tl.load` 加载的所有 tensor
2. 计算过程中产生的中间 tensor
3. `tl.store` 暂存的输出 tensor
4. 类型转换后的变量（bf16 → float32 大小翻倍）

**注意事项**：
- 不同数据类型占用不同：bf16 = 2 Bytes/元素，float32 = 4 Bytes/元素
- 对二维 tensor 使用一维索引和 mask 会额外占用大量 UB
- 二维 tensor 形状如 `(num_heads, head_size)` 需按完整形状计算

**计算公式**：
```
S_token = max(S_token_load, S_token_compute, S_token_store) + S_static

其中：
S_token_load = Σ(load_tensor_i × bytes_per_element_i)
S_token_compute = Σ(load_tensor_i × bytes_per_element_i) + Σ(intermediate_tensor_j × bytes_per_element_j)
S_token_store = Σ(store_tensor_k × bytes_per_element_k)
S_static = 循环体外加载到UB的权重等静态变量（较小时可忽略）

N = 85 * 1024 // S_token  （使用整数除法）
```

**计算示例**：
以 add_kernel 为例：
```python
@triton.jit
def add_kernel(x_ptr, y_ptr, out_ptr, n, BLOCK_SIZE: tl.constexpr): 
    pid = tl.program_id(0) 
    NUM_CORE = tl.num_programs(0)
    NUM_BLOCKS = tl.cdiv(n, BLOCK_SIZE)
    for block_idx in range(pid, NUM_BLOCKS, NUM_CORE):
        block_start = block_idx * BLOCK_SIZE
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n
        x = tl.load(x_ptr + offsets, mask=mask)
        y = tl.load(y_ptr + offsets, mask=mask)
        output = x + y
        tl.store(output_ptr + offsets, output, mask=mask)
```

- load 阶段：加载 `x` 和 `y`，UB占用峰值为 `BLOCK_SIZE × 2 + BLOCK_SIZE × 2 = 4 × BLOCK_SIZE` bytes
- add 阶段：`x`、`y` 和 `output` 同时存在，UB占用峰值为 `6 × BLOCK_SIZE` bytes
- store 阶段：仅 `output` 占用 UB，峰值为 `2 × BLOCK_SIZE` bytes

```
S_token = max(4, 6, 2) × BLOCK_SIZE = 6 × BLOCK_SIZE bytes
N = 85 × 1024 // (6 × BLOCK_SIZE)
```

---

## 3. Block大小限制

- 最大Block大小：1024
- 推荐Block大小：256
- 根据UB容量动态调整

---

## 4. MTE / Vector / Scalar 流水并行

### 4.1 理想流水

```
时间 →
Scalar: [addr1] [addr2] [addr3] ...
MTE:    [load1] [load2] [load3] ...
Vector:         [comp1] [comp2] ...
MTE:            [store1][store2]...
```

### 4.2 破坏流水的常见操作

| 操作 | 影响 | 替代方案 |
|------|------|---------|
| `tl.load` with mask | MTE 等待 Vector 生成 mask | mask 预计算 |
| `tl.load` with other | 内部调用 tl.where，阻止 load 并行 | 去掉 other，手动 tl.where |
| 大量 Scalar 计算 | Scalar 流水成为瓶颈 | 预计算、tl.arange 索引 |
| range() 循环 | 可能影响流水并行 | 确保循环体内 load/vector 可并行 |

### 4.3 流水线约束

- 避免带`other`参数的`tl.load`
- 确保循环迭代可独立执行
- 分离load和where操作

**错误示例**：
```python
x = tl.load(ptr + offsets, mask=mask, other=0.0)  # 影响流水线
```

**正确示例**：
```python
x = tl.load(ptr + offsets, mask=mask)
x = tl.where(mask, x, 0.0)  # 分离操作
```

### 4.4 检查流水是否正常

使用 msprof 工具采集 profiling 数据后，检查：
- MTE 和 Vector 是否有重叠执行区间
- Scalar 是否存在长时间独占执行
- 各引擎的利用率是否均衡

---

## 5. 核数约束

### 5.1 基本约束

- Vector核数：108
- grid大小建议不超过核数的2倍
- 避免过度分核

### 5.2 分核原则

- **负载均衡**：将输入数据尽量均匀分配给各 Vector 核
- **逻辑一致**：每个 program ID 对应的 kernel 处理逻辑相同
- **避免过度细分**：合理控制 grid 大小
- **循环补偿**：当分核数超过可用核心数时，在 kernel 内部通过循环处理

### 5.3 典型分核模式

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

## 6. 数据对齐与连续性

### 6.1 内存连续性要求

传入 Triton kernel 的所有 tensor 必须内存连续：
```python
tensor = tensor.contiguous()
```

### 6.2 连续加载

`tl.load` 应加载连续的多行数据。若数据分布离散（如经过 index_select），需逐行加载。

### 6.3 对齐

尽量保证加载的数据起始地址对齐（通常 32 Bytes 对齐），可提升 MTE 搬运效率。

---

## 7. 内存对齐

- 优先16字节对齐访问
- 避免跨缓存行访问
- 连续内存访问性能最优

---

## Common Pitfalls

| Issue | Cause | Fix |
|-------|-------|-----|
| UB溢出 | Block过大或中间变量多 | 减小BLOCK_SIZE或减少变量，使用 `//` 计算 N |
| 流水线不工作 | 带other的load | 分离load和where |
| 分核不均 | grid配置不当 | 调整grid大小，使用front_core/tail_core策略 |
| 性能差 | 内存访问不连续 | 优化访问模式，多Token并行处理 |
| 编译错误 | if分支内同名变量形状不一致 | 统一分支内变量形状 |

---

## Performance Tips

- Double Buffering：MTE与Vector并行
- 多Token并行：减少循环次数，批量处理
- Block Pointer：规则访问优化
- 加载计算交织：隐藏访存延迟
