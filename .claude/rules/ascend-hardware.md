---
paths:
  - "**/ascend/**/*.py"
  - "**/*ascend*.py"
---

# Ascend 硬件约束

Ascend NPU硬件约束规则，自动应用于Ascend相关代码。

---

## 1. 整体架构

昇腾 NPU 的Vector计算涉及三大引擎：
- **Scalar**：标量计算（地址计算、循环控制、条件判断）
- **MTE（Memory Transfer Engine）**：数据搬运（GM ↔ UB）
- **Vector**：向量计算（算术运算、规约、类型转换等）

三者可以流水并行执行，是性能优化的关键。
此外，Ascend NPU 还有 Cube 引擎，用于处理矩阵计算（如 tl.dot）。

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

## 3. UB容量限制

| 芯片型号 | UB容量 | Double Buffering可用 | 建议使用量 |
|---------|--------|---------------------|-----------|
| 910B | 192 KB | 96 KB | ~85 KB |

**硬性约束**：单次循环UB占用必须 ≤ 85KB（为启用Double Buffering）

### UB占用计算公式

```
S_token = max(S_token_load, S_token_compute, S_token_store) + S_static

其中：
S_token_load = Σ(load_tensor_i × bytes_per_element_i)
S_token_compute = Σ(load_tensor_i × bytes_per_element_i) + Σ(intermediate_tensor_j × bytes_per_element_j)
S_token_store = Σ(store_tensor_k × bytes_per_element_k)
S_static = 循环体外加载到UB的权重等静态变量（较小时可忽略）

N = 85 * 1024 // S_token  （使用整数除法）
```

---

## 4. Block大小限制

- BLOCK_SIZE 必须 < 65536
- 线程块所占内存必须符合硬件限制
- 若shape过大，可对循环进行多次切分

---

## 5. 流水并行约束

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

## 6. 数据对齐与连续性

- 传入 Triton kernel 的所有 tensor 必须内存连续
- 尽量保证加载的数据起始地址对齐（256 Bytes）
- `tl.load` 应加载连续的多行数据

---

## Common Pitfalls

| Issue | Cause | Fix |
|-------|-------|-----|
| UB溢出 | Block过大或中间变量多 | 减小BLOCK_SIZE或减少变量 |
| 流水线不工作 | 带other的load | 分离load和where |
| 分核不均 | grid配置不当 | 调整grid大小，使用front_core/tail_core策略 |
| 性能差 | 内存访问不连续 | 优化访问模式 |
