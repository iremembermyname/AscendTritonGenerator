---
paths:
  - "**/*.py"
  - "**/triton/**/*.py"
---

# Triton 代码规范

Triton代码规范，自动应用于所有Python文件。

---

## 1. Kernel设计规范

- 使用`tl.constexpr`声明编译期常量
- 保持kernel函数简洁，单一职责
- 输出张量使用 `torch.empty` 或 `torch.empty_like` 创建

---

## 2. 禁用语法

| 禁用语法 | 替代方案 |
|---------|---------|
| `return` | 使用mask控制流程 |
| `break` | 使用mask控制流程 |
| `continue` | 使用mask控制流程 |
| `lambda`表达式 | 使用内联函数或tl.where |
| 链式布尔运算 | 分步计算mask |
| 张量直接索引 | 使用tl.load/tl.store |
| `while` 循环 | 使用 for + if 替代 |

### while循环替代方案

```python
# 错误：while 循环
i = 0
while i < n_iters:
    # 处理逻辑
    i += 1

# 正确：for + if 替代
for i in range(MAX_ITERS):
    if i < n_iters:
        # 处理逻辑
```

---

## 3. constexpr 正确用法

- **仅在内核参数中使用**: `BLOCK_SIZE: tl.constexpr`
- **不可在host侧使用**: 启动函数中不可用tl.constexpr

```python
@triton.jit
def my_kernel(
    data_ptr,
    num_tokens,  # 可以是动态值
    hidden_size: tl.constexpr,  # 用于 reshape/arange 等
):
    data = tl.load(data_ptr + tl.arange(0, hidden_size))
```

---

## 4. Grid设置规范

- **维度限制**：grid必须是tuple类型，最多3维
- **大小限制**：各维度乘积不超过65535
- **大shape处理**：使用交错循环 `for i in range(pid, total, core_num)`

```python
# 推荐：固定核心数启动
grid = (num_cores,)
kernel[grid](...)

# kernel内部交错处理
for i in range(pid, total_items, core_num):
    # 处理逻辑
```

---

## 5. 切片操作规范

Triton不支持Python风格的直接切片语法：

| 操作 | API |
|------|-----|
| 单元素提取 | `tl.get_element(tensor, (index,))` |
| 切片提取 | `tl.extract_slice(tensor, offsets, sizes, strides)` |
| 切片插入 | `tl.insert_slice(ful, sub, offsets, sizes, strides)` |

**重要限制**：禁止对`tl.arange`生成的张量使用`get_element()`

```python
# 错误：offsets = base + tl.arange(0, BLOCK_SIZE); value = tl.get_element(offsets, [i])
# 正确：value = base + i
```

---

## 6. 标量类型转换

- **仅支持to(type)**: 如`scalar.to(tl.float16)`
- **禁止使用**: `tl.float16(scalar)`

---

## 7. 内存访问规范

- 优先连续内存访问
- 使用mask处理边界条件
- 避免离散访问模式
- 合并对同一地址的多次load

---

## 8. 数值稳定性

- 使用float32进行中间计算
- 减最大值防止exp溢出
- 检查除零和NaN
- 避免大数相减

---

## Common Pitfalls

| Issue | Cause | Fix |
|-------|-------|-----|
| 编译失败 | 使用了return/break/continue | 移除禁用语句，使用mask控制 |
| Grid超限 | grid总大小超过65535 | 使用交错循环处理 |
| 切片语法错误 | 使用了`b[0]`或`b[i:j]` | 使用`tl.get_element`或`tl.extract_slice` |
| 类型转换错误 | 使用`tl.float16(scalar)` | 改用`scalar.to(tl.float16)` |
| 精度损失 | BF16累加 | 使用float32累加 |
| NaN输出 | 除零或负数开方 | 添加保护 |
