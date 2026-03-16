# 常见精度问题

本文档收录Triton算子开发中常见的精度问题及其解决方案。

---

## 1. 数值溢出/下溢

### 1.1 指数运算溢出

**问题**：计算 `exp(x)` 时，当 `x` 过大导致结果溢出。

**症状**：
- 输出包含 Inf
- Softmax 输出全为 0 或 NaN

**解决方案**：

```python
# 错误：直接计算exp
exp_x = tl.exp(x)

# 正确：减去最大值后再计算
max_x = tl.max(x, axis=0)
exp_x = tl.exp(x - max_x)
```

### 1.2 除法下溢

**问题**：除以非常小的数导致结果过大。

**症状**：
- 输出包含 Inf 或非常大的值
- 相对误差很大

**解决方案**：

```python
# 错误：直接除法
result = a / b

# 正确：添加小的epsilon
eps = 1e-10
result = a / (b + eps)
```

---

## 2. 精度损失

### 2.1 BF16/FP16 精度损失

**问题**：低精度浮点数导致精度损失。

**症状**：
- 相对误差在 1e-2 到 1e-3 范围
- 累加操作误差较大

**解决方案**：

```python
# 错误：直接在低精度下计算
result = tl.sum(x, axis=0)  # x 是 bf16

# 正确：提升精度后计算
x_f32 = x.to(tl.float32)
result = tl.sum(x_f32, axis=0)
result = result.to(tl.bfloat16)
```

### 2.2 累加精度损失

**问题**：大量小数值累加导致精度损失。

**症状**：
- 长序列累加结果误差大
- 误差随序列长度增加

**解决方案**：

```python
# 使用 Kahan 求和或分块求和
def accurate_sum(x, block_size):
    total = 0.0
    for i in range(0, x.shape[0], block_size):
        block = x[i:i+block_size]
        total = total + tl.sum(block.to(tl.float32), axis=0)
    return total
```

---

## 3. NaN/Inf 问题

### 3.1 除零导致 NaN

**问题**：0/0 产生 NaN。

**症状**：
- 输出包含 NaN
- 特定输入形状下出现

**解决方案**：

```python
# 错误：可能除零
result = a / b

# 正确：添加保护
eps = 1e-10
result = tl.where(b != 0, a / (b + eps), 0.0)
```

### 3.2 负数开方

**问题**：对负数开平方产生 NaN。

**症状**：
- 输出包含 NaN
- 特定输入值下出现

**解决方案**：

```python
# 错误：可能对负数开方
result = tl.sqrt(x)

# 正确：添加保护
result = tl.sqrt(tl.maximum(x, 0.0))
```

### 3.3 负数对数

**问题**：对负数或零取对数产生 NaN 或 Inf。

**症状**：
- 输出包含 NaN 或 -Inf
- 输入包含零或负数时出现

**解决方案**：

```python
# 错误：可能对非正数取对数
result = tl.log(x)

# 正确：添加保护
eps = 1e-10
result = tl.log(tl.maximum(x, eps))
```

---

## 4. 类型转换问题

### 4.1 类型转换精度损失

**问题**：类型转换时精度损失。

**症状**：
- 转换后数值变化
- 累积误差

**解决方案**：

```python
# 尽量在计算过程中保持高精度
# 只在输入输出时进行类型转换

# 计算过程使用 float32
x_f32 = x.to(tl.float32)
result_f32 = compute(x_f32)
result = result_f32.to(tl.bfloat16)
```

### 4.2 整数溢出

**问题**：整数运算溢出。

**症状**：
- 结果为负数或异常大
- 特定输入范围下出现

**解决方案**：

```python
# 使用足够大的整数类型
# 或转换为浮点数计算
index = tl.arange(0, N).to(tl.int64)  # 使用 int64
```

---

## 5. 边界条件问题

### 5.1 空输入

**问题**：输入为空张量时出错。

**症状**：
- 运行时错误
- 形状不匹配

**解决方案**：

```python
# 在 host 函数中检查
def my_operator(x):
    if x.numel() == 0:
        return torch.empty_like(x)
    # 正常处理
    ...
```

### 5.2 单元素输入

**问题**：单元素输入时规约操作异常。

**症状**：
- 规约结果异常
- 形状问题

**解决方案**：

```python
# 确保规约操作正确处理单元素
result = tl.sum(x, axis=0, keepdims=True)
```

---

## 6. Softmax 特有问题

### 6.1 数值稳定性

**问题**：Softmax 计算中数值不稳定。

**症状**：
- 输出全为 0 或 NaN
- 特定输入范围下出现

**解决方案**：

```python
@triton.jit
def stable_softmax(x):
    # 减去最大值
    max_x = tl.max(x, axis=0)
    x_shifted = x - max_x
    
    # 计算 exp
    exp_x = tl.exp(x_shifted)
    
    # 求和
    sum_exp = tl.sum(exp_x, axis=0)
    
    # 归一化
    return exp_x / sum_exp
```

### 6.2 在线 Softmax

**问题**：分块计算 Softmax 时精度问题。

**解决方案**：

```python
@triton.jit
def online_softmax(x, BLOCK_SIZE: tl.constexpr):
    # 在线算法，逐步更新最大值和求和
    m = float("-inf")
    l = 0.0
    
    for i in range(0, x.shape[0], BLOCK_SIZE):
        block = x[i:i+BLOCK_SIZE]
        block_max = tl.max(block)
        m_new = tl.maximum(m, block_max)
        
        # 更新求和
        l = l * tl.exp(m - m_new) + tl.sum(tl.exp(block - m_new))
        m = m_new
    
    # 最终归一化
    return tl.exp(x - m) / l
```

---

## 7. LayerNorm 特有问题

### 7.1 方差为零

**问题**：输入完全相同时方差为零，导致除零。

**症状**：
- 输出包含 Inf 或 NaN
- 输入为常量时出现

**解决方案**：

```python
@triton.jit
def layernorm(x, eps: tl.constexpr):
    mean = tl.mean(x, axis=0)
    var = tl.variance(x, axis=0)
    
    # 添加 epsilon 防止除零
    rstd = tl.rsqrt(var + eps)
    
    return (x - mean) * rstd
```

---

## 8. 矩阵乘法特有问题

### 8.1 累加精度

**问题**：矩阵乘法累加时精度损失。

**症状**：
- 大矩阵乘法误差较大
- 误差随矩阵大小增加

**解决方案**：

```python
@triton.jit
def matmul_kernel(...):
    # 使用 float32 累加器
    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
    
    for k in range(0, K, BLOCK_K):
        a = tl.load(...).to(tl.float32)
        b = tl.load(...).to(tl.float32)
        acc += tl.dot(a, b)
    
    # 最后再转换回目标精度
    result = acc.to(tl.bfloat16)
```

---

## 9. 调试技巧

### 9.1 分步验证

```python
# 将复杂计算分解为简单步骤
# 每步验证结果

# 步骤1
step1_result = compute_step1(input)
print(f"Step1: min={step1_result.min()}, max={step1_result.max()}")

# 步骤2
step2_result = compute_step2(step1_result)
print(f"Step2: min={step2_result.min()}, max={step2_result.max()}")
```

### 9.2 边界值测试

```python
# 测试边界值
test_cases = [
    torch.zeros(shape),           # 全零
    torch.ones(shape),            # 全一
    torch.full(shape, 1e10),      # 大值
    torch.full(shape, 1e-10),     # 小值
    torch.randn(shape) * 1e10,    # 大范围随机
]
```

### 9.3 对比参考实现

```python
# 使用 PyTorch 参考实现对比
expected = torch_reference(input)
output = triton_operator(input)

# 逐元素对比
diff = torch.abs(output - expected)
print(f"Max diff: {diff.max()}")
print(f"Mean diff: {diff.mean()}")
```

---

## 10. 精度验证清单

- [ ] 检查 NaN/Inf
- [ ] 检查数值范围
- [ ] 对比参考实现
- [ ] 测试边界情况
- [ ] 测试多种形状
- [ ] 测试多种数据类型
- [ ] 检查累加精度
- [ ] 检查类型转换
