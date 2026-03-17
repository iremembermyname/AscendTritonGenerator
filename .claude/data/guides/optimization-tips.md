# Triton优化技巧

本文档提供Triton算子优化的实用技巧，重点关注Ascend NPU平台。

---

## 1. 内存访问优化

### 1.1 连续内存访问

**原则**：尽量访问连续的内存地址，避免离散访问。

```python
# 好：连续访问
offsets = tl.arange(0, BLOCK_SIZE)
x = tl.load(ptr + offsets)

# 差：离散访问
indices = [0, 5, 10, 15, ...]  # 非连续
x = tl.load(ptr + indices)
```

### 1.2 合并Load操作

**原则**：合并对同一地址的多次load。

```python
# 差：多次load同一数据
x1 = tl.load(ptr + offsets)
x2 = tl.load(ptr + offsets)  # 重复load

# 好：复用load结果
x = tl.load(ptr + offsets)
x1 = x
x2 = x
```

### 1.3 使用Block Pointer

**原则**：对于规则访问模式，使用block pointer。

```python
# 使用make_block_ptr
block_ptr = tl.make_block_ptr(
    base=ptr,
    shape=(M, N),
    strides=(stride_m, stride_n),
    offsets=(0, 0),
    block_shape=(BLOCK_M, BLOCK_N),
    order=(1, 0),
)
x = tl.load(block_ptr)
block_ptr = tl.advance(block_ptr, (BLOCK_M, 0))
```

---

## 2. UB容量优化

### 2.1 控制UB使用量

**原则**：单次循环UB占用 ≤ 85KB。

```python
# 计算UB占用
def calculate_ub(block_size, num_tensors, dtype_bytes=2):
    return block_size * num_tensors * dtype_bytes / 1024  # KB

# 示例：BLOCK_SIZE=1024, 3个tensor, BF16
ub_kb = calculate_ub(1024, 3, 2)  # 6 KB
```

### 2.2 减少中间变量

**原则**：减少同时存活的中间变量数量。

```python
# 差：多个中间变量
a = x + y
b = a * z
c = b - w
result = c / v

# 好：合并计算
result = ((x + y) * z - w) / v
```

### 2.3 及时释放变量

**原则**：计算完成后及时store，释放UB空间。

```python
# 好：边计算边store
for i in range(num_blocks):
    block = compute_block(i)
    tl.store(output_ptr + offsets, block)  # 及时释放
```

---

## 3. 流水线优化

### 3.1 避免带other的load

**原则**：避免使用带other参数的load，它会阻止MTE独立执行。

```python
# 差：带other的load
x = tl.load(ptr + offsets, mask=mask, other=0.0)

# 好：分离load和where
x = tl.load(ptr + offsets, mask=mask)
x = tl.where(mask, x, 0.0)
```

### 3.2 避免数据依赖

**原则**：确保循环迭代可独立执行。

```python
# 差：迭代间有依赖
for i in range(N):
    offset = prev_offset + stride  # 依赖前一次
    x = tl.load(ptr + offset)
    prev_offset = offset

# 好：独立计算偏移
for i in range(N):
    offset = base + i * stride  # 独立计算
    x = tl.load(ptr + offset)
```

### 3.3 加载计算交织

**原则**：加载和计算交替进行，避免等待。

```python
# 好：加载计算交织
for i in range(num_blocks):
    x = tl.load(x_ptr + offsets)  # MTE加载
    y = compute(x)                # Vector计算
    tl.store(y_ptr + offsets, y)  # MTE存储
```

---

## 4. 计算优化

### 4.1 使用高精度累加

**原则**：累加操作使用float32。

```python
# 差：低精度累加
acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.bfloat16)

# 好：高精度累加
acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
# 最后再转换
result = acc.to(tl.bfloat16)
```

### 4.2 减少冗余计算

**原则**：避免重复计算相同的值。

```python
# 差：重复计算
for i in range(N):
    x = tl.exp(input[i])  # 每次都计算exp
    y = x / sum_exp

# 好：预计算
exp_values = tl.exp(input)
sum_exp = tl.sum(exp_values)
for i in range(N):
    y = exp_values[i] / sum_exp
```

### 4.3 使用tl.arange代替循环

**原则**：用向量化操作代替标量循环。

```python
# 差：标量循环
for i in range(BLOCK_SIZE):
    offsets[i] = i

# 好：向量化
offsets = tl.arange(0, BLOCK_SIZE)
```

---

## 5. 分核优化

### 5.1 负载均衡

**原则**：均匀分配工作量到各核。

```python
# 计算每个核的工作量
num_tokens_per_core = (num_tokens + num_cores - 1) // num_cores

# 分配工作
for core_id in range(num_cores):
    start = core_id * num_tokens_per_core
    end = min(start + num_tokens_per_core, num_tokens)
    # 处理 [start, end) 范围
```

### 5.2 避免过度分核

**原则**：不要创建过多的program。

```python
# 差：过度分核
grid = (num_elements,)  # 每个元素一个program

# 好：合理分核
grid = (triton.cdiv(num_elements, BLOCK_SIZE),)
```

---

## 6. Mask优化

### 6.1 预计算mask

**原则**：在循环外预计算mask。

```python
# 差：每次循环计算mask
for i in range(num_blocks):
    offsets = i * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n  # 每次计算

# 好：预计算mask
base_offsets = tl.arange(0, BLOCK_SIZE)
for i in range(num_blocks):
    offsets = i * BLOCK_SIZE + base_offsets
    mask = offsets < n
```

### 6.2 避免二维mask

**原则**：避免使用二维mask，它占用大量UB。

```python
# 差：二维mask
mask = offsets_m[:, None] < M and offsets_n[None, :] < N

# 好：使用insert_slice
for i in range(M):
    row = tl.load(ptr + i * stride + offsets_n)
    tl.store(out_ptr + i * stride + offsets_n, row)
```

---

## 7. 特定算子优化

### 7.1 Softmax优化

```python
@triton.jit
def optimized_softmax(x_ptr, output_ptr, M, N, stride_m, BLOCK_N: tl.constexpr):
    row_idx = tl.program_id(0)
    row_start = row_idx * stride_m
    
    # 一次遍历找最大值和求和
    max_val = float("-inf")
    sum_exp = 0.0
    
    for block_start in range(0, N, BLOCK_N):
        offsets = block_start + tl.arange(0, BLOCK_N)
        mask = offsets < N
        x = tl.load(x_ptr + row_start + offsets, mask=mask)
        
        block_max = tl.max(x, axis=0)
        new_max = tl.maximum(max_val, block_max)
        
        # 更新sum_exp
        sum_exp = sum_exp * tl.exp(max_val - new_max)
        exp_x = tl.exp(x - new_max)
        sum_exp += tl.sum(exp_x, axis=0)
        
        max_val = new_max
    
    # 归一化
    for block_start in range(0, N, BLOCK_N):
        offsets = block_start + tl.arange(0, BLOCK_N)
        mask = offsets < N
        x = tl.load(x_ptr + row_start + offsets, mask=mask)
        out = tl.exp(x - max_val) / sum_exp
        tl.store(output_ptr + row_start + offsets, out, mask=mask)
```

### 7.2 LayerNorm优化

```python
@triton.jit
def optimized_layernorm(x_ptr, y_ptr, w_ptr, b_ptr, M, N, stride_m, eps: tl.constexpr, BLOCK_N: tl.constexpr):
    row_idx = tl.program_id(0)
    row_start = row_idx * stride_m
    
    # 一次遍历计算均值和方差
    sum_x = 0.0
    sum_x2 = 0.0
    
    for block_start in range(0, N, BLOCK_N):
        offsets = block_start + tl.arange(0, BLOCK_N)
        mask = offsets < N
        x = tl.load(x_ptr + row_start + offsets, mask=mask).to(tl.float32)
        
        sum_x += tl.sum(x, axis=0)
        sum_x2 += tl.sum(x * x, axis=0)
    
    mean = sum_x / N
    var = sum_x2 / N - mean * mean
    rstd = tl.rsqrt(var + eps)
    
    # 归一化
    for block_start in range(0, N, BLOCK_N):
        offsets = block_start + tl.arange(0, BLOCK_N)
        mask = offsets < N
        x = tl.load(x_ptr + row_start + offsets, mask=mask)
        w = tl.load(w_ptr + offsets, mask=mask)
        b = tl.load(b_ptr + offsets, mask=mask)
        
        y = (x - mean) * rstd * w + b
        tl.store(y_ptr + row_start + offsets, y, mask=mask)
```

---

## 8. 性能调试技巧

### 8.1 使用msprof

```bash
# 采集性能数据
msprof op --output=./profile --kernel-name="my_kernel" \
    --warm-up=20 --launch-count=20 python test.py

# 分析结果
# 查看 Task Duration, MTE/Vector Utilization, UB Usage
```

### 8.2 对比优化前后

```python
# 记录优化前性能
before_time = benchmark_kernel(kernel_fn, inputs)

# 应用优化
optimized_code = apply_optimization(code)

# 记录优化后性能
after_time = benchmark_kernel(optimized_kernel_fn, inputs)

# 计算加速比
speedup = before_time / after_time
```

### 8.3 逐步优化

```python
# 一次只应用一个优化，验证效果
optimizations = [
    ("remove_load_other", remove_load_other),
    ("optimize_memory_access", optimize_memory_access),
    ("adjust_block_size", adjust_block_size),
]

for name, opt_fn in optimizations:
    code = opt_fn(code)
    time = benchmark(code)
    print(f"{name}: {time} ms")
```
