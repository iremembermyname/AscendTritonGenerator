# Ascend 扩展 API

本文档介绍Ascend NPU上的Triton扩展API。

---

## 目录

1. [insert_slice](#1-insert_slice)
2. [extract_slice](#2-extract_slice)
3. [tl.gather](#3-tlgather)
4. [care_padding](#4-care_padding)
5. [tl.get_element](#5-tlget_element)
6. [while循环替代方案](#6-while循环替代方案)
7. [动态获取核心数](#7-动态获取核心数)

---

## 1. insert_slice

**功能**：将一个tensor插入到另一个tensor的指定位置，实现数据合并写出到GM，提升性能。

### 接口定义

```python
def insert_slice(ful, sub, offsets, sizes, strides, _builder=None, _generator=None) -> tensor:
    """
    将sub插入到ful的指定位置
    
    :param ful: 目标tensor，接收插入的数据
    :param sub: 要插入的tensor
    :param offsets: 插入位置的偏移量，tuple of ints
    :param sizes: 插入数据的大小，tuple of ints
    :param strides: 插入数据的步长，tuple of ints
    :return: 插入后的tensor
    """
```

### 使用场景

- MOE Token重排：数据随机读取，写出位置连续
- 多个从不同位置读取的tensor合并后一次写出
- 替代cat操作，规避负数offset导致的离散访存

### 示例

```python
@triton.jit
def npu_token_rearrangement_kernel(x_ptr, indices, output_ptr, n_elements, S: tl.constexpr, D: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    dtype = output_ptr.type.element_ty
    
    # 1. 准备输出tensor
    output = tl.full((BLOCK_SIZE, D), 0, dtype=dtype)
    
    # 2. 批量加载重排索引
    idx_offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    idx_mask = idx_offset < S
    idx = tl.load(indices + idx_offset, idx_mask)
    
    # 3. 循环加载数据并插入到output
    for i in tl.range(0, BLOCK_SIZE):
        data_offset = D * tl.get_element(idx, (i,)) + tl.arange(0, D)[None, :]
        data = tl.load(x_ptr + data_offset, data_offset < n_elements)
        output = tl.insert_slice(output, data, [i, 0], [1, D], [1, 1])
    
    # 4. 批量写出到GM
    out_offset = pid * BLOCK_SIZE * D + tl.arange(0, BLOCK_SIZE)[:, None] + tl.arange(0, D)[None, :]
    tl.store(output_ptr + out_offset, output, out_offset < n_elements)
```

---

## 2. extract_slice

**功能**：从一个tensor中提取指定位置的数据，实现批量读取到UB后截取部分处理。

### 接口定义

```python
def extract_slice(ful, offsets, sizes, strides, _builder=None, _generator=None) -> tensor:
    """
    从ful中提取指定位置的数据
    
    :param ful: 源tensor
    :param offsets: 提取位置的偏移量，tuple of ints
    :param sizes: 提取数据的大小，tuple of ints
    :param strides: 提取数据的步长，tuple of ints
    :return: 提取的tensor
    """
```

### 使用场景

- MOE Token反重排：读取连续数据块，分散写出
- 批量读取后分散操作

### 示例

```python
@triton.jit
def npu_token_reverse_kernel(x_ptr, indices, output_ptr, n_elements, S: tl.constexpr, D: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE * D
    
    # 1. 批量加载数据
    data_offset = D * tl.arange(0, BLOCK_SIZE)[:, None] + tl.arange(0, D)[None, :]
    data = tl.load(x_ptr + block_start + data_offset, data_offset < n_elements)
    
    # 2. 批量加载索引
    idx = tl.load(indices + pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE))
    
    # 3. 逐个提取并存储
    for i in tl.arange(0, BLOCK_SIZE):
        x_sub = tl.extract_slice(data, [i, 0], [1, D], [1, 1])
        output_offset = D * tl.get_element(idx, (i,)) + tl.arange(0, D)[None, :]
        tl.store(output_ptr + output_offset, x_sub, output_offset < n_elements)
```

---

## 3. tl.gather

**功能**：从UB中的tensor按索引选择数据，用于离散访存优化。

### 使用场景

- 替代直接从GM离散访问
- 先批量加载到UB，再gather筛选

### 示例

```python
@triton.jit
def pick_kernel(x_ptr, idx_ptr, y_ptr, M: tl.constexpr, N: tl.constexpr):
    pid = tl.program_id(0)
    rm = tl.arange(0, M)  # 加载全部数据
    rn = tl.arange(0, N)
    
    idx = tl.load(idx_ptr + rn)
    mask = idx < M
    
    # 批量加载到UB
    x_shared = tl.load(x_ptr + rm)  # [M]
    # 从UB中gather
    val = tl.gather(x_shared, idx, 0)
    
    tl.store(y_ptr + rn, val, mask=mask)
```

---

## 4. care_padding

**功能**：`tl.load`的扩展参数，用于优化带mask的加载性能。

### 背景

- `tl.load`的`other`参数：当mask为False时，填充`other`值（默认0）
- NPU实现分两步：先将尾块填成`other`值，再加载真实数据
- 这会导致多余的数据搬运和阻塞

### 参数说明

```python
tl.load(ptr + offsets, mask=mask, care_padding=False)
```

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| care_padding | bool | True | False时跳过尾块填充，提升性能 |

### 注意事项

- 设置`care_padding=False`时，尾块中的值是随机数
- 如果业务依赖`other`的值，不能使用此优化
- 同时指定`other`和`care_padding`时，`care_padding`被忽略

### 示例

```python
# 不需要other值时，使用care_padding=False提升性能
idx = tl.arange(0, N)
mask = idx < M  # M是变量
data = tl.load(ptr + idx, mask=mask, care_padding=False)
```

---

## 5. tl.get_element

**功能**：从tensor中获取指定位置的元素，用于循环中逐个处理。

### 示例

```python
for i in tl.range(0, BLOCK_SIZE):
    # 获取idx tensor中第i个元素
    index = tl.get_element(idx, (i,))
    # 使用index加载数据
    data = tl.load(x_ptr + index * D + tl.arange(0, D))
```

### 重要限制

禁止对`tl.arange`生成的张量使用`get_element()`：
- `tl.arange`是编译时索引表达式，非实际张量
- 需直接计算而非提取

```python
# 错误：offsets = base + tl.arange(0, BLOCK_SIZE); value = tl.get_element(offsets, [i])
# 正确：value = base + i
```

---

## 6. while循环替代方案

Ascend后端不支持`while`循环，需根据循环上限是否为编译时常量选择替代方案。

### 情况1：循环上限是静态值

```python
# 错误：while 循环
i = 0
while i < N_ITERS:  # N_ITERS 是编译时常量
    # 处理逻辑
    i += 1

# 正确：直接用 for range
for i in range(N_ITERS):  # N_ITERS: tl.constexpr
    # 处理逻辑
```

### 情况2：循环上限是动态值

```python
# 错误：while 循环（n_iters 是运行时动态值）
@triton.jit
def kernel_while(ptr, n_iters, TILE: tl.constexpr):
    i = 0
    while i < n_iters:
        offset = i * TILE + tl.arange(0, TILE)
        data = tl.load(ptr + offset)
        tl.store(ptr + offset, data * 2)
        i += 1

# 正确：for + if 替代方案
@triton.jit
def kernel_for_if(
    ptr,
    n_iters,              # 运行时动态值
    TILE: tl.constexpr,
    MAX_ITERS: tl.constexpr,  # 编译时常量上界
):
    for i in range(MAX_ITERS):
        if i < n_iters:
            offset = i * TILE + tl.arange(0, TILE)
            data = tl.load(ptr + offset)
            tl.store(ptr + offset, data * 2)
```

---

## 7. 动态获取核心数

根据算子类型选择对应的核心数，**必须在`__init__`中获取**：

```python
import torch_npu

class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # 在__init__中获取核心数，只执行一次
        try:
            # 向量计算类算子使用VEC核心数
            self.VEC_CORE_NUM = torch_npu.npu.npu_config.get_device_limit(0).get("vector_core_num", 40)
            # 矩阵计算类算子使用CUBE核心数
            self.CUBE_CORE_NUM = torch_npu.npu.npu_config.get_device_limit(0).get("cube_core_num", 20)
        except:
            self.VEC_CORE_NUM = 40   # Ascend 910B4 默认
            self.CUBE_CORE_NUM = 20  # Ascend 910B4 默认
```

**注意**：`torch_npu`的import和`get_device_limit`调用会触发设备同步，因此**禁止在forward中调用**。
