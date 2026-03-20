# Gather Scatter

Gather 和 Scatter 是处理非连续内存访问的关键技术，常用于 MoE、稀疏注意力等场景。

## 算子描述

**Gather**：根据索引从张量中取值
```
out[i] = x[indices[i]]
```

**Scatter**：根据索引写入张量
```
out[indices[i]] += x[i]
```

**特征**：
- 非连续内存访问
- 需要高效的索引处理
- 常用于 MoE (Mixture of Experts)

## 核心代码

### Gather Kernel

```python
import triton
import triton.language as tl


@triton.jit
def gather_kernel(
    x_ptr,
    out_ptr,
    indices_ptr,
    INDICES_LENGTH: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    SUB_BLOCK_SIZE: tl.constexpr,
    NUM_COLUMNS: tl.constexpr,
    BLOCK_X: tl.constexpr,
    TOP_K: tl.constexpr,
):
    """
    Gather 操作：根据索引从 x 中取值
    
    out[i] = x[indices[i] // TOP_K]
    """
    pid = tl.program_id(0)
    idx_begin = pid * BLOCK_SIZE
    idx_end = tl.minimum((pid + 1) * BLOCK_SIZE, INDICES_LENGTH)
    
    for idx_in_block in range(0, BLOCK_SIZE, SUB_BLOCK_SIZE):
        idx_offset = idx_begin + idx_in_block
        if idx_offset < idx_end:
            idx_offsets = tl.arange(0, SUB_BLOCK_SIZE) + idx_offset
            idx_mask = idx_offsets < idx_end
            cur_indices = tl.load(indices_ptr + idx_offsets, idx_mask, other=0)
            
            for col_offset in range(0, NUM_COLUMNS, BLOCK_X):
                tmp_buf = tl.zeros((SUB_BLOCK_SIZE, BLOCK_X), x_ptr.dtype.element_ty)
                col_offsets = tl.arange(0, BLOCK_X) + col_offset
                col_mask = col_offsets < NUM_COLUMNS
                
                for i in range(0, SUB_BLOCK_SIZE):
                    idx = tl.get_element(cur_indices, (i,)) // TOP_K * NUM_COLUMNS
                    val = tl.load(x_ptr + idx + col_offsets, col_mask)
                    tmp_buf = tl.insert_slice(tmp_buf, val[None,:], offsets=(i, 0), sizes=(1, BLOCK_X), strides=(1, 1))
                
                tl.store(out_ptr + idx_offsets[:, None] * NUM_COLUMNS + col_offsets[None, :],
                         tmp_buf, idx_mask[:, None] & col_mask[None, :])


def gather(x, indices, top_k):
    out = torch.empty((indices.shape[0], x.shape[1]), dtype=x.dtype, device=x.device)
    
    num_core = 40  # 或从设备获取
    indices_length = indices.shape[0]
    block_size = (indices_length - 1) // num_core + 1
    num_columns = x.shape[1]
    block_x = min(num_columns, 20480)
    sub_block_size = max((80 * 1024 - block_x * 2) // (block_x * 2 + 4), 1)
    
    gather_kernel[(num_core,)](
        x, out, indices,
        indices_length,
        block_size, sub_block_size,
        num_columns, block_x,
        top_k,
        multibuffer=True
    )
    return out
```

### Scatter Kernel

```python
@triton.jit
def scatter_kernel(
    x_ptr,
    out_ptr,
    weights_ptr,
    indices_ptr,
    INDICES_LENGTH: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    SUB_BLOCK_SIZE: tl.constexpr,
    NUM_COLUMNS: tl.constexpr,
    BLOCK_X: tl.constexpr,
    TOP_K: tl.constexpr,
    SCALE: tl.constexpr,
):
    """
    Scatter 操作：根据索引写入到 out
    
    out[indices[i]] += x[i] * weights[i] (可选加权)
    """
    pid = tl.program_id(0)
    idx_begin = pid * BLOCK_SIZE
    idx_end = tl.minimum((pid + 1) * BLOCK_SIZE, INDICES_LENGTH)
    
    for idx_in_block in range(0, BLOCK_SIZE, SUB_BLOCK_SIZE):
        idx_offset = idx_begin + idx_in_block
        idx_offsets = tl.arange(0, SUB_BLOCK_SIZE) + idx_offset
        idx_mask = idx_offsets < idx_end
        cur_indices = tl.load(indices_ptr + idx_offsets, idx_mask, other=0)
        
        for col_offset in range(0, NUM_COLUMNS, BLOCK_X):
            col_offsets = tl.arange(0, BLOCK_X) + col_offset
            col_mask = col_offsets < NUM_COLUMNS
            cur_x = tl.load(x_ptr + idx_offsets[:, None] * NUM_COLUMNS + col_offsets[None, :],
                            idx_mask[:, None] & col_mask[None, :])
            
            for i in range(0, SUB_BLOCK_SIZE):
                if i + idx_offset < idx_end:
                    idx = tl.get_element(cur_indices, (i,))
                    val = tl.extract_slice(cur_x, offsets=(i, 0), sizes=(1, BLOCK_X), strides=(1, 1))
                    if SCALE:
                        scale = tl.load(weights_ptr + idx)
                        val = val.to(tl.float32) * scale.to(tl.float32)
                    
                    tl.store(out_ptr + idx * NUM_COLUMNS + col_offsets, val.to(out_ptr.dtype.element_ty).reshape(BLOCK_X), col_mask)


def scatter(x, indices, weights, top_k):
    tokens = indices.shape[0] // top_k
    out = torch.zeros((tokens, top_k, x.shape[1]), dtype=x.dtype, device=x.device)
    
    num_core = 40
    indices_length = indices.shape[0]
    block_size = (indices_length - 1) // num_core + 1
    num_columns = x.shape[1]
    block_x = min(num_columns, 6144)
    sub_block_size = max((80 * 1024 - block_x * 12) // (block_x * 2 + 4), 1)
    
    scale = weights is not None
    scatter_kernel[(num_core,)](
        x, out, weights, indices,
        indices.shape[0],
        block_size, sub_block_size,
        num_columns, block_x,
        top_k, scale,
        multibuffer=True
    )
    return out.sum(dim=1) if top_k > 1 else out.view(tokens, x.shape[1])
```

## 使用示例

```python
import torch

# Gather 测试
x = torch.randn(1024, 768, device='npu', dtype=torch.float16)
indices = torch.randint(0, 1024, (4096,), dtype=torch.int32, device='npu')
top_k = 4

gathered = gather(x, indices, top_k)
print(f"Gather output shape: {gathered.shape}")

# Scatter 测试
weights = torch.rand(4096, device='npu', dtype=torch.float16)
scattered = scatter(gathered, indices, weights, top_k)
print(f"Scatter output shape: {scattered.shape}")

print("✅ Gather/Scatter 完成")
```
