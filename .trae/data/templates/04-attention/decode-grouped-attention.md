# Decode Grouped Attention

Decode 阶段的 Grouped Multi-Head Attention (GQA/MQA/MLA) 算子，是 LLM 推理的核心算子。

## 算子描述

**功能**：在 Decode 阶段，计算 Q（单个 token）与 KV Cache（多个历史 token）的注意力

**特征**：
- Q 是一维的（batch, head, dim）
- KV Cache 是 2D/3D 的（seq_len, head, dim）
- 需要处理变长序列
- 访存模式复杂：离散索引 gather

## 硬件约束

| 约束 | 值 | 说明 |
|------|-----|------|
| UB 占用 | ≤ 85KB/循环 | 需要存储 Q、K、V、attention score 等 |
| L0 Cache | 用于 KV Cache | 需要高效的 gather 模式 |

## 核心代码

### 简化版 Decode Attention

```python
import triton
import triton.language as tl


@triton.jit
def decode_attention_kernel(
    Q,                    # [B, H, D] query
    K_Buffer,             # [max_seq, H_kv, D] key cache
    V_Buffer,             # [max_seq, H_kv, D] value cache
    Out,                  # [B, H, D] output
    kv_indptr,            # [B+1] kv 序列长度累积
    kv_indices,           # [total_kv] kv 索引
    stride_qb, stride_qh, stride_qd,
    stride_kb, stride_kh, stride_kd,
    stride_vb, stride_vh, stride_vd,
    stride_ob, stride_oh, stride_od,
    B, H, H_kv, max_seq, D,
    BLOCK_N: tl.constexpr,
):
    """
    简化的 decode attention kernel
    
    使用 online softmax 避免先计算完整 softmax
    """
    cur_batch = tl.program_id(0)
    cur_head = tl.program_id(1)
    
    # 计算当前 batch 的 KV 范围
    kv_start = tl.load(kv_indptr + cur_batch)
    kv_end = tl.load(kv_indptr + cur_batch + 1)
    seq_len = kv_end - kv_start
    
    # 加载 Q
    q_offsets = cur_batch * stride_qb + cur_head * stride_qh + tl.arange(0, D)
    q = tl.load(Q + q_offsets, mask=q_offsets < B * stride_qb + H * stride_qh + D)
    
    # 初始化累加器
    e_max = float("-inf")
    e_sum = 0.0
    acc = tl.zeros((D,), dtype=tl.float32)
    
    # 遍历 KV
    for start_n in range(0, seq_len, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)
        mask = offs_n < seq_len
        
        # Gather K
        kv_idx = tl.load(kv_indices + kv_start + offs_n, mask=mask, other=0)
        k_offsets = kv_idx * stride_kb + (cur_head // (H // H_kv)) * stride_kh + tl.arange(0, D)
        k = tl.load(K_Buffer + k_offsets, mask=mask[:, None], other=0.0)
        
        # QK^T
        qk = tl.dot(q, k)
        qk *= 1.0 / (D ** 0.5)
        qk = tl.where(mask, qk, float("-inf"))
        
        # Online softmax
        n_e_max = tl.maximum(tl.max(qk, axis=0), e_max)
        re_scale = tl.exp(e_max - n_e_max)
        p = tl.exp(qk - n_e_max)
        acc *= re_scale
        
        # Gather V
        v_offsets = kv_idx * stride_vb + (cur_head // (H // H_kv)) * stride_vh + tl.arange(0, D)
        v = tl.load(V_Buffer + v_offsets, mask=mask[:, None], other=0.0)
        
        acc += tl.dot(p.to(v.dtype), v)
        e_sum = e_sum * re_scale + tl.sum(p, axis=0)
        e_max = n_e_max
    
    # 归一化
    out = acc / e_sum
    out = out.to(Out.dtype.element_ty)
    
    # 存储输出
    out_offsets = cur_batch * stride_ob + cur_head * stride_oh + tl.arange(0, D)
    tl.store(Out + out_offsets, out, mask=out_offsets < B * stride_ob + H * stride_oh + D)


def decode_attention(
    q,                    # [B, H, D]
    k_buffer,             # [max_seq, H_kv, D]
    v_buffer,             # [max_seq, H_kv, D]
    kv_indptr,            # [B+1]
    kv_indices,           # [total_kv]
):
    B, H, D = q.shape
    max_seq, H_kv, _ = k_buffer.shape
    
    out = torch.empty_like(q)
    
    BLOCK_N = 16
    
    grid = (B, H)
    decode_attention_kernel[grid](
        q, k_buffer, v_buffer, out,
        kv_indptr, kv_indices,
        q.stride(0), q.stride(1), q.stride(2),
        k_buffer.stride(0), k_buffer.stride(1), k_buffer.stride(2),
        v_buffer.stride(0), v_buffer.stride(1), v_buffer.stride(2),
        out.stride(0), out.stride(1), out.stride(2),
        B, H, H_kv, max_seq, D,
        BLOCK_N
    )
    
    return out
```

### 完整版（来自实际代码）

完整版本包含：
1. Split KV 支持（multi-split decoding）
2. GQA/MQA 支持（grouped KV heads）
3. MLA 支持（multi-latent attention）
4. 高效的 gather 模式

详见 `triton-ascend-ops/tutorial/best_practice/002-decode_grouped_attention.py`

## 使用示例

```python
import torch

B, H, H_kv, D = 4, 32, 8, 128
max_seq = 2048

q = torch.randn(B, H, D, device='npu', dtype=torch.bfloat16)
k_buffer = torch.randn(max_seq, H_kv, D, device='npu', dtype=torch.bfloat16)
v_buffer = torch.randn(max_seq, H_kv, D, device='npu', dtype=torch.bfloat16)

# 模拟变长序列
seq_lens = torch.tensor([512, 1024, 768, 2048])
kv_indptr = torch.zeros(B + 1, dtype=torch.int32)
kv_indptr[1:] = torch.cumsum(seq_lens, dim=0)
total_kv = seq_lens.sum().item()
kv_indices = torch.arange(total_kv, dtype=torch.int32, device='npu')

out = decode_attention(q, k_buffer, v_buffer, kv_indptr, kv_indices)

print(f"Output shape: {out.shape}")
print("✅ Decode Attention 完成")
```

## 常见错误

| 错误 | 原因 | 解决方案 |
|------|------|---------|
| 性能差 | 离散索引 gather | 使用连续索引 + 批量处理 |
| 精度损失 | online softmax 实现错误 | 正确实现 re_scale |
| UB 溢出 | BLOCK_N 过大 | 减小 BLOCK_N |
