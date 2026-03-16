"""
Test Data Generator for Precision Verification

Generates various types of test data for verifying Triton operator precision.
"""

import torch
import numpy as np
from typing import List, Tuple, Optional, Union
from enum import Enum


class DataType(Enum):
    FLOAT16 = "float16"
    BFLOAT16 = "bfloat16"
    FLOAT32 = "float32"
    INT8 = "int8"
    INT32 = "int32"


class EdgeCaseType(Enum):
    ZEROS = "zeros"
    ONES = "ones"
    LARGE_VALUES = "large"
    SMALL_VALUES = "small"
    MIXED_SCALE = "mixed"
    POSITIVE_ONLY = "positive"
    NEGATIVE_ONLY = "negative"
    SPARSE = "sparse"


def get_torch_dtype(dtype: Union[str, DataType]) -> torch.dtype:
    dtype_map = {
        DataType.FLOAT16: torch.float16,
        DataType.BFLOAT16: torch.bfloat16,
        DataType.FLOAT32: torch.float32,
        DataType.INT8: torch.int8,
        DataType.INT32: torch.int32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
        "int8": torch.int8,
        "int32": torch.int32,
    }
    return dtype_map.get(dtype, torch.float32)


def generate_random_data(
    shape: Tuple[int, ...],
    dtype: Union[str, DataType] = "float16",
    device: str = "npu",
    seed: Optional[int] = None,
    mean: float = 0.0,
    std: float = 1.0,
) -> torch.Tensor:
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
    
    torch_dtype = get_torch_dtype(dtype)
    
    if torch_dtype in [torch.float16, torch.bfloat16]:
        data = torch.randn(shape, dtype=torch.float32, device=device)
        data = data * std + mean
        data = data.to(torch_dtype)
    elif torch_dtype == torch.float32:
        data = torch.randn(shape, dtype=torch.float32, device=device)
        data = data * std + mean
    elif torch_dtype in [torch.int8, torch.int32]:
        data = torch.randint(-100, 100, shape, dtype=torch_dtype, device=device)
    else:
        data = torch.randn(shape, dtype=torch_dtype, device=device)
    
    return data


def generate_edge_case_data(
    shape: Tuple[int, ...],
    dtype: Union[str, DataType] = "float16",
    edge_case: Union[str, EdgeCaseType] = EdgeCaseType.ZEROS,
    device: str = "npu",
) -> torch.Tensor:
    torch_dtype = get_torch_dtype(dtype)
    
    if edge_case == EdgeCaseType.ZEROS or edge_case == "zeros":
        return torch.zeros(shape, dtype=torch_dtype, device=device)
    
    elif edge_case == EdgeCaseType.ONES or edge_case == "ones":
        return torch.ones(shape, dtype=torch_dtype, device=device)
    
    elif edge_case == EdgeCaseType.LARGE_VALUES or edge_case == "large":
        data = torch.randn(shape, dtype=torch.float32, device=device)
        data = data * 1000
        return data.to(torch_dtype)
    
    elif edge_case == EdgeCaseType.SMALL_VALUES or edge_case == "small":
        data = torch.randn(shape, dtype=torch.float32, device=device)
        data = data * 0.001
        return data.to(torch_dtype)
    
    elif edge_case == EdgeCaseType.MIXED_SCALE or edge_case == "mixed":
        data = torch.randn(shape, dtype=torch.float32, device=device)
        scales = torch.pow(10, torch.randint(-3, 4, shape, device=device).float())
        data = data * scales
        return data.to(torch_dtype)
    
    elif edge_case == EdgeCaseType.POSITIVE_ONLY or edge_case == "positive":
        data = torch.rand(shape, dtype=torch_dtype, device=device) * 10
        return data
    
    elif edge_case == EdgeCaseType.NEGATIVE_ONLY or edge_case == "negative":
        data = -torch.rand(shape, dtype=torch_dtype, device=device) * 10
        return data
    
    elif edge_case == EdgeCaseType.SPARSE or edge_case == "sparse":
        data = torch.randn(shape, dtype=torch_dtype, device=device)
        mask = torch.rand(shape, device=device) > 0.9
        data = data * mask.to(torch_dtype)
        return data
    
    else:
        return torch.zeros(shape, dtype=torch_dtype, device=device)


def generate_test_shapes(
    base_shape: Tuple[int, ...],
    variations: List[str] = None,
) -> List[Tuple[int, ...]]:
    if variations is None:
        variations = ["same", "smaller", "larger", "power_of_2"]
    
    shapes = []
    
    for var in variations:
        if var == "same":
            shapes.append(base_shape)
        elif var == "smaller":
            shape = tuple(max(1, s // 2) for s in base_shape)
            shapes.append(shape)
        elif var == "larger":
            shape = tuple(s * 2 for s in base_shape)
            shapes.append(shape)
        elif var == "power_of_2":
            shape = tuple(2 ** int(np.log2(s)) for s in base_shape)
            shapes.append(shape)
        elif var == "non_power_of_2":
            shape = tuple(max(1, s - 1) if s > 1 else s for s in base_shape)
            shapes.append(shape)
        elif var == "prime":
            def next_prime(n):
                if n < 2:
                    return 2
                for i in range(n, 2 * n):
                    for j in range(2, int(i ** 0.5) + 1):
                        if i % j == 0:
                            break
                    else:
                        return i
                return n
            shape = tuple(next_prime(s) for s in base_shape)
            shapes.append(shape)
    
    return shapes


def generate_batch_test_data(
    shapes: List[Tuple[int, ...]],
    dtype: Union[str, DataType] = "float16",
    device: str = "npu",
    seed: Optional[int] = None,
) -> List[torch.Tensor]:
    if seed is not None:
        torch.manual_seed(seed)
    
    data_list = []
    for i, shape in enumerate(shapes):
        data = generate_random_data(shape, dtype, device, seed=seed + i if seed else None)
        data_list.append(data)
    
    return data_list


def generate_sequence_data(
    batch_size: int,
    seq_len: int,
    hidden_size: int,
    dtype: Union[str, DataType] = "float16",
    device: str = "npu",
    seed: Optional[int] = None,
) -> torch.Tensor:
    return generate_random_data(
        (batch_size, seq_len, hidden_size),
        dtype,
        device,
        seed,
    )


def generate_attention_data(
    batch_size: int,
    num_heads: int,
    seq_len: int,
    head_dim: int,
    dtype: Union[str, DataType] = "float16",
    device: str = "npu",
    seed: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if seed is not None:
        torch.manual_seed(seed)
    
    q = generate_random_data(
        (batch_size, num_heads, seq_len, head_dim),
        dtype, device, seed
    )
    k = generate_random_data(
        (batch_size, num_heads, seq_len, head_dim),
        dtype, device, seed + 1 if seed else None
    )
    v = generate_random_data(
        (batch_size, num_heads, seq_len, head_dim),
        dtype, device, seed + 2 if seed else None
    )
    
    return q, k, v


def generate_matmul_data(
    M: int, N: int, K: int,
    dtype: Union[str, DataType] = "float16",
    device: str = "npu",
    seed: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if seed is not None:
        torch.manual_seed(seed)
    
    a = generate_random_data((M, K), dtype, device, seed)
    b = generate_random_data((K, N), dtype, device, seed + 1 if seed else None)
    
    return a, b


def generate_layernorm_data(
    batch_size: int,
    hidden_size: int,
    dtype: Union[str, DataType] = "float16",
    device: str = "npu",
    seed: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if seed is not None:
        torch.manual_seed(seed)
    
    x = generate_random_data((batch_size, hidden_size), dtype, device, seed)
    weight = torch.ones(hidden_size, dtype=get_torch_dtype(dtype), device=device)
    bias = torch.zeros(hidden_size, dtype=get_torch_dtype(dtype), device=device)
    
    return x, weight, bias
