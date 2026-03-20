# Test Template

测试模板用于验证 Triton 算子的正确性和性能。

## 测试框架

### 正确性测试

```python
import torch
import pytest


def test_correctness():
    """测试正确性"""
    torch.manual_seed(42)
    
    # 准备输入数据
    x = torch.randn(128, 1024, device='npu', dtype=torch.float16)
    y = torch.randn(128, 1024, device='npu', dtype=torch.float16)
    
    # 运行 Triton 算子
    output = my_operator(x, y)
    
    # 运行 PyTorch 参考实现
    expected = torch_reference(x, y)
    
    # 验证结果
    assert torch.allclose(output, expected, rtol=1e-3, atol=1e-3)
    print("✅ Correctness test passed!")


def test_dtype(dtype):
    """测试不同数据类型"""
    torch.manual_seed(42)
    
    x = torch.randn(128, 1024, device='npu', dtype=dtype)
    y = torch.randn(128, 1024, device='npu', dtype=dtype)
    
    output = my_operator(x, y)
    expected = torch_reference(x, y)
    
    assert torch.allclose(output, expected, rtol=1e-3, atol=1e-3)
    print(f"✅ {dtype} test passed!")


if __name__ == "__main__":
    # 运行正确性测试
    test_correctness()
    
    # 测试不同数据类型
    for dtype in [torch.float16, torch.bfloat16, torch.float32]:
        test_dtype(dtype)
```

### 性能测试

```python
import torch
import time


def test_performance():
    """测试性能"""
    x = torch.randn(128, 1024, device='npu', dtype=torch.float16)
    y = torch.randn(128, 1024, device='npu', dtype=torch.float16)
    
    # Warmup
    for _ in range(10):
        _ = my_operator(x, y)
    
    # Benchmark
    sync_device = torch.npu if hasattr(torch, 'npu') and torch.npu.is_available() else torch.cuda
    sync_device.synchronize()
    
    start = time.time()
    for _ in range(100):
        _ = my_operator(x, y)
    sync_device.synchronize()
    end = time.time()
    
    avg_time_ms = (end - start) / 100 * 1000
    print(f"Average time: {avg_time_ms:.3f} ms")
    
    return avg_time_ms


def compare_performance():
    """与 PyTorch 对比性能"""
    x = torch.randn(128, 1024, device='npu', dtype=torch.float16)
    y = torch.randn(128, 1024, device='npu', dtype=torch.float16)
    
    # Triton 算子
    triton_time = test_performance()
    
    # PyTorch 参考实现
    torch_time = test_torch_performance(x, y)
    
    speedup = torch_time / triton_time
    print(f"Speedup: {speedup:.2f}x")
    
    return speedup
```

### 使用 Triton Benchmark 工具

```python
import triton
import torch


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['size'],
        x_vals=[2**i for i in range(12, 28, 1)],
        x_log=True,
        line_arg='provider',
        line_vals=['triton', 'torch'],
        line_names=['Triton', 'Torch'],
        styles=[('blue', '-'), ('green', '-')],
        ylabel='GB/s',
        plot_name='vector-add-performance',
        args={},
    ))
def benchmark(size, provider):
    x = torch.rand(size, device='npu', dtype=torch.float32)
    y = torch.rand(size, device='npu', dtype=torch.float32)
    quantiles = [0.5, 0.2, 0.8]
    
    if provider == 'torch':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: x + y, quantiles=quantiles)
    if provider == 'triton':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: add(x, y), quantiles=quantiles)
    
    gbps = lambda ms: 3 * x.numel() * x.element_size() * 1e-9 / (ms * 1e-3)
    return gbps(ms), gbps(max_ms), gbps(min_ms)


if __name__ == "__main__":
    benchmark.run(print_data=True, show_plots=True)
```

## 完整测试示例

```python
import torch
import triton
import triton.language as tl
import pytest


@triton.jit
def my_kernel(x_ptr, y_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0)
    out = x + y
    tl.store(out_ptr + offsets, out, mask=mask)


def my_operator(x, y):
    n_elements = x.numel()
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    out = torch.empty_like(x)
    my_kernel[grid](x, y, out, n_elements, BLOCK_SIZE)
    return out


def torch_reference(x, y):
    return x + y


class TestMyOperator:
    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
    def test_correctness(self, dtype):
        torch.manual_seed(42)
        x = torch.randn(128, 1024, device='npu', dtype=dtype)
        y = torch.randn(128, 1024, device='npu', dtype=dtype)
        
        output = my_operator(x, y)
        expected = torch_reference(x, y)
        
        assert torch.allclose(output, expected, rtol=1e-3, atol=1e-3)
    
    @pytest.mark.parametrize("size", [2**i for i in range(10, 20)])
    def test_sizes(self, size):
        torch.manual_seed(42)
        x = torch.randn(size, device='npu', dtype=torch.float16)
        y = torch.randn(size, device='npu', dtype=torch.float16)
        
        output = my_operator(x, y)
        expected = torch_reference(x, y)
        
        assert torch.allclose(output, expected, rtol=1e-3, atol=1e-3)
    
    def test_performance(self):
        x = torch.randn(128, 1024, device='npu', dtype=torch.float16)
        y = torch.randn(128, 1024, device='npu', dtype=torch.float16)
        
        # Warmup
        for _ in range(10):
            _ = my_operator(x, y)
        
        # Benchmark
        torch.npu.synchronize()
        import time
        start = time.time()
        for _ in range(100):
            _ = my_operator(x, y)
        torch.npu.synchronize()
        end = time.time()
        
        avg_time_ms = (end - start) / 100 * 1000
        print(f"Average time: {avg_time_ms:.3f} ms")
        assert avg_time_ms < 1.0  # 性能要求


if __name__ == "__main__":
    test = TestMyOperator()
    test.test_correctness(torch.float16)
    test.test_performance()
```
