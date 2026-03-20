# Benchmark Template

性能测试模板用于系统性地评估和对比 Triton 算子的性能。

## 基础性能测试

### 简单 Benchmark

```python
import torch
import time


def benchmark_operator(operator, input_shapes, device='npu', dtype=torch.float16, iterations=100):
    """
    基础性能测试
    
    参数:
        operator: 要测试的算子函数
        input_shapes: 输入形状列表
        device: 测试设备
        dtype: 数据类型
        iterations: 迭代次数
    
    返回:
        dict: 性能数据
    """
    results = {}
    
    sync_fn = torch.npu.synchronize if device == 'npu' else torch.cuda.synchronize
    
    for shape in input_shapes:
        # 准备输入
        if isinstance(shape, tuple):
            inputs = [torch.randn(shape, device=device, dtype=dtype) for _ in range(2)]
        else:
            inputs = [torch.randn(shape, device=device, dtype=dtype) for _ in range(2)]
        
        # Warmup
        for _ in range(10):
            _ = operator(*inputs)
        sync_fn()
        
        # Benchmark
        start = time.time()
        for _ in range(iterations):
            _ = operator(*inputs)
        sync_fn()
        end = time.time()
        
        avg_time_ms = (end - start) / iterations * 1000
        
        # 计算带宽
        total_bytes = sum(inp.numel() * inp.element_size() for inp in inputs) * 2  # read + write
        bandwidth_gbps = total_bytes / (avg_time_ms * 1e-6) / 1e9
        
        results[shape] = {
            'avg_time_ms': avg_time_ms,
            'bandwidth_gbps': bandwidth_gbps,
        }
        
        print(f"Shape: {shape}")
        print(f"  Time: {avg_time_ms:.3f} ms")
        print(f"  Bandwidth: {bandwidth_gbps:.2f} GB/s")
    
    return results
```

## 使用 Triton Benchmark 工具

### 完整 Benchmark

```python
import triton
import torch
import triton.testing


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['M', 'N'],
        x_vals=[(128 * i, 128 * i) for i in range(2, 33)],
        line_arg='provider',
        line_vals=['triton', 'torch'],
        line_names=['Triton', 'Torch'],
        styles=[('blue', '-'), ('green', '-')],
        ylabel='GB/s',
        plot_name='matmul-performance',
        args={},
    ))
def benchmark_matmul(M, N, provider):
    K = 256
    
    a = torch.randn((M, K), device='npu', dtype=torch.float16)
    b = torch.randn((K, N), device='npu', dtype=torch.float16)
    quantiles = [0.5, 0.2, 0.8]
    
    if provider == 'torch':
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: torch.matmul(a, b),
            quantiles=quantiles
        )
    
    if provider == 'triton':
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: my_matmul(a, b),
            quantiles=quantiles
        )
    
    # 计算带宽
    total_bytes = (M * K + K * N + M * N) * 2  # read A, read B, write C
    gbps = lambda ms: total_bytes * 2 / (ms * 1e-6) / 1e9  # read + write
    
    return gbps(ms), gbps(max_ms), gbps(min_ms)


if __name__ == "__main__":
    benchmark_matmul.run(print_data=True, show_plots=True)
```

### 多配置对比

```python
@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['size'],
        x_vals=[2**i for i in range(12, 28, 1)],
        line_arg='block_size',
        line_vals=[256, 512, 1024, 2048],
        line_names=['BS=256', 'BS=512', 'BS=1024', 'BS=2048'],
        styles=[('blue', '-'), ('red', '-'), ('green', '-'), ('orange', '-')],
        ylabel='Time (ms)',
        plot_name='block-size-sweep',
        args={},
    ))
def benchmark_block_size(size, block_size):
    x = torch.rand(size, device='npu', dtype=torch.float32)
    y = torch.rand(size, device='npu', dtype=torch.float32)
    quantiles = [0.5, 0.2, 0.8]
    
    ms, min_ms, max_ms = triton.testing.do_bench(
        lambda: add_with_block_size(x, y, block_size),
        quantiles=quantiles
    )
    
    return ms, max_ms, min_ms


if __name__ == "__main__":
    benchmark_block_size.run(print_data=True, show_plots=True)
```

## 详细性能分析

### Profiling 工具集成

```python
import torch
import time


def profile_operator(operator, inputs, warmup=10, iterations=100):
    """
    详细性能分析
    
    参数:
        operator: 要分析的算子
        inputs: 输入数据
        warmup: warmup 次数
        iterations: 测试迭代次数
    """
    sync_fn = torch.npu.synchronize if hasattr(torch, 'npu') else torch.cuda.synchronize
    
    # Warmup
    for _ in range(warmup):
        _ = operator(*inputs)
    sync_fn()
    
    # 记录时间
    start = time.time()
    for _ in range(iterations):
        _ = operator(*inputs)
    sync_fn()
    end = time.time()
    
    avg_time_ms = (end - start) / iterations * 1000
    
    print(f"\n=== Performance Profile ===")
    print(f"Average time: {avg_time_ms:.3f} ms")
    print(f"Total time: {(end - start) * 1000:.3f} ms")
    print(f"Iterations: {iterations}")
    
    # 计算吞吐量
    total_elements = sum(inp.numel() for inp in inputs) * 2
    throughput_elems = total_elements / (avg_time_ms * 1e-3)
    print(f"Throughput: {throughput_elems / 1e6:.2f} M elements/s")
    
    return avg_time_ms
```

## 使用示例

```python
import torch
import triton
import triton.language as tl


@triton.jit
def add_kernel(x_ptr, y_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0)
    out = x + y
    tl.store(out_ptr + offsets, out, mask=mask)


def add_triton(x, y, BLOCK_SIZE=1024):
    n_elements = x.numel()
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    out = torch.empty_like(x)
    add_kernel[grid](x, y, out, n_elements, BLOCK_SIZE)
    return out


def add_torch(x, y):
    return x + y


if __name__ == "__main__":
    # 基础测试
    shapes = [(128 * i, 128 * i) for i in range(2, 33)]
    
    results = benchmark_operator(
        add_triton,
        shapes,
        device='npu',
        dtype=torch.float16
    )
    
    # 对比测试
    @triton.testing.perf_report(
        triton.testing.Benchmark(
            x_names=['size'],
            x_vals=[2**i for i in range(12, 28, 1)],
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
            ms, min_ms, max_ms = triton.testing.do_bench(lambda: add_triton(x, y), quantiles=quantiles)
        
        gbps = lambda ms: 3 * x.numel() * x.element_size() * 1e-9 / (ms * 1e-3)
        return gbps(ms), gbps(max_ms), gbps(min_ms)
    
    benchmark.run(print_data=True, show_plots=True)
```
