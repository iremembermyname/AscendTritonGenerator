---
name: benchmark-comparison
description: Triton算子与PyTorch原生实现的性能对比测试。使用warmup+多次运行取平均的方式测量执行时间，不依赖msprof工具。触发场景：用户提到"性能对比"、"对比Triton和Torch"、"benchmark"、"加速比"、"性能测试"、"快速测速"。Use when comparing Triton operator performance against PyTorch native implementation using simple timing method.
---

# benchmark-comparison

Triton算子与PyTorch原生实现的性能对比测试，使用warmup+多次运行取平均的方式测量执行时间。

## When to Use

This skill is triggered when:

- 用户要求对比Triton和Torch的性能
- 需要快速测量算子执行时间
- 不需要详细的硬件性能指标
- 用户提到"benchmark"、"加速比"、"性能测试"

## 前置检查

**重要**: 在执行性能对比前，必须先确认输入内容包含Triton算子定义。

### 如何识别Triton算子

Triton算子通常具有以下特征：

1. **函数装饰器**: 使用 `@triton.jit` 或 `@triton.autotune`
2. **Kernel函数**: 包含 `tl.load`, `tl.store`, `tl.compute` 等Triton API
3. **编译调用**: 通过 `kernel[grid](...)` 方式启动

### 检查步骤

1. 阅读用户提供的代码或文件
2. 搜索以下Triton特征：
   - `@triton.jit` 装饰器
   - `def kernel_name(...):` - kernel函数定义
   - `tl.load`, `tl.store`, `tl.arange` - Triton API调用
   - `kernel[grid](...)` - Kernel启动模式
3. 如果确认是Triton算子，继续执行性能对比流程
4. 如果不是Triton算子，向用户说明当前skill只支持Triton算子分析

## Workflow

```
输入检查 → 环境确认 → 生成测试代码 → 运行测试 → 分析结果 → 输出报告
```

---

## Step 1: 环境确认

### 1.1 确认设备可用

```bash
python -c "import torch; print(torch.npu.is_available())"
```

### 1.2 确认Triton可用

```bash
python -c "import triton; print(triton.__version__)"
```

---

## Step 2: 生成测试代码

### 2.1 性能测试模板

生成测试脚本 `benchmark_comparison.py`：

```python
import torch
import time


def test_performance(func, *args, warmup=10, repeat=100, **kwargs):
    """
    测试函数执行性能
    
    Args:
        func: 要测试的函数
        *args: 函数参数
        warmup: 预热次数
        repeat: 重复测试次数
        **kwargs: 函数关键字参数
    
    Returns:
        avg_time_ms: 平均执行时间（毫秒）
    """
    sync_device = torch.npu if hasattr(torch, 'npu') and torch.npu.is_available() else torch.cuda
    
    # Warmup
    for _ in range(warmup):
        _ = func(*args, **kwargs)
    
    # Synchronize before timing
    sync_device.synchronize()
    
    # Benchmark
    start = time.time()
    for _ in range(repeat):
        _ = func(*args, **kwargs)
    sync_device.synchronize()
    end = time.time()
    
    avg_time_ms = (end - start) / repeat * 1000
    return avg_time_ms


def compare_performance(triton_func, torch_func, *args, warmup=10, repeat=100, **kwargs):
    """
    对比Triton和Torch函数的性能
    
    Args:
        triton_func: Triton算子函数
        torch_func: PyTorch参考实现函数
        *args: 函数参数
        warmup: 预热次数
        repeat: 重复测试次数
        **kwargs: 函数关键字参数
    
    Returns:
        dict: 包含triton_time, torch_time, speedup的字典
    """
    print("=" * 50)
    print("Performance Comparison")
    print("=" * 50)
    
    # Test Triton version
    print(f"\nTesting Triton version...")
    triton_time = test_performance(triton_func, *args, warmup=warmup, repeat=repeat, **kwargs)
    print(f"Triton average time: {triton_time:.3f} ms")
    
    # Test Torch version
    print(f"\nTesting Torch version...")
    torch_time = test_performance(torch_func, *args, warmup=warmup, repeat=repeat, **kwargs)
    print(f"Torch average time: {torch_time:.3f} ms")
    
    # Calculate speedup
    speedup = torch_time / triton_time
    improvement = (torch_time - triton_time) / torch_time * 100
    
    print(f"\n{'=' * 50}")
    print(f"Speedup: {speedup:.2f}x")
    print(f"Improvement: {improvement:.1f}%")
    print(f"{'=' * 50}")
    
    return {
        'triton_time_ms': triton_time,
        'torch_time_ms': torch_time,
        'speedup': speedup,
        'improvement_percent': improvement
    }


# ==================== 用户配置区域 ====================

# 导入用户的Triton算子
from your_module import triton_kernel


def torch_reference(x, y):
    """PyTorch参考实现"""
    return x + y


def run_benchmark():
    """运行性能对比测试"""
    # 设置测试参数
    BATCH_SIZE = 128
    SEQ_LEN = 1024
    HIDDEN_SIZE = 4096
    DTYPE = torch.float16
    
    # 准备输入数据
    x = torch.randn(BATCH_SIZE, SEQ_LEN, HIDDEN_SIZE, device='npu', dtype=DTYPE)
    y = torch.randn(BATCH_SIZE, SEQ_LEN, HIDDEN_SIZE, device='npu', dtype=DTYPE)
    
    # 运行对比测试
    results = compare_performance(
        triton_kernel,
        torch_reference,
        x, y,
        warmup=10,
        repeat=100
    )
    
    return results


if __name__ == "__main__":
    run_benchmark()
```

### 2.2 多形状测试模板

```python
import torch
import time


def benchmark_shapes(triton_func, torch_func, shapes, dtype=torch.float16, warmup=10, repeat=100):
    """
    测试多个形状的性能
    
    Args:
        triton_func: Triton算子函数
        torch_func: PyTorch参考实现函数
        shapes: 形状列表，每个元素是输入张量的形状元组
        dtype: 数据类型
        warmup: 预热次数
        repeat: 重复测试次数
    """
    sync_device = torch.npu if hasattr(torch, 'npu') and torch.npu.is_available() else torch.cuda
    
    print("=" * 80)
    print(f"{'Shape':<30} {'Triton (ms)':<15} {'Torch (ms)':<15} {'Speedup':<10}")
    print("=" * 80)
    
    results = []
    for shape in shapes:
        # 准备输入
        inputs = [torch.randn(*s, device='npu', dtype=dtype) for s in shape]
        
        # Warmup
        for _ in range(warmup):
            _ = triton_func(*inputs)
            _ = torch_func(*inputs)
        
        # Test Triton
        sync_device.synchronize()
        start = time.time()
        for _ in range(repeat):
            _ = triton_func(*inputs)
        sync_device.synchronize()
        triton_time = (time.time() - start) / repeat * 1000
        
        # Test Torch
        sync_device.synchronize()
        start = time.time()
        for _ in range(repeat):
            _ = torch_func(*inputs)
        sync_device.synchronize()
        torch_time = (time.time() - start) / repeat * 1000
        
        speedup = torch_time / triton_time
        results.append({
            'shape': shape,
            'triton_time_ms': triton_time,
            'torch_time_ms': torch_time,
            'speedup': speedup
        })
        
        shape_str = str(shape)
        print(f"{shape_str:<30} {triton_time:<15.3f} {torch_time:<15.3f} {speedup:<10.2f}x")
    
    print("=" * 80)
    return results


# 使用示例
if __name__ == "__main__":
    from your_module import triton_kernel
    
    def torch_reference(x, y):
        return x + y
    
    shapes = [
        ((128, 1024), (128, 1024)),
        ((256, 1024), (256, 1024)),
        ((512, 1024), (512, 1024)),
        ((1024, 1024), (1024, 1024)),
    ]
    
    benchmark_shapes(triton_kernel, torch_reference, shapes)
```

### 2.3 多数据类型测试模板

```python
import torch
import time


def benchmark_dtypes(triton_func, torch_func, shape, dtypes=[torch.float16, torch.bfloat16, torch.float32], warmup=10, repeat=100):
    """
    测试多种数据类型的性能
    
    Args:
        triton_func: Triton算子函数
        torch_func: PyTorch参考实现函数
        shape: 输入张量形状
        dtypes: 数据类型列表
        warmup: 预热次数
        repeat: 重复测试次数
    """
    sync_device = torch.npu if hasattr(torch, 'npu') and torch.npu.is_available() else torch.cuda
    
    print("=" * 80)
    print(f"{'Dtype':<15} {'Triton (ms)':<15} {'Torch (ms)':<15} {'Speedup':<10}")
    print("=" * 80)
    
    results = []
    for dtype in dtypes:
        # 准备输入
        x = torch.randn(*shape, device='npu', dtype=dtype)
        y = torch.randn(*shape, device='npu', dtype=dtype)
        
        # Warmup
        for _ in range(warmup):
            _ = triton_func(x, y)
            _ = torch_func(x, y)
        
        # Test Triton
        sync_device.synchronize()
        start = time.time()
        for _ in range(repeat):
            _ = triton_func(x, y)
        sync_device.synchronize()
        triton_time = (time.time() - start) / repeat * 1000
        
        # Test Torch
        sync_device.synchronize()
        start = time.time()
        for _ in range(repeat):
            _ = torch_func(x, y)
        sync_device.synchronize()
        torch_time = (time.time() - start) / repeat * 1000
        
        speedup = torch_time / triton_time
        results.append({
            'dtype': str(dtype),
            'triton_time_ms': triton_time,
            'torch_time_ms': torch_time,
            'speedup': speedup
        })
        
        print(f"{str(dtype):<15} {triton_time:<15.3f} {torch_time:<15.3f} {speedup:<10.2f}x")
    
    print("=" * 80)
    return results
```

---

## Step 3: 运行测试

### 3.1 执行测试脚本

```bash
python benchmark_comparison.py
```

### 3.2 测试参数说明

| 参数 | 说明 | 推荐值 |
|------|------|-------|
| `warmup` | 预热次数，用于消除首次运行的额外开销 | 10-20 |
| `repeat` | 重复测试次数，用于取平均值 | 100 |

### 3.3 关键注意事项

1. **必须使用 synchronize()**: 在计时前后调用 `torch.npu.synchronize()` 确保所有操作完成
2. **预热很重要**: 首次运行可能有额外开销（编译、内存分配等）
3. **多次取平均**: 单次测试不稳定，需要多次运行取平均值

---

## Step 4: 结果分析

### 4.1 性能对比判断

| Speedup | 结论 | 建议 |
|---------|------|------|
| > 1.5x | Triton显著优于Torch | 实现有效 |
| 1.0x - 1.5x | Triton略优于Torch | 可考虑进一步优化 |
| 0.8x - 1.0x | 性能相当 | 评估是否值得迁移 |
| < 0.8x | Torch更优 | 检查Triton实现 |

### 4.2 常见问题诊断

| 问题 | 可能原因 | 解决方案 |
|------|---------|---------|
| 首次运行很慢 | JIT编译开销 | 增加warmup次数 |
| 时间波动大 | 系统负载不稳定 | 增加repeat次数 |
| Triton比Torch慢 | 实现问题 | 使用msprof-profiling分析瓶颈 |

---

## Step 5: 输出报告

```markdown
# 性能对比报告

## 基本信息
- 算子名称: <op_name>
- 测试形状: <shape>
- 数据类型: <dtype>
- 测试时间: <timestamp>
- 测试方法: warmup + 多次运行取平均

---

## 性能对比结果

| 版本 | 平均时间 (ms) |
|------|--------------|
| Triton | ... ms |
| Torch | ... ms |

### 性能提升

- **加速比**: ...x
- **提升百分比**: ...%

---

## 结论

<根据对比结果填写>

---

## 建议

- 如果加速比 > 1.5x: 实现有效，可以考虑部署
- 如果加速比 < 1.0x: 建议使用 `msprof-profiling` 技能进行深度分析
```

---

## 与其他组件的协作

```
verify-precision skill (精度验证通过)
        ↓
benchmark-comparison skill (快速性能对比) ← 当前
        ↓
msprof-profiling skill (如需深度分析)
        ↓
triton-expert agent (根据分析结果优化)
```

**对比完成后**:
- 如果性能满意，可以结束
- 如果需要深入分析瓶颈，使用 `msprof-profiling` 技能

---

## 与 msprof-profiling 的区别

| 特性 | benchmark-comparison | msprof-profiling |
|------|---------------------|------------------|
| 工具 | time.time() + synchronize | msprof |
| 指标 | 仅时间对比 | 详细硬件指标 |
| 用途 | 快速对比、回归测试 | 深度分析、瓶颈定位 |
| 输出 | 简单加速比 | 完整性能报告 |
| 依赖 | 无特殊依赖 | 需要msprof工具 |

**选择建议**:
- 只需快速对比性能 → 使用 `benchmark-comparison`
- 需要详细分析瓶颈 → 使用 `msprof-profiling`
