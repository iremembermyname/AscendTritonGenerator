"""
Profiler for Triton Performance Analysis

Provides tools for profiling and analyzing Triton kernel performance on Ascend NPU.
"""

import subprocess
import json
import os
import re
import time
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum


class MetricType(Enum):
    TASK_DURATION = "task_duration"
    MTE_UTILIZATION = "mte_utilization"
    VECTOR_UTILIZATION = "vector_utilization"
    UB_USAGE = "ub_usage"
    MEMORY_BANDWIDTH = "memory_bandwidth"


@dataclass
class PerformanceMetrics:
    task_duration_us: float
    mte_utilization: float
    vector_utilization: float
    ub_usage_kb: float
    memory_bandwidth_gbps: float
    compute_throughput: float
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_duration_us": self.task_duration_us,
            "mte_utilization": self.mte_utilization,
            "vector_utilization": self.vector_utilization,
            "ub_usage_kb": self.ub_usage_kb,
            "memory_bandwidth_gbps": self.memory_bandwidth_gbps,
            "compute_throughput": self.compute_throughput,
        }


def benchmark_kernel(
    kernel_fn,
    inputs: List[Any],
    warmup: int = 20,
    repeats: int = 100,
    synchronize: bool = True,
) -> Dict[str, float]:
    import torch
    
    for _ in range(warmup):
        _ = kernel_fn(*inputs)
    
    if synchronize:
        torch.npu.synchronize()
    
    start = time.perf_counter()
    for _ in range(repeats):
        _ = kernel_fn(*inputs)
    if synchronize:
        torch.npu.synchronize()
    end = time.perf_counter()
    
    total_time_s = end - start
    avg_time_ms = total_time_s / repeats * 1000
    avg_time_us = avg_time_ms * 1000
    
    return {
        "total_time_s": total_time_s,
        "avg_time_ms": avg_time_ms,
        "avg_time_us": avg_time_us,
        "repeats": repeats,
    }


def run_msprof(
    script_path: str,
    kernel_name: str,
    output_dir: str,
    warmup: int = 20,
    launch_count: int = 20,
) -> Dict[str, Any]:
    cmd = [
        "msprof", "op",
        f"--output={output_dir}",
        f"--kernel-name={kernel_name}",
        f"--warm-up={warmup}",
        f"--launch-count={launch_count}",
        "python", script_path,
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    return {
        "returncode": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
        "output_dir": output_dir,
    }


def parse_msprof_output(output_dir: str) -> PerformanceMetrics:
    summary_file = Path(output_dir) / "summary.json"
    
    if summary_file.exists():
        with open(summary_file) as f:
            data = json.load(f)
        
        return PerformanceMetrics(
            task_duration_us=data.get("Task Duration", 0),
            mte_utilization=data.get("MTE Utilization", 0),
            vector_utilization=data.get("Vector Utilization", 0),
            ub_usage_kb=data.get("UB Usage", 0),
            memory_bandwidth_gbps=data.get("Memory Bandwidth", 0),
            compute_throughput=data.get("Compute Throughput", 0),
        )
    
    return PerformanceMetrics(
        task_duration_us=0,
        mte_utilization=0,
        vector_utilization=0,
        ub_usage_kb=0,
        memory_bandwidth_gbps=0,
        compute_throughput=0,
    )


def analyze_performance(
    metrics: PerformanceMetrics,
    baseline: Optional[PerformanceMetrics] = None,
) -> Dict[str, Any]:
    analysis = {
        "metrics": metrics.to_dict(),
        "bottlenecks": [],
        "suggestions": [],
    }
    
    if metrics.task_duration_us > 1000:
        analysis["bottlenecks"].append({
            "type": "high_latency",
            "description": f"Task duration is high: {metrics.task_duration_us:.2f} us",
            "suggestion": "Consider optimizing kernel or increasing parallelism",
        })
    
    if metrics.mte_utilization < 0.5:
        analysis["bottlenecks"].append({
            "type": "low_mte_utilization",
            "description": f"MTE utilization is low: {metrics.mte_utilization:.2%}",
            "suggestion": "Check memory access patterns and pipeline efficiency",
        })
    
    if metrics.vector_utilization < 0.5:
        analysis["bottlenecks"].append({
            "type": "low_vector_utilization",
            "description": f"Vector utilization is low: {metrics.vector_utilization:.2%}",
            "suggestion": "Increase computational density or reduce control flow",
        })
    
    if metrics.ub_usage_kb > 85:
        analysis["bottlenecks"].append({
            "type": "ub_overflow",
            "description": f"UB usage exceeds safe limit: {metrics.ub_usage_kb:.2f} KB",
            "suggestion": "Reduce block size or number of intermediate variables",
        })
    
    if baseline:
        speedup = baseline.task_duration_us / metrics.task_duration_us if metrics.task_duration_us > 0 else 0
        analysis["comparison"] = {
            "speedup": speedup,
            "duration_improvement": baseline.task_duration_us - metrics.task_duration_us,
            "mte_improvement": metrics.mte_utilization - baseline.mte_utilization,
            "vector_improvement": metrics.vector_utilization - baseline.vector_utilization,
        }
    
    return analysis


def estimate_ub_usage(
    block_size: int,
    num_inputs: int,
    num_outputs: int,
    num_intermediates: int,
    dtype_bytes: int = 2,
) -> float:
    total_elements = block_size * (num_inputs + num_outputs + num_intermediates)
    ub_bytes = total_elements * dtype_bytes
    ub_kb = ub_bytes / 1024
    return ub_kb


def calculate_optimal_block_size(
    ub_limit_kb: float = 85,
    num_inputs: int = 1,
    num_outputs: int = 1,
    num_intermediates: int = 1,
    dtype_bytes: int = 2,
) -> int:
    total_tensors = num_inputs + num_outputs + num_intermediates
    ub_limit_bytes = ub_limit_kb * 1024
    max_elements = ub_limit_bytes // (total_tensors * dtype_bytes)
    
    power_of_2 = 1
    while power_of_2 * 2 <= max_elements:
        power_of_2 *= 2
    
    return min(power_of_2, 1024)


def profile_kernel_iterations(
    kernel_fn,
    inputs: List[Any],
    iteration_counts: List[int] = [10, 50, 100, 200],
) -> Dict[str, List[float]]:
    results = {
        "iterations": [],
        "times_ms": [],
    }
    
    for n in iteration_counts:
        benchmark_result = benchmark_kernel(kernel_fn, inputs, warmup=10, repeats=n)
        results["iterations"].append(n)
        results["times_ms"].append(benchmark_result["avg_time_ms"])
    
    return results


def generate_performance_report(
    metrics: PerformanceMetrics,
    analysis: Dict[str, Any],
    operator_name: str = "unknown",
) -> str:
    report = f"""# 性能分析报告

## 算子信息
- 算子名称: {operator_name}

## 性能指标

| 指标 | 值 |
|------|-----|
| 任务耗时 | {metrics.task_duration_us:.2f} us |
| MTE利用率 | {metrics.mte_utilization:.2%} |
| Vector利用率 | {metrics.vector_utilization:.2%} |
| UB使用量 | {metrics.ub_usage_kb:.2f} KB |
| 内存带宽 | {metrics.memory_bandwidth_gbps:.2f} GB/s |

## 性能瓶颈

"""
    
    if analysis["bottlenecks"]:
        for bottleneck in analysis["bottlenecks"]:
            report += f"### {bottleneck['type']}\n\n"
            report += f"- **描述**: {bottleneck['description']}\n"
            report += f"- **建议**: {bottleneck['suggestion']}\n\n"
    else:
        report += "未发现明显性能瓶颈。\n"
    
    if "comparison" in analysis:
        comp = analysis["comparison"]
        report += f"""## 性能对比

| 指标 | 变化 |
|------|------|
| 加速比 | {comp['speedup']:.2f}x |
| 耗时改善 | {comp['duration_improvement']:.2f} us |
| MTE改善 | {comp['mte_improvement']:.2%} |
| Vector改善 | {comp['vector_improvement']:.2%} |

"""
    
    return report
