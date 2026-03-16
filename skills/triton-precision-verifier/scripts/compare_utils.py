"""
Comparison Utilities for Precision Verification

Provides functions for comparing outputs and analyzing precision differences.
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum


class ComparisonResult(Enum):
    PASS = "pass"
    FAIL = "fail"
    WARNING = "warning"


@dataclass
class PrecisionMetrics:
    max_abs_error: float
    mean_abs_error: float
    max_rel_error: float
    mean_rel_error: float
    rmse: float
    mae: float
    passed: bool
    result: ComparisonResult


def compare_tensors(
    output: torch.Tensor,
    expected: torch.Tensor,
    rtol: float = 1e-3,
    atol: float = 1e-3,
) -> PrecisionMetrics:
    output = output.float()
    expected = expected.float()
    
    abs_error = torch.abs(output - expected)
    rel_error = abs_error / (torch.abs(expected) + 1e-10)
    
    max_abs_error = abs_error.max().item()
    mean_abs_error = abs_error.mean().item()
    max_rel_error = rel_error.max().item()
    mean_rel_error = rel_error.mean().item()
    
    rmse = torch.sqrt((abs_error ** 2).mean()).item()
    mae = mean_abs_error
    
    passed = torch.allclose(output, expected, rtol=rtol, atol=atol)
    
    if passed:
        result = ComparisonResult.PASS
    elif max_rel_error < 0.1:
        result = ComparisonResult.WARNING
    else:
        result = ComparisonResult.FAIL
    
    return PrecisionMetrics(
        max_abs_error=max_abs_error,
        mean_abs_error=mean_abs_error,
        max_rel_error=max_rel_error,
        mean_rel_error=mean_rel_error,
        rmse=rmse,
        mae=mae,
        passed=passed,
        result=result,
    )


def check_nan_inf(tensor: torch.Tensor) -> Dict[str, Any]:
    nan_count = torch.isnan(tensor).sum().item()
    inf_count = torch.isinf(tensor).sum().item()
    
    return {
        "has_nan": nan_count > 0,
        "has_inf": inf_count > 0,
        "nan_count": nan_count,
        "inf_count": inf_count,
        "total_elements": tensor.numel(),
        "nan_ratio": nan_count / tensor.numel(),
        "inf_ratio": inf_count / tensor.numel(),
    }


def find_large_error_positions(
    output: torch.Tensor,
    expected: torch.Tensor,
    threshold: float = 1e-2,
    max_samples: int = 10,
) -> Dict[str, Any]:
    abs_error = torch.abs(output.float() - expected.float())
    large_error_mask = abs_error > threshold
    
    large_error_count = large_error_mask.sum().item()
    
    if large_error_count == 0:
        return {
            "count": 0,
            "positions": [],
            "errors": [],
        }
    
    positions = torch.nonzero(large_error_mask)
    errors = abs_error[large_error_mask]
    
    sorted_indices = torch.argsort(errors, descending=True)
    top_positions = positions[sorted_indices[:max_samples]]
    top_errors = errors[sorted_indices[:max_samples]]
    
    return {
        "count": large_error_count,
        "positions": top_positions.tolist(),
        "errors": top_errors.tolist(),
        "max_error": errors.max().item(),
        "mean_error": errors.mean().item(),
    }


def analyze_error_distribution(
    output: torch.Tensor,
    expected: torch.Tensor,
) -> Dict[str, Any]:
    abs_error = torch.abs(output.float() - expected.float())
    
    percentiles = [0.5, 0.9, 0.95, 0.99, 0.999, 1.0]
    percentile_values = torch.quantile(abs_error, torch.tensor(percentiles))
    
    histogram = torch.histc(abs_error, bins=100)
    bin_edges = torch.linspace(0, abs_error.max(), 101)
    
    return {
        "percentiles": {
            f"{int(p*100)}%": v.item()
            for p, v in zip(percentiles, percentile_values)
        },
        "histogram": histogram.tolist(),
        "bin_edges": bin_edges.tolist(),
        "mean": abs_error.mean().item(),
        "std": abs_error.std().item(),
        "min": abs_error.min().item(),
        "max": abs_error.max().item(),
    }


def compare_with_reference(
    output: torch.Tensor,
    reference_fn,
    *args,
    rtol: float = 1e-3,
    atol: float = 1e-3,
    **kwargs,
) -> Dict[str, Any]:
    with torch.no_grad():
        expected = reference_fn(*args, **kwargs)
    
    metrics = compare_tensors(output, expected, rtol, atol)
    nan_inf = check_nan_inf(output)
    large_errors = find_large_error_positions(output, expected)
    distribution = analyze_error_distribution(output, expected)
    
    return {
        "metrics": metrics,
        "nan_inf": nan_inf,
        "large_errors": large_errors,
        "distribution": distribution,
    }


def batch_compare(
    outputs: List[torch.Tensor],
    expecteds: List[torch.Tensor],
    rtol: float = 1e-3,
    atol: float = 1e-3,
) -> Dict[str, Any]:
    assert len(outputs) == len(expecteds), "Output and expected lists must have same length"
    
    results = []
    all_passed = True
    
    for i, (output, expected) in enumerate(zip(outputs, expecteds)):
        metrics = compare_tensors(output, expected, rtol, atol)
        results.append({
            "index": i,
            "shape": list(output.shape),
            "metrics": metrics,
        })
        if not metrics.passed:
            all_passed = False
    
    return {
        "all_passed": all_passed,
        "total_tests": len(outputs),
        "passed_tests": sum(1 for r in results if r["metrics"].passed),
        "results": results,
    }


def generate_comparison_report(
    comparison_result: Dict[str, Any],
    operator_name: str = "unknown",
) -> str:
    metrics = comparison_result.get("metrics", {})
    nan_inf = comparison_result.get("nan_inf", {})
    large_errors = comparison_result.get("large_errors", {})
    distribution = comparison_result.get("distribution", {})
    
    report = f"""# 精度对比报告

## 算子信息
- 算子名称: {operator_name}

## 验证结果
- 状态: {'通过' if metrics.get('passed', False) else '失败'}

## 误差统计

| 指标 | 值 |
|------|-----|
| 最大绝对误差 | {metrics.get('max_abs_error', 'N/A'):.6e} |
| 平均绝对误差 | {metrics.get('mean_abs_error', 'N/A'):.6e} |
| 最大相对误差 | {metrics.get('max_rel_error', 'N/A'):.6e} |
| 平均相对误差 | {metrics.get('mean_rel_error', 'N/A'):.6e} |
| RMSE | {metrics.get('rmse', 'N/A'):.6e} |
| MAE | {metrics.get('mae', 'N/A'):.6e} |

## NaN/Inf 检查

| 指标 | 值 |
|------|-----|
| 包含NaN | {'是' if nan_inf.get('has_nan', False) else '否'} |
| NaN数量 | {nan_inf.get('nan_count', 0)} |
| 包含Inf | {'是' if nan_inf.get('has_inf', False) else '否'} |
| Inf数量 | {nan_inf.get('inf_count', 0)} |

## 大误差位置

"""
    
    if large_errors.get("count", 0) > 0:
        report += f"- 大误差数量: {large_errors['count']}\n"
        report += f"- 最大误差: {large_errors['max_error']:.6e}\n"
        report += f"- 平均误差: {large_errors['mean_error']:.6e}\n"
    else:
        report += "- 未发现大误差\n"
    
    report += "\n## 误差分布\n\n"
    
    percentiles = distribution.get("percentiles", {})
    for p, v in percentiles.items():
        report += f"- {p}: {v:.6e}\n"
    
    return report


def diagnose_precision_issue(
    output: torch.Tensor,
    expected: torch.Tensor,
) -> List[Dict[str, Any]]:
    issues = []
    
    nan_inf = check_nan_inf(output)
    if nan_inf["has_nan"]:
        issues.append({
            "type": "nan_detected",
            "severity": "high",
            "description": f"Output contains {nan_inf['nan_count']} NaN values",
            "suggestion": "Check for division by zero, log of negative values, or numerical overflow",
        })
    
    if nan_inf["has_inf"]:
        issues.append({
            "type": "inf_detected",
            "severity": "high",
            "description": f"Output contains {nan_inf['inf_count']} Inf values",
            "suggestion": "Check for numerical overflow or missing normalization",
        })
    
    metrics = compare_tensors(output, expected)
    
    if metrics.max_rel_error > 0.1:
        issues.append({
            "type": "large_relative_error",
            "severity": "high",
            "description": f"Maximum relative error is {metrics.max_rel_error:.2%}",
            "suggestion": "Check algorithm correctness or numerical stability",
        })
    
    if metrics.mean_rel_error > 0.01:
        issues.append({
            "type": "high_mean_error",
            "severity": "medium",
            "description": f"Mean relative error is {metrics.mean_rel_error:.2%}",
            "suggestion": "Consider using higher precision or adjusting algorithm",
        })
    
    large_errors = find_large_error_positions(output, expected, threshold=0.01)
    if large_errors["count"] > output.numel() * 0.01:
        issues.append({
            "type": "widespread_large_errors",
            "severity": "medium",
            "description": f"{large_errors['count']} positions ({large_errors['count']/output.numel()*100:.2f}%) have error > 1%",
            "suggestion": "Review algorithm implementation for systematic errors",
        })
    
    return issues
