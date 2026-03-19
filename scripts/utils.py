"""
Core evaluation module for testing PyTorch vs Triton kernels
"""

import os
import sys
import tempfile
import importlib
import importlib.util
import time
from typing import Tuple, Dict, Any, Optional, List
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch_npu


@dataclass
class EvalResult:
    compiled: bool = False
    correctness: bool = False
    max_diff: float = 0.0
    avg_diff: float = 0.0
    pytorch_time: float = 0.0
    triton_time: float = 0.0
    speedup: float = 0.0
    error_message: str = ""


# 默认配置
DEFAULT_PRECISION = "fp16"
DEFAULT_DEVICE = "npu"
DEFAULT_TOLERANCE = 1e-2  # fp16 tolerance


def load_pytorch_model(file_path: str) -> Tuple[nn.Module, callable, callable]:
    """Load PyTorch model from file"""
    spec = importlib.util.spec_from_file_location("pytorch_model", file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    Model = module.Model
    get_inputs = module.get_inputs
    get_init_inputs = module.get_init_inputs

    init_inputs = get_init_inputs()
    model = Model(*init_inputs)

    return model, get_inputs, get_init_inputs


def load_triton_model(file_path: str, entry_point: str = "ModelNew") -> nn.Module:
    """Load Triton model from file"""
    spec = importlib.util.spec_from_file_location("triton_model", file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    ModelNew = getattr(module, entry_point)
    model = ModelNew()

    return model


def check_correctness_once(
    pytorch_model: nn.Module,
    triton_model: nn.Module,
    get_inputs: callable,
    verbose: bool = True,
) -> Tuple[bool, float, float, str]:
    """Check correctness once between PyTorch and Triton implementations"""

    dtype = torch.float16
    tolerance = DEFAULT_TOLERANCE
    device = DEFAULT_DEVICE

    pytorch_model = pytorch_model.to(device=device, dtype=dtype)
    triton_model = triton_model.to(device=device, dtype=dtype)

    pytorch_model.eval()
    triton_model.eval()

    try:
        with torch.no_grad():
            inputs = get_inputs()
            inputs = [x.to(device=device, dtype=dtype) for x in inputs]

            torch.npu.synchronize()
            pytorch_output = pytorch_model(*inputs)
            torch.npu.synchronize()

            torch.npu.synchronize()
            triton_output = triton_model(*inputs)
            torch.npu.synchronize()

            if pytorch_output.shape != triton_output.shape:
                error_msg = f"Shape mismatch: PyTorch {pytorch_output.shape} vs Triton {triton_output.shape}"
                if verbose:
                    print(f"[FAIL] {error_msg}")
                return False, 0.0, 0.0, error_msg

            max_diff = torch.max(torch.abs(pytorch_output - triton_output)).item()
            avg_diff = torch.mean(torch.abs(pytorch_output - triton_output)).item()

            if not torch.allclose(pytorch_output, triton_output, atol=tolerance, rtol=tolerance):
                if verbose:
                    print(f"[FAIL] max_diff={max_diff:.6f}, avg_diff={avg_diff:.6f}")
                return False, max_diff, avg_diff, ""
            else:
                if verbose:
                    print(f"[PASS] max_diff={max_diff:.6f}, avg_diff={avg_diff:.6f}")
                return True, max_diff, avg_diff, ""

    except Exception as e:
        error_msg = f"Runtime error: {str(e)}"
        if verbose:
            print(f"[ERROR] {error_msg}")
        return False, 0.0, 0.0, error_msg


def measure_performance(
    model: nn.Module,
    get_inputs: callable,
    num_trials: int = 100,
    warmup: int = 10,
    verbose: bool = True,
) -> float:
    """Measure average runtime of a model"""

    dtype = torch.float16
    device = DEFAULT_DEVICE

    model = model.to(device=device, dtype=dtype)
    model.eval()

    with torch.no_grad():
        inputs = get_inputs()
        inputs = [x.to(device=device, dtype=dtype) for x in inputs]

        # Warmup
        if verbose:
            print(f"  Warmup ({warmup} iterations)...")
        for _ in range(warmup):
            _ = model(*inputs)
        torch.npu.synchronize()

        # Measure
        if verbose:
            print(f"  Measuring ({num_trials} iterations)...")
        times = []
        for _ in range(num_trials):
            torch.npu.synchronize()
            start = time.perf_counter()
            _ = model(*inputs)
            torch.npu.synchronize()
            end = time.perf_counter()
            times.append((end - start) * 1000)  # Convert to ms

    avg_time = sum(times) / len(times)
    return avg_time


def eval_single_operator(
    pytorch_file: str,
    triton_file: str,
    verbose: bool = True,
) -> EvalResult:
    """
    Evaluate a single operator: PyTorch vs Triton
    First check correctness (1 trial), then measure performance if passed.

    Args:
        pytorch_file: Path to PyTorch kernel file
        triton_file: Path to Triton kernel file
        verbose: Print verbose output

    Returns:
        EvalResult with evaluation metrics
    """
    result = EvalResult()

    try:
        if verbose:
            print(f"Loading PyTorch model from: {pytorch_file}")
        pytorch_model, get_inputs, _ = load_pytorch_model(pytorch_file)

        if verbose:
            print(f"Loading Triton model from: {triton_file}")
        triton_model = load_triton_model(triton_file)

        result.compiled = True

    except Exception as e:
        result.error_message = f"Compilation failed: {str(e)}"
        if verbose:
            print(f"[ERROR] {result.error_message}")
        return result

    # Step 1: Check correctness (1 trial)
    try:
        if verbose:
            print(f"\n[1/2] Checking correctness...")
        correctness, max_diff, avg_diff, error_msg = check_correctness_once(
            pytorch_model,
            triton_model,
            get_inputs,
            verbose=verbose,
        )
        result.correctness = correctness
        result.max_diff = max_diff
        result.avg_diff = avg_diff

        if error_msg:
            result.error_message = error_msg
            return result

        if not correctness:
            result.error_message = f"Correctness check failed: max_diff={max_diff:.6f}"
            return result

    except Exception as e:
        result.error_message = f"Correctness check error: {str(e)}"
        if verbose:
            print(f"[ERROR] {result.error_message}")
        return result

    # Step 2: Measure performance (only if correctness passed)
    try:
        if verbose:
            print(f"\n[2/2] Measuring performance...")
            print(f"  PyTorch:")
        result.pytorch_time = measure_performance(
            pytorch_model, get_inputs, verbose=verbose
        )

        if verbose:
            print(f"  Triton:")
        result.triton_time = measure_performance(
            triton_model, get_inputs, verbose=verbose
        )

        if result.triton_time > 0:
            result.speedup = result.pytorch_time / result.triton_time

        if verbose:
            print(f"\n  Performance Results:")
            print(f"    PyTorch: {result.pytorch_time:.4f} ms")
            print(f"    Triton:  {result.triton_time:.4f} ms")
            print(f"    Speedup: {result.speedup:.2f}x")

    except Exception as e:
        result.error_message = f"Performance measurement error: {str(e)}"
        if verbose:
            print(f"[ERROR] {result.error_message}")

    return result


def get_available_levels(pytorch_dir: str) -> List[str]:
    """Get all available levels from pytorch directory"""
    levels = []
    if os.path.exists(pytorch_dir):
        for item in os.listdir(pytorch_dir):
            item_path = os.path.join(pytorch_dir, item)
            if os.path.isdir(item_path) and item.startswith("level"):
                levels.append(item)
    return sorted(levels)


def eval_level(
    pytorch_dir: str,
    triton_dir: str,
    level: str,
    verbose: bool = False,
) -> Dict[str, EvalResult]:
    """
    Evaluate all operators for a specific level

    Args:
        pytorch_dir: Base directory for PyTorch kernels
        triton_dir: Base directory for Triton kernels
        level: Level name (e.g., "level1", "level2")
        verbose: Print verbose output

    Returns:
        Dictionary mapping operator name to EvalResult
    """
    results = {}

    level_pytorch_dir = os.path.join(pytorch_dir, level)
    level_triton_dir = os.path.join(triton_dir, level)

    if not os.path.exists(level_pytorch_dir):
        print(f"[WARN] Level directory not found: {level_pytorch_dir}")
        return results

    pytorch_files = sorted([f for f in os.listdir(level_pytorch_dir) if f.endswith(".py")])

    if not pytorch_files:
        print(f"[WARN] No PyTorch kernel files found in {level_pytorch_dir}")
        return results

    print(f"\n{'='*70}")
    print(f"Level: {level}")
    print(f"Found {len(pytorch_files)} operators")
    print(f"{'='*70}")

    for pytorch_file in pytorch_files:
        operator_name = pytorch_file.replace(".py", "")
        triton_file = os.path.join(level_triton_dir, pytorch_file)

        if not os.path.exists(triton_file):
            if verbose:
                print(f"[SKIP] {operator_name}: Triton file not found")
            continue

        print(f"\n--- Evaluating: {operator_name} ---")

        result = eval_single_operator(
            os.path.join(level_pytorch_dir, pytorch_file),
            triton_file,
            verbose=verbose,
        )

        results[operator_name] = result

    return results


def eval_all_levels(
    pytorch_dir: str,
    triton_dir: str,
    levels: Optional[List[str]] = None,
    verbose: bool = False,
) -> Dict[str, Dict[str, EvalResult]]:
    """
    Evaluate all operators across all levels

    Args:
        pytorch_dir: Base directory for PyTorch kernels
        triton_dir: Base directory for Triton kernels
        levels: List of levels to evaluate, or None for all
        verbose: Print verbose output

    Returns:
        Nested dictionary: {level: {operator_name: EvalResult}}
    """
    all_results = {}

    if levels is None:
        levels = get_available_levels(pytorch_dir)

    for level in levels:
        print(f"\nEvaluating level: {level}")
        level_results = eval_level(
            pytorch_dir,
            triton_dir,
            level,
            verbose=verbose,
        )
        all_results[level] = level_results

    return all_results


def print_summary(all_results: Dict[str, Dict[str, EvalResult]]) -> None:
    """Print summary of evaluation results"""
    print("\n" + "=" * 80)
    print("EVALUATION SUMMARY")
    print("=" * 80)

    total_operators = 0
    total_correct = 0

    for level, results in all_results.items():
        print(f"\n--- Level: {level} ---")

        if not results:
            print("  No operators evaluated")
            continue

        level_total = len(results)
        level_correct = sum(1 for r in results.values() if r.correctness)

        total_operators += level_total
        total_correct += level_correct

        print(f"  Operators: {level_total}, Correct: {level_correct} ({100*level_correct/level_total:.1f}%)")

        print(f"\n  {'Operator':<25} {'Correct':<8} {'PyTorch(ms)':<12} {'Triton(ms)':<12} {'Speedup':<10}")
        print(f"  {'-'*75}")

        for name, result in sorted(results.items()):
            correct_str = "pass" if result.correctness else "fail"
            pytorch_time_str = f"{result.pytorch_time:.4f}" if result.pytorch_time > 0 else "-"
            triton_time_str = f"{result.triton_time:.4f}" if result.triton_time > 0 else "-"
            speedup_str = f"{result.speedup:.2f}x" if result.speedup > 0 else "-"
            print(f"  {name:<25} {correct_str:<8} {pytorch_time_str:<12} {triton_time_str:<12} {speedup_str:<10}")

    print("\n" + "=" * 80)
    print("OVERALL SUMMARY")
    print("=" * 80)
    print(f"Total operators: {total_operators}")
    print(f"Correct:         {total_correct} ({100*total_correct/total_operators:.1f}%)")
    print("=" * 80)
