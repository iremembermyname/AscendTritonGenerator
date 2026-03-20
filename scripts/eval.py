"""
Evaluation Script for Triton Operators
评测 Triton 算子

Usage:
    python scripts/eval.py                              # 评测所有 level
    python scripts/eval.py --level level1               # 评测指定 level
    python scripts/eval.py --level level1 --operator 01_relu   # 评测单个算子
    python scripts/eval.py --list                       # 列出所有可用 levels
    python scripts/eval.py --verbose                    # 详细输出
    python scripts/eval.py --save-csv                   # 保存结果到CSV文件
"""

import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.utils import (
    eval_level,
    eval_all_levels,
    eval_single_operator,
    print_summary,
    save_results_to_csv,
    get_available_levels,
    EvalResult,
)


def main():
    parser = argparse.ArgumentParser(description="Evaluate Triton operators")

    parser.add_argument(
        "--level",
        type=str,
        default=None,
        help="Specific level to evaluate (e.g., level1, level2). If not specified, evaluates all levels.",
    )
    parser.add_argument(
        "--operator",
        type=str,
        default=None,
        help="Specific operator to evaluate (e.g., 01_relu). Must be used with --level.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available levels",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Verbose output",
    )
    parser.add_argument(
        "--save-csv",
        action="store_true",
        help="Save evaluation results to CSV file",
    )

    args = parser.parse_args()

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    pytorch_dir = os.path.join(base_dir, "pytorch_kernels")
    triton_dir = os.path.join(base_dir, "triton_kernels")

    print(f"PyTorch dir: {pytorch_dir}")
    print(f"Triton dir:  {triton_dir}")
    print(f"Precision:   fp16")
    print(f"Device:      npu")

    if args.list:
        levels = get_available_levels(pytorch_dir)
        print("\nAvailable levels:")
        for level in levels:
            print(f"  - {level}")
        return 0

    # 评测单个算子
    if args.level and args.operator:
        pytorch_file = os.path.join(pytorch_dir, args.level, f"{args.operator}.py")
        triton_file = os.path.join(triton_dir, args.level, f"{args.operator}.py")

        if not os.path.exists(pytorch_file):
            print(f"[ERROR] PyTorch file not found: {pytorch_file}")
            return 1
        if not os.path.exists(triton_file):
            print(f"[ERROR] Triton file not found: {triton_file}")
            return 1

        print(f"\n{'='*70}")
        print(f"Evaluating single operator: {args.level}/{args.operator}")
        print(f"{'='*70}")

        result = eval_single_operator(
            pytorch_file=pytorch_file,
            triton_file=triton_file,
            verbose=args.verbose,
        )

        # 打印单个算子结果
        print("\n" + "=" * 70)
        print("RESULT")
        print("=" * 70)
        print(f"Compiled:    {result.compiled}")
        print(f"Correct:     {result.correctness}")
        print(f"Max Diff:    {result.max_diff:.2e}")
        print(f"Avg Diff:    {result.avg_diff:.2e}")
        if result.pytorch_time > 0:
            print(f"PyTorch:     {result.pytorch_time:.4f} ms")
            print(f"Triton:      {result.triton_time:.4f} ms")
            print(f"Speedup:     {result.speedup:.2f}x")
        if result.error_message:
            print(f"Error:       {result.error_message}")
        print("=" * 70)

        # 保存单个算子结果到CSV（如果指定了--save-csv）
        if args.save_csv:
            results = {args.level: {args.operator: result}}
            csv_path = save_results_to_csv(results, output_dir=base_dir)
            print(f"\nResults saved to: {csv_path}")

        return 0 if result.correctness else 1

    # 评测指定level
    if args.level:
        results = {}
        results[args.level] = eval_level(
            pytorch_dir=pytorch_dir,
            triton_dir=triton_dir,
            level=args.level,
            verbose=args.verbose,
        )
    else:
        # 评测所有level
        results = eval_all_levels(
            pytorch_dir=pytorch_dir,
            triton_dir=triton_dir,
            verbose=args.verbose,
        )

    print_summary(results)

    # 保存结果到CSV（如果指定了--save-csv）
    if args.save_csv:
        csv_path = save_results_to_csv(results, output_dir=base_dir)
        print(f"\nResults saved to: {csv_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
