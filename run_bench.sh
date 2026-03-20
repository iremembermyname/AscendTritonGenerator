#!/bin/bash

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
PYTORCH_DIR="$PROJECT_DIR/pytorch_kernels"
TRITON_DIR="$PROJECT_DIR/triton_kernels"

echo "Scanning for operators to convert..."

# 获取所有需要转换的算子
for level_dir in "$PYTORCH_DIR"/level*/; do
    level=$(basename "$level_dir")
    
    for pytorch_file in "$level_dir"/*.py; do
        operator=$(basename "$pytorch_file" .py)
        triton_file="$TRITON_DIR/$level/$operator.py"
        
        echo "Converting: $level/$operator"
        mkdir -p "$TRITON_DIR/$level"
        
        # 调用Claude Code进行转换
        PROMPT="你是一名专业的AI算子优化工程师，擅长将PyTorch算子转换为高性能的Triton算子。

任务：将 pytorch_kernels/$level/$operator.py 转换为 triton_kernels/$level/$operator.py

要求：
1. 仔细阅读PyTorch算子的实现逻辑
2. 使用Triton编写等效实现，保持接口一致（ModelNew类，forward方法）
3. 确保数值精度正确，与PyTorch实现误差在可接受范围内
4. 针对昇腾NPU优化性能（20个核心，合理使用grid-stride loop）
5. 生成的代码要高效，性能尽可能超越PyTorch基线

请直接生成完整的Triton实现代码并保存到目标路径。"
        
        claude -p "$PROMPT" --dangerously-skip-permissions
    done
done

# 运行评测
echo "Running evaluation..."
cd "$PROJECT_DIR"
python scripts/eval.py --save-csv

echo "Done!"
