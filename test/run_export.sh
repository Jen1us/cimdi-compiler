#!/bin/bash

# 安装依赖包
pip install -r requirements.txt

# 基础使用 - 导出到CPU
python qwen_to_onnx.py

# 自定义参数使用
python qwen_to_onnx.py \
    --model_name "Qwen/Qwen1.5-MoE-A2.7B" \
    --output "my_qwen_model.onnx" \
    --max_length 1024 \
    --device cpu \
    --verify

# 如果有GPU，可以使用CUDA加速（需要安装CUDA版本的PyTorch）
# python qwen_to_onnx.py --device cuda --verify