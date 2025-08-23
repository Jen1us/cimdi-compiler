#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Qwen1.5-MoE-A2.7B模型ONNX导出工具
将Hugging Face的Qwen1.5-MoE-A2.7B模型转换为ONNX格式
"""

import torch
import torch.onnx
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
import argparse
import os
import sys

def export_qwen_to_onnx(
    model_name: str = "Qwen/Qwen1.5-MoE-A2.7B",
    output_path: str = "qwen_moe_2.7b.onnx",
    max_length: int = 512,
    batch_size: int = 1,
    device: str = "cpu",
    opset_version: int = 14,
    use_cache: bool = False,
    use_dynamo: bool = False
):
    """
    将Qwen1.5-MoE-A2.7B模型导出为ONNX格式
    
    Args:
        model_name: 模型名称或路径
        output_path: ONNX文件输出路径
        max_length: 最大序列长度
        batch_size: 批次大小
        device: 运行设备 (cpu/cuda)
        opset_version: ONNX opset版本
        use_cache: 是否使用KV缓存 (注意：使用缓存会让导出更复杂)
        use_dynamo: 是否使用dynamo导出 (MoE模型可能有兼容性问题)
    """
    
    print(f"正在加载模型和分词器: {model_name}")
    
    # 加载分词器和模型配置
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    
    # 加载模型
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        torch_dtype=torch.float32,  # ONNX导出建议使用float32
        device_map=device if device == "cpu" else "auto",
        use_cache=use_cache
    )
    
    # 设置为评估模式
    model.eval()
    
    # 如果使用GPU，将模型移到指定设备
    if device != "cpu" and torch.cuda.is_available():
        model = model.to(device)
    
    print(f"模型已加载到设备: {device}")
    
    # 创建示例输入
    # 对于语言模型，通常输入是input_ids和attention_mask
    sample_text = "你好，我是Qwen模型。"
    inputs = tokenizer(
        sample_text,
        return_tensors="pt",
        max_length=max_length,
        padding="max_length",
        truncation=True
    )
    
    # 调整批次大小
    if batch_size > 1:
        input_ids = inputs["input_ids"].repeat(batch_size, 1)
        attention_mask = inputs["attention_mask"].repeat(batch_size, 1)
    else:
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
    
    # 将输入移到相同设备
    if device != "cpu":
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
    
    # 定义输入名称
    input_names = ["input_ids", "attention_mask"]
    output_names = ["logits"]
    
    # 定义动态轴 (允许可变的批次大小和序列长度)
    dynamic_axes = None
    if not use_dynamo:
        dynamic_axes = {
            "input_ids": {0: "batch_size", 1: "sequence_length"},
            "attention_mask": {0: "batch_size", 1: "sequence_length"},
            "logits": {0: "batch_size", 1: "sequence_length"}
        }
    
    print("开始导出ONNX模型...")
    print(f"使用dynamo: {use_dynamo}")
    
    try:
        # 进行ONNX导出
        with torch.no_grad():
            export_kwargs = {
                "export_params": True,  # 存储训练好的参数权重
                "opset_version": opset_version,  # ONNX版本
                "do_constant_folding": True,  # 是否执行常量折叠优化
                "input_names": input_names,  # 输入名称
                "output_names": output_names,  # 输出名称
                "verbose": False  # 是否打印详细信息
            }
            
            if use_dynamo:
                # 使用dynamo时不设置dynamic_axes，而是使用dynamic_shapes
                export_kwargs["dynamo"] = True
            else:
                # 传统导出方式使用dynamic_axes
                export_kwargs["dynamic_axes"] = dynamic_axes
            
            torch.onnx.export(
                model,  # 要导出的模型
                (input_ids, attention_mask),  # 模型输入的示例
                output_path,  # 输出文件路径
                **export_kwargs
            )
        
        print(f"ONNX模型已成功导出到: {output_path}")
        
        # 检查文件大小
        file_size = os.path.getsize(output_path) / (1024 * 1024)  # MB
        print(f"导出的ONNX文件大小: {file_size:.2f} MB")
        
        return True
        
    except Exception as e:
        print(f"导出失败: {str(e)}")
        return False

def verify_onnx_model(onnx_path: str):
    """
    验证导出的ONNX模型
    """
    try:
        import onnx
        import onnxruntime as ort
        
        print("验证ONNX模型...")
        
        # 加载并检查ONNX模型
        onnx_model = onnx.load(onnx_path)
        onnx.checker.check_model(onnx_model)
        print("ONNX模型结构验证通过")
        
        # 创建ONNX Runtime会话进行推理测试
        ort_session = ort.InferenceSession(onnx_path)
        
        # 获取输入输出信息
        input_names = [input.name for input in ort_session.get_inputs()]
        output_names = [output.name for output in ort_session.get_outputs()]
        
        print(f"模型输入: {input_names}")
        print(f"模型输出: {output_names}")
        
        # 创建测试输入
        test_input_ids = torch.randint(0, 1000, (1, 10)).numpy()
        test_attention_mask = torch.ones(1, 10).numpy()
        
        # 运行推理
        outputs = ort_session.run(
            output_names,
            {
                "input_ids": test_input_ids,
                "attention_mask": test_attention_mask
            }
        )
        
        print(f"ONNX模型推理测试成功，输出形状: {outputs[0].shape}")
        return True
        
    except ImportError:
        print("警告: 未安装onnx和onnxruntime，跳过模型验证")
        print("可以运行: pip install onnx onnxruntime")
        return False
    except Exception as e:
        print(f"ONNX模型验证失败: {str(e)}")
        return False

def main():
    parser = argparse.ArgumentParser(description="将Qwen1.5-MoE-A2.7B模型导出为ONNX格式")
    parser.add_argument("--model_name", default="Qwen/Qwen1.5-MoE-A2.7B", help="模型名称或路径")
    parser.add_argument("--output", default="qwen_moe_2.7b.onnx", help="输出ONNX文件路径")
    parser.add_argument("--max_length", type=int, default=512, help="最大序列长度")
    parser.add_argument("--batch_size", type=int, default=1, help="批次大小")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"], help="运行设备")
    parser.add_argument("--opset_version", type=int, default=14, help="ONNX opset版本")
    parser.add_argument("--use_dynamo", action="store_true", help="是否使用dynamo导出(对MoE模型可能有问题)")
    parser.add_argument("--verify", action="store_true", help="是否验证导出的ONNX模型")
    
    args = parser.parse_args()
    
    print("=" * 50)
    print("Qwen1.5-MoE-A2.7B ONNX导出工具")
    print("=" * 50)
    
    # 检查设备可用性
    if args.device == "cuda" and not torch.cuda.is_available():
        print("警告: CUDA不可用，切换到CPU")
        args.device = "cpu"
    
    # 导出模型
    success = export_qwen_to_onnx(
        model_name=args.model_name,
        output_path=args.output,
        max_length=args.max_length,
        batch_size=args.batch_size,
        device=args.device,
        opset_version=args.opset_version,
        use_dynamo=args.use_dynamo
    )
    
    if success and args.verify:
        verify_onnx_model(args.output)
    
    print("=" * 50)
    if success:
        print("导出完成！")
    else:
        print("导出失败！")
        sys.exit(1)

if __name__ == "__main__":
    main()