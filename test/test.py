import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch_mlir.fx import export_and_import

# --- 1. 加载模型和分词器 ---
print("正在加载 Qwen1.5-MoE-A2.7B 模型...")
model_id = "allenai/OLMoE-1B-7B-0924"

# 为了在内存有限的机器上运行，我们使用 float16，并让模型停留在 CPU 上
# 如果您有足够的 VRAM，可以移除 device_map="cpu"
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    device_map="cpu" 
)
tokenizer = AutoTokenizer.from_pretrained(model_id)

# 确保模型处于评估模式，这会禁用 dropout 等训练特有的层
model.eval()
print("模型加载完成。")

# --- 2. 准备虚拟输入 (Dummy Inputs) ---
# 为了导出一个通用的图，我们不使用真实文本，而是创建符合格式的虚拟张量
# 这样可以避免图被固定在某个特定的输入长度上
batch_size = 1
# 使用一个较短的序列长度来进行导出，便于处理
# 在导出后，这个维度通常可以保持动态
seq_len = 16 

# input_ids: 模型的 token 输入
dummy_input_ids = torch.randint(
    0, model.config.vocab_size, 
    (batch_size, seq_len), 
    dtype=torch.long
)
# attention_mask: 告诉模型哪些 token 是有效的，哪些是 padding
dummy_attention_mask = torch.ones(
    batch_size, seq_len, 
    dtype=torch.long
)

print(f"创建虚拟输入: input_ids shape={dummy_input_ids.shape}, attention_mask shape={dummy_attention_mask.shape}")

# --- 3. 使用 torch.export 导出模型 ---
# 这是关键步骤。`torch.export` 会捕捉到模型的动态性，
# 包括 MoE 门控网络 (gating network) 中的数据依赖判断。
print("正在使用 torch.export 导出模型...")
try:
    # `torch.export` 需要一个 `kwargs` 字典来传递命名参数
    exported_program = torch.export.export(
        model, 
        args=(), # 没有位置参数
        kwargs={
            'input_ids': dummy_input_ids,
            'attention_mask': dummy_attention_mask
        }
    )
    print("torch.export 成功！")
except Exception as e:
    print(f"torch.export 失败: {e}")
    # MoE 模型通常需要较新版本的 PyTorch (2.2+) 才能很好地支持
    print("请确保您的 PyTorch 版本足够新以支持 MoE 模型的导出。")
    exit()
    
# --- 4. 将 ExportedProgram 转换为 Torch Dialect MLIR ---
print("正在将导出的图编译为 Torch Dialect MLIR...")
mlir_module = export_and_import(
    exported_program,
    # 可以添加其他参数，如 func_name 等
)
module_str = str(mlir_module)
print("MLIR 编译为字符串完成。")

# --- 5. 保存 MLIR 到文件 ---
output_filename = "MoE-Llama-1B-4E.torch.mlir"
print(f"正在将 MLIR 代码保存到: {output_filename}")
with open(output_filename, "w", encoding="utf-8") as f:
    f.write(module_str) #直接写入编译好的字符串

print("="*50)
print("成功！")
print(f"您现在可以在 {output_filename} 中查看生成的 Torch Dialect IR。")
print("下一步，您可以使用 `torch-mlir-opt` 将其降低到 Linalg Dialect。")
print("="*50)

print("="*50)
print("成功！")
print(f"您现在可以在 {output_filename} 中查看生成的 Torch Dialect IR。")
print("下一步，您可以使用 `torch-mlir-opt` 将其降低到 Linalg Dialect。")
print("="*50)