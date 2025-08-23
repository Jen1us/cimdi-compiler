# export_only.py
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import gc
import pickle # 导入 pickle 库

print("正在加载模型...")
model_id = "llama-moe/LLaMA-MoE-v1-3_0B-2_16"
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    device_map="cpu" 
)
model.eval()
print("模型加载完成。")

batch_size = 1
seq_len = 16 
dummy_input_ids = torch.randint(0, model.config.vocab_size, (batch_size, seq_len), dtype=torch.long)
dummy_attention_mask = torch.ones(batch_size, seq_len, dtype=torch.long)

with torch.no_grad():
    print("正在使用 torch.export 导出模型...")
    exported_program = torch.export.export(model, args=(), kwargs={'input_ids': dummy_input_ids, 'attention_mask': dummy_attention_mask})
    print("torch.export 成功！")

    del model
    gc.collect()
    print("原始模型已从内存中删除。")

    # --- 核心修改在这里：使用 pickle 保存 ---
    output_filename = "llama-moe.pkl"
    print(f"正在使用 pickle 保存 ExportedProgram 到文件: {output_filename}")
    with open(output_filename, "wb") as f: # 注意是 "wb" (write binary)
        pickle.dump(exported_program, f)
    print(f"ExportedProgram 已保存到 {output_filename}")