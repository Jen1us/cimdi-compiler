# convert_only.py
import torch
import torch_mlir.fx
from torch_mlir.fx import export_and_import
import gc
import pickle # 导入 pickle 库

# --- 核心修改在这里：使用 pickle 加载 ---
input_filename = "llama-moe.pkl"
print(f"正在使用 pickle 从文件加载 ExportedProgram: {input_filename}")
with open(input_filename, "rb") as f: # 注意是 "rb" (read binary)
    exported_program = pickle.load(f)
print("ExportedProgram 加载完成。")

print("正在将导出的图编译为 Torch Dialect MLIR...")
mlir_module = export_and_import(exported_program)
print("MLIR 转换完成。")

del exported_program
gc.collect()
print("ExportedProgram 已从内存中删除。")

print("正在将 MLIR 序列化为字符串...")
module_str = str(mlir_module)
print("MLIR 编译为字符串完成。")

output_filename = "OLMoE-1B.torch.mlir"
print(f"正在将 MLIR 代码保存到: {output_filename}")
with open(output_filename, "w", encoding="utf-8") as f:
    f.write(module_str)
print("保存完成。")