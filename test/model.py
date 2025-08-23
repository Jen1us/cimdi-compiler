import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_mlir.fx import export_and_import

# 1. 模型定义 (Expert 和 SimpleModelWithMoE 无需修改)
class Expert(nn.Module):
    def __init__(self, d_model, d_hidden):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_hidden),
            nn.ReLU(),
            nn.Linear(d_hidden, d_model)
        )
    def forward(self, x):
        return self.net(x)

class MoELayer(nn.Module):
    def __init__(self, d_model, num_experts, d_hidden):
        super().__init__()
        self.d_model = d_model
        self.num_experts = num_experts
        self.gate = nn.Linear(d_model, num_experts)
        self.experts = nn.ModuleList([Expert(d_model, d_hidden) for _ in range(num_experts)])

    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        
        # 显式计算形状，避免使用 -1
        x_reshaped = x.view(batch_size * seq_len, self.d_model)
        
        gate_logits = self.gate(x_reshaped)
        expert_weights, expert_indices = torch.max(F.softmax(gate_logits, dim=-1), dim=-1)
        expert_weights = expert_weights.unsqueeze(-1)
        expert_outputs = torch.stack([expert(x_reshaped) for expert in self.experts], dim=1)
        idx = expert_indices.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, self.d_model)
        selected_expert_output = torch.gather(expert_outputs, 1, idx).squeeze(1)
        final_output = selected_expert_output * expert_weights
        return final_output.view(batch_size, seq_len, self.d_model)

class SimpleModelWithMoE(nn.Module):
    def __init__(self, vocab_size, d_model, nhead, num_experts, d_hidden):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.attention = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        self.moe = MoELayer(d_model, num_experts, d_hidden)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.embedding.weight

    def forward(self, input_ids):
        x = self.embedding(input_ids)
        attn_output, _ = self.attention(x, x, x)
        moe_output = self.moe(attn_output)
        logits = self.lm_head(moe_output)
        return logits

# --- 导出流程 ---
vocab_size = 1000
d_model = 128
nhead = 4
num_experts = 8
d_hidden = 256
model = SimpleModelWithMoE(vocab_size, d_model, nhead, num_experts, d_hidden)
model.eval()
example_input = torch.randint(0, vocab_size, (1, 10))

print("正在将模型导出并导入为 MLIR 'torch' 方言...")
torch_dialect_module = export_and_import(model, example_input, output_type="LINALG_ON_TENSORS")
print("'torch' 方言模块已生成！")

output_torch_filename = "model.torch.mlir"
with open(output_torch_filename, "w") as f:
    f.write(str(torch_dialect_module))

print(f"\n模型的 'torch' 方言表示已保存到 {output_torch_filename}")
print("下一步，请在您的终端中运行 torch-mlir-opt 命令。")