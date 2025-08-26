import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_mlir.fx import export_and_import

# INT4 量化线性层
class QuantizedLinearINT4(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # 量化权重 (INT4: -8 到 7)
        self.register_buffer('quantized_weight', torch.randint(-8, 8, (out_features, in_features), dtype=torch.int8))
        self.register_buffer('weight_scale', torch.randn(out_features, 1))
        self.register_buffer('weight_zero_point', torch.randint(-8, 8, (out_features, 1), dtype=torch.int8))
        
        if bias:
            self.bias = nn.Parameter(torch.randn(out_features))
        else:
            self.register_parameter('bias', None)
    
    def forward(self, x):
        # 反量化权重
        dequantized_weight = (self.quantized_weight.float() - self.weight_zero_point.float()) * self.weight_scale
        return F.linear(x, dequantized_weight, self.bias)

# 1. 模型定义 - INT4量化版本
class QuantizedExpert(nn.Module):
    def __init__(self, d_model, d_hidden):
        super().__init__()
        self.linear1 = QuantizedLinearINT4(d_model, d_hidden)
        self.relu = nn.ReLU()
        self.linear2 = QuantizedLinearINT4(d_hidden, d_model)
    
    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        return x

# 原始Expert类保持不变，用于对比
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

# 量化MoE层
class QuantizedMoELayer(nn.Module):
    def __init__(self, d_model, num_experts, d_hidden):
        super().__init__()
        self.d_model = d_model
        self.num_experts = num_experts
        self.gate = QuantizedLinearINT4(d_model, num_experts)
        self.experts = nn.ModuleList([QuantizedExpert(d_model, d_hidden) for _ in range(num_experts)])

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

# 原始MoE层保持不变，用于对比
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

# 量化版本的SimpleModelWithMoE
class QuantizedSimpleModelWithMoE(nn.Module):
    def __init__(self, vocab_size, d_model, nhead, num_experts, d_hidden):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.attention = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        self.moe = QuantizedMoELayer(d_model, num_experts, d_hidden)
        self.lm_head = QuantizedLinearINT4(d_model, vocab_size, bias=False)

    def forward(self, input_ids):
        x = self.embedding(input_ids)
        attn_output, _ = self.attention(x, x, x)
        moe_output = self.moe(attn_output)
        logits = self.lm_head(moe_output)
        return logits

# 原始版本保持不变，用于对比
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

# 原始模型
print("=== 原始模型 ===")
original_model = SimpleModelWithMoE(vocab_size, d_model, nhead, num_experts, d_hidden)
original_model.eval()

# 量化模型
print("=== INT4量化模型 ===")
quantized_model = QuantizedSimpleModelWithMoE(vocab_size, d_model, nhead, num_experts, d_hidden)
quantized_model.eval()

example_input = torch.randint(0, vocab_size, (1, 10))

print("正在将量化模型导出并导入为 MLIR 'linalg' 方言...")
torch_dialect_module = export_and_import(quantized_model, example_input, output_type="LINALG_ON_TENSORS")
print("量化模型的 'linalg' 方言模块已生成！")

output_torch_filename = "quantized_model.torch.mlir"
with open(output_torch_filename, "w") as f:
    f.write(str(torch_dialect_module))

print(f"\n量化模型的 'torch' 方言表示已保存到 {output_torch_filename}")
print("量化模型包含INT4量化的MoE层！")
print("下一步，请在您的终端中运行 torch-mlir-opt 命令。")