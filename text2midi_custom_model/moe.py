
import torch
import torch.nn as nn
import torch.nn.functional as F

class MoEFeedForward(nn.Module):
    def __init__(self, input_dim, hidden_dim=2048, num_experts=4, dropout=0.1):
        super().__init__()
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, input_dim),
                nn.Dropout(dropout)
            ) for _ in range(num_experts)
        ])
        self.gate = nn.Linear(input_dim, num_experts)

    def forward(self, x):
        gate_logits = self.gate(x)
        gate_weights = F.softmax(gate_logits, dim=-1)
        expert_outputs = torch.stack([expert(x) for expert in self.experts], dim=-1)
        return torch.sum(expert_outputs * gate_weights.unsqueeze(2), dim=-1)
