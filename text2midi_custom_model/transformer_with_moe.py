
import torch
import torch.nn as nn
from transformer_model import Transformer
from moe_feedforward import MoEFeedForward
class TransformerWithMoE(nn.Module):
    def __init__(self, vocab_size, device='cpu'):
        super().__init__()
        self.core = Transformer(vocab_size, 768, 8, 2048, 18, 1024, False, 8, device=device)
        self.moe = MoEFeedForward(input_dim=768, hidden_dim=2048, num_experts=4)
        self.output_proj = nn.Linear(768, vocab_size)

    def forward(self, input_ids, decoder_input_ids=None, attention_mask=None):
        core_out = self.core(input_ids, decoder_input_ids, attention_mask)
        moe_out = self.moe(core_out)
        logits = self.output_proj(moe_out)
        return logits



    def generate(self, input_ids, attention_mask=None, max_len=2000, temperature=1.0):
        return self.core.generate(input_ids, attention_mask, max_len=max_len, temperature=temperature)

