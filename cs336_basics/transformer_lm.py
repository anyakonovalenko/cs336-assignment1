import torch
import torch.nn as nn
from einops import rearrange, einsum
from cs336_basics.rmsnorm import RMSNorm
from cs336_basics.causal_multi_head import MultiHead
from cs336_basics.swiglu import SwiGLU
from cs336_basics.transformer_block import TransformerBlock
from cs336_basics.embedding import Embedding
from cs336_basics.rmsnorm import RMSNorm
from cs336_basics.linear import Linear

class TransformerLM(nn.Module):
    def __init__(self, vocab_size, context_length, d_model, num_layers, num_heads, d_ff, theta = None, device = None, dtype = None):
        super().__init__()

        self.vocab_size = vocab_size
        self.num_layers = num_layers
        self.layers = nn.ModuleList([
            TransformerBlock(d_model=d_model, num_heads=num_heads, d_ff=d_ff,
                             use_rope=True, max_seq_len=context_length, theta=theta)
            for _ in range(num_layers)])
        self.embedding = Embedding(vocab_size, d_model)
        self.final_norm = RMSNorm(d_model)
        self.lm_head = Linear(d_model, self.vocab_size)

    def forward(self, tokens_ids: torch.Tensor) -> torch.Tensor:

        embedded_input = self.embedding.forward(tokens_ids)
        result = embedded_input
        for layer in self.layers:
            result = layer.forward(result)
        result = self.final_norm.forward(result)
        logits = self.lm_head.forward(result)
        return logits



