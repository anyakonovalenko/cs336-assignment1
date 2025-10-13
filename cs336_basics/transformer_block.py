import torch
import torch.nn as nn
from einops import rearrange, einsum
from cs336_basics.rmsnorm import RMSNorm
from cs336_basics.causal_multi_head import MultiHead
from cs336_basics.swiglu import SwiGLU

class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, use_rope, max_seq_len = None, theta = None, device = None, dtype = None):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.use_rope = use_rope
        self.max_seq_len = max_seq_len
        self.theta = theta
        self.device = device
        self.dtype = dtype
        self.rms1 = RMSNorm(d_model)
        self.rms2 = RMSNorm(d_model)
        self.multihead = MultiHead(d_model, num_heads, use_rope=True, theta=theta, max_seq_len = max_seq_len)
        self.swiglu = SwiGLU(d_model = d_model, d_ff = self.d_ff)

        # in_features: Float[Tensor, " batch sequence_length d_model"],
        # ) -> Float[Tensor, " batch sequence_length d_model"]:

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pre_norm_attention = self.rms1.forward(x)
        attention = self.multihead.forward(pre_norm_attention)
        res_1 = x + attention
        pre_norm_ffn = self.rms2.forward(res_1)
        ffn = self.swiglu.forward(pre_norm_ffn)
        result = res_1 + ffn
        return result




