import torch
import torch.nn as nn
from einops import rearrange, einsum
from cs336_basics.linear import Linear
from cs336_basics.scaled_dot_product import scaled_dot_product_attention
from cs336_basics.rope import RoPE


class MultiHead(nn.Module):
    def __init__(self, d_model, n_heads, use_rope=False, theta=None, max_seq_len = None, device = None, dtype = None):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.use_rope = use_rope
        self.device = device
        self.dtype = dtype
        self.d_k = d_model // n_heads
        self.d_v = d_model // n_heads
        self.W_Q = Linear(d_model, n_heads * self.d_k)
        self.W_K = Linear(d_model, n_heads * self.d_k)
        self.W_V = Linear(d_model, n_heads * self.d_v)
        self.W_O = Linear(n_heads * self.d_v, d_model)
        self.theta = theta
        self.max_seq_len = max_seq_len

        # in_features (Float[Tensor, "... sequence_length d_in"])

    def forward(self, x: torch.Tensor, token_positions = None) -> torch.Tensor:
        sequence_length = x.shape[-2]
        self.token_positions = token_positions
        if self.use_rope:
            self.rope = RoPE(self.theta, self.d_k, self.max_seq_len)

        Q = self.W_Q.forward(x)  # ( ... n_heads * self.d_k, sequence_length)
        K = self.W_K.forward(x)  # ( ... n_heads * self.d_k, sequence_length)
        V = self.W_V.forward(x)  # ( ... n_heads * self.d_v, sequence_length)
        Q = rearrange(Q, 'batch seq (heads d_k) -> batch heads seq d_k',
                      heads=self.n_heads, d_k=self.d_k)
        K = rearrange(K, 'batch seq (heads d_k) -> batch heads seq d_k',
                      heads=self.n_heads, d_k=self.d_k)
        V = rearrange(V, 'batch seq (heads d_v) -> batch heads seq d_v',
                      heads=self.n_heads, d_v=self.d_v)
        causal_mask = torch.tril(torch.ones((sequence_length, sequence_length), dtype=torch.bool))

        if self.use_rope:
            Q = self.rope(Q, token_positions)
            K = self.rope(K, token_positions)

        attn_output = scaled_dot_product_attention(Q, K, V, mask= causal_mask)

        attn_output = rearrange(attn_output, '... heads seq d_v -> ... seq (heads d_v)')

        result = self.W_O.forward(attn_output)

        return result






