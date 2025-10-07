import torch
import torch.nn as nn
from cs336_basics.linear import Linear
from einops import rearrange, einsum, repeat

class RoPE(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len:int, device = None):
        super().__init__()
        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len
        self.device = device
        positions = torch.arange(max_seq_len, device=device)  # [max_seq_len]
        dim_pairs = torch.arange(d_k // 2, device=device)  # [d_k//2]
        positions = rearrange(positions, 'i -> i 1')  # [max_seq_len, 1]
        dim_pairs = rearrange(dim_pairs, 'k -> 1 k')  # [1, d_k//2]

        # Now compute angles - broadcasting will create [max_seq_len, d_k//2]
        angles = positions / (theta ** (2 * dim_pairs / d_k))
        self.register_buffer('sin_angles', torch.sin(angles), persistent=False)
        self.register_buffer('cos_angles', torch.cos(angles), persistent=False)


    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        pos_sin = self.sin_angles[token_positions] # [..., seq_len, d_k//2] (for each position it will select sin)
        sin_expanded = repeat(pos_sin, '... seq pairs -> ... seq (pairs two)', two=2)
        pos_cos = self.cos_angles[token_positions]  # [..., seq_len, d_k//2] (for each position it will select sin)
        cos_expanded = repeat(pos_cos, '... seq pairs -> ... seq (pairs two)', two=2)

        x_pairs = rearrange(x, '... seq (pairs two) -> ... seq pairs two', two=2)  # [..., seq_len, d_k//2, 2]

        # Create the rotated version: swap and negate
        # For each pair (x1, x2) -> (-x2, x1)
        x_rotated = torch.stack([-x_pairs[..., 1], x_pairs[..., 0]], dim=-1)  # [..., seq_len, d_k//2, 2]

        # Flatten back to original shape
        x_rotated = rearrange(x_rotated, '... seq pairs two -> ... seq (pairs two)')  # [..., seq_len, d_k]

        # Apply rotation formula: x * cos + x_rotated * sin
        return x * cos_expanded + x_rotated * sin_expanded



