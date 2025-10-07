import torch
import torch.nn as nn
from einops import rearrange, einsum

class RMSNorm(nn.Module):
    def __init__(self, d_model, eps = 1*10^-5, device = None, dtype = None):
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.device = device
        self.dtype = dtype
        self.weights = nn.Parameter(torch.ones(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.sqrt(torch.mean(x**2, dim=-1,keepdim=True) + self.eps)
        return (x/rms)*self.weights


