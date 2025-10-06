import torch
import torch.nn as nn
from einops import rearrange, einsum

class Linear(nn.Module):
    def __init__(self, in_features, out_features, device = None, dtype = None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.device = device
        self.dtype = dtype
        sigma = 2/(in_features + out_features)
        self.W = nn.Parameter(nn.init.trunc_normal_(torch.empty(out_features, in_features), 0, sigma, -3*sigma, 3*sigma))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        result = einsum(self.W, x, "d_out d_in, ... d_in -> ... d_out") #x@self.W.tranpose()
        return result


