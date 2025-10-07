import torch
import torch.nn as nn
from cs336_basics.linear import Linear
from einops import rearrange, einsum

class SwiGLU(nn.Module):
    def __init__(self, d_model, device = None, dtype = None):
        super().__init__()
        self.d_model = d_model
        self.device = device
        self.dtype = dtype
        self.d_ff  = round((8/3 * d_model) / 64) * 64
        self.l1 = Linear(self.d_model, self.d_ff)
        self.l2 = Linear(self.d_ff, self.d_model)
        self.l3 = Linear(self.d_model, self.d_ff)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input1 = self.l1.forward(x)
        input3 = self.l3.forward(x)
        silu = input1*torch.sigmoid(input1)
        glu = silu*input3
        return self.l2.forward(glu)


