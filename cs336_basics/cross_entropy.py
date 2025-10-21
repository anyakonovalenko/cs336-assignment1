import torch
import torch.nn as nn
from einops import rearrange, einsum
from cs336_basics.softmax import softmax
import math



def cross_entropy(o: torch.Tensor, targets: torch.Tensor):
    # inputs: Float[Tensor, " batch_size vocab_size"], targets: Int[Tensor, " batch_size"]
    # log(∑ exp(oᵢ[a])-max) - oᵢ[xᵢ₊₁] + max
    max_values = torch.max(o, dim=-1, keepdim=True).values
    logits = o[torch.arange(targets.shape[0]), targets]
    result = torch.log(torch.sum(torch.exp(o - max_values), dim=-1)) - logits + max_values.squeeze(-1)
    return torch.mean(result)

