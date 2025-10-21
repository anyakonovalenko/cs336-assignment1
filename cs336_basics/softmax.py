import torch
import torch.nn as nn
from einops import rearrange, einsum

#across i-th position mean that a_i000 and others dimension are fixed
#for matrix (3,4) for first dimension across rows for second across columns (you fix the other dimensions)

def softmax(in_features: torch.Tensor, dim: int):

    max_values = torch.max(in_features, dim=dim, keepdim = True).values # or unsqueeze(dim) later
    norm_values = in_features - max_values
    exp_sums = torch.sum(torch.exp(norm_values), dim=dim, keepdim = True)
    result = torch.exp(norm_values)/ exp_sums
    return result

