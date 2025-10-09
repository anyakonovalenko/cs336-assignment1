import torch
import torch.nn as nn
from einops import rearrange, einsum
from cs336_basics.softmax import softmax
import math

#across i-th position mean that a_i000 and others dimension are fixed
#for matrix (3,4) for first dimension across rows for second across columns (you fix the other dimensions)

def scaled_dot_product_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor,  mask: torch.Tensor):

        # Q (Float[Tensor, " ... queries d_k"]): Query tensor
        # K (Float[Tensor, " ... keys d_k"]): Key tensor
        # V (Float[Tensor, " ... values d_v"]): Values tensor
        # mask (Bool[Tensor, " ... queries keys"] | None): Mask tensor
    # Returns:
    #     Float[Tensor, " ... queries d_v"]: Output of SDPA
    #     q_transpose = rearrange(Q, " ... queries d_k -> ... d_k queries")
        nominator = einsum(Q, K, "... queries d_k, ... keys d_k -> ... queries keys")
        d_k = Q.shape[-1]
        if mask is not None:
            # Where mask is True, keep nominamor; where False, use -inf
            nominator = torch.where(mask, nominator, torch.tensor(float('-inf')))
        A = softmax(nominator/math.sqrt(d_k), dim=-1)
        result = einsum(A, V, "... queries values, ... values d_v -> ... queries d_v")
        return result

