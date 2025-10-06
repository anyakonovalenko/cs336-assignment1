import torch
import torch.nn as nn
from einops import rearrange, einsum

class Embedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, device = None, dtype = None):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.device = device
        self.dtype = dtype
        self.weights = nn.Parameter(nn.init.trunc_normal_(torch.empty(num_embeddings, embedding_dim), 0, 1, -3, 3))


    def forward(self, tokens_ids: torch.Tensor) -> torch.Tensor:
        return self.weights[tokens_ids] #advanced indexing


