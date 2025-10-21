import torch
import torch.nn as nn
from einops import rearrange, einsum
from cs336_basics.softmax import softmax
import math
from collections.abc import Iterable




def gradient_clipping(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float, epsilon=1e-6):

    l2_norm = 0
    for param in parameters:
        if param.grad is not None:
            l2_norm += torch.sum(param.grad.data**2)
    l2_norm = math.sqrt(l2_norm)

    for param in parameters:
        if l2_norm >  max_l2_norm:
            if param.grad is not None:
                scale_factor = max_l2_norm / (l2_norm + epsilon)
                param.grad.data *= scale_factor
    return

