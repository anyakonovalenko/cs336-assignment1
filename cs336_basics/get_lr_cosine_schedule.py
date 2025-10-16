import torch
import torch.nn as nn
from einops import rearrange, einsum
import math


def get_lr_cosine_schedule(it: int,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int, #Tw
    cosine_cycle_iters: int): #Tc

    if it < warmup_iters:
        alpha_t = (it/warmup_iters)*max_learning_rate
    elif it <= cosine_cycle_iters:
        progress = (it - warmup_iters) / (cosine_cycle_iters - warmup_iters)
        alpha_t = min_learning_rate + 0.5*(1+math.cos(progress*math.pi))*(max_learning_rate-min_learning_rate)
    else:
        alpha_t = min_learning_rate
    return alpha_t

