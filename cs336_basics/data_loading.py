import torch
import torch.nn as nn
from einops import rearrange, einsum
from cs336_basics.softmax import softmax
import math
from collections.abc import Iterable
import numpy.typing as npt
import numpy as np

def data_loading(dataset: npt.NDArray, batch_size: int, context_length: int, device: str):
    inputs = torch.empty((batch_size, context_length), dtype=torch.long)
    targets = torch.empty((batch_size, context_length), dtype=torch.long)
    for bs in range(batch_size):
        starting_point = np.random.randint(0, len(dataset) - context_length)
        inputs[bs] = torch.from_numpy(dataset[starting_point:starting_point+context_length])
        targets[bs] = torch.from_numpy(dataset[starting_point+1:starting_point + context_length+1])

    return inputs.to(device), targets.to(device)

