import torch
import torch.nn as nn
from einops import rearrange, einsum
from cs336_basics.softmax import softmax
import math
from collections.abc import Iterable
import numpy.typing as npt
import numpy as np

import os
from collections.abc import Iterable
from typing import IO, Any, BinaryIO

def save_checkpoint(model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: str | os.PathLike | BinaryIO | IO[bytes]):
    torch.save({'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'iteration': iteration}, out)
    return




def load_checkpoint(source: str | os.PathLike | BinaryIO | IO[bytes],
                    model: torch.nn.Module,
                    optimizer: torch.optim.Optimizer):
    source_dict = torch.load(source)
    model.load_state_dict(source_dict['model'])
    optimizer.load_state_dict(source_dict['optimizer'])

    return source_dict['iteration']
