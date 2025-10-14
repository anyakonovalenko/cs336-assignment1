import torch
import torch.nn as nn

# torch.randn()

from collections.abc import Callable, Iterable
from typing import Optional
import torch
import math

#for W parameter this is where gradients are stored: W.grad (shows how much change the value)
#in regular implementation it was dw


# optim.SGD([
#     {'params': model.base.parameters(), 'lr': 1e-2},
#     {'params': model.classifier.parameters()}
# ], lr=1e-3, momentum=0.9)

# [
#   {
#     'params': [
#       Parameter (shape: torch.Size([10, 784])),
#       Parameter (shape: torch.Size([10]))
#     ],
#     'lr': 0.001
#   }
# ]


class SGD(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = {"lr": lr}
        super().__init__(params, defaults) # {weights:{"lr":lr}} in self.param_groups

    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]  # Get the learning rate.
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]  # Get state associated with p.
                t = state.get("t", 0)  # Get iteration number from the state, or initial value.
                grad = p.grad.data  # Get the gradient of loss with respect to p.
                p.data -= lr / math.sqrt(t + 1) * grad  # Update weight tensor in-place.
                state["t"] = t + 1  # Increment iteration number.
        return loss


weights = nn.Parameter(5*torch.randn((10,10)))
opt = SGD([weights], lr=1e3)

#1e1 decrease
#1e2 decrease faster
#1e3 diverge (go up to inf)

# Every tensor of torch.Tensor has the backward() method defined !!!!
# it stores computation graph (to take derivates) in grad_fn => not each tensor will have it

for t in range(100):
    opt.zero_grad()
    loss = (weights**2).mean() # Builds computation graph
    print(loss.cpu().item())
    loss.backward()  # Fills weights.grad with values
    opt.step()      # weights -= lr * weights.grad