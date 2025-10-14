import torch
import torch.nn as nn
from collections import defaultdict

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

# self.state{"parameter": {'t':value}}
class AdamW(torch.optim.Optimizer):
    def __init__(self, params, lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.01):
        defaults = {"lr": lr, "betas": betas, "eps": eps, "weight_decay": weight_decay}
        super().__init__(params, defaults) # {weights:{"lr":lr}} in self.param_groups

    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]  # Get the learning rate.
            betas = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]
            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad.data

                if len(self.state[p]) == 0:
                    self.state[p]['step'] = 0
                    self.state[p]['m'] = torch.zeros_like(p.data)
                    self.state[p]['v'] = torch.zeros_like(p.data)
                self.state[p]['step'] += 1

                self.state[p]['m'] = betas[0] * self.state[p]['m'] + (1-betas[0]) * grad
                self.state[p]['v'] = betas[1] * self.state[p]['v'] + (1 - betas[1]) * grad**2
                alpha_t = lr*(math.sqrt(1-betas[1]**self.state[p]['step'])/(1-betas[0]**self.state[p]['step']))
                p.data -= (alpha_t*self.state[p]['m']) / (torch.sqrt(self.state[p]['v']) + eps)
                p.data -= weight_decay*p.data*lr

        return loss


# weights = nn.Parameter(5*torch.randn((10,10)))
# opt = SGD([weights], lr=1e3)

#1e1 decrease
#1e2 decrease faster
#1e3 diverge (go up to inf)

# Every tensor of torch.Tensor has the backward() method defined !!!!
# it stores computation graph (to take derivates) in grad_fn => not each tensor will have it

# for t in range(100):
#     opt.zero_grad()
#     loss = (weights**2).mean() # Builds computation graph
#     print(loss.cpu().item())
#     loss.backward()  # Fills weights.grad with values
#     opt.step()      # weights -= lr * weights.grad