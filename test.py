# https://pytorch.org/functorch/stable/notebooks/jacobians_hessians.html
# Requires PyTorch >= 1.11

from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.benchmark import Timer
import torch.autograd.forward_ad as fwAD
from functorch import jvp


B = 128
N = 3096
M = 10
model = nn.Sequential(
    nn.Linear(N, 1024),
    nn.ReLU(),
    nn.Linear(1024, 1024),
    nn.ReLU(),
    nn.Linear(1024, M),
)
x = torch.randn(B, N)
v = torch.randn_like(x)
g = torch.randn(B, M)


model.zero_grad()
# Method 1
# Use PyTorch autograd forward mode
with fwAD.dual_level():
    fx, Jv = fwAD.unpack_dual(model(fwAD.make_dual(x, v)))
(fx + Jv).backward(g)
grad = torch.cat([param.grad.clone().flatten() for param in model.parameters()])


model.zero_grad()
# Method 2
# Using functorch vjp
fx2, Jv2 = jvp(func=model, primals=(x, ), tangents=(v, ))
(fx2 + Jv2).backward(g)
grad2 = torch.cat([param.grad.clone().flatten() for param in model.parameters()])

print(torch.dist(fx, fx2), torch.dist(Jv, Jv2), torch.dist(grad, grad2))

# print(primal_output[:2,:2].reshape(-1), primal_output2[:2,:2].reshape(-1))
# print(tangent_output[:2,:2].reshape(-1), tangent_output2[:2,:2].reshape(-1))
