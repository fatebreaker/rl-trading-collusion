"""N independent MLPs evaluated as one batched matmul.

Every (session, trader) pair needs its own network. Stacking their weights on
a leading dimension and using baddbmm keeps them fully independent (no shared
parameters, and Adam is elementwise so optimiser state is independent too)
while running as a single kernel on CPU.
"""

from __future__ import annotations

import math

import torch
from torch import nn


class BatchedMLP(nn.Module):
    def __init__(self, n_nets: int, d_in: int, d_out: int, hidden: int = 64, depth: int = 2):
        super().__init__()
        dims = [d_in] + [hidden] * depth + [d_out]
        self.weights = nn.ParameterList()
        self.biases = nn.ParameterList()
        for i, (a, b) in enumerate(zip(dims[:-1], dims[1:])):
            bound = 1.0 / math.sqrt(a)
            w = torch.empty(n_nets, a, b).uniform_(-bound, bound)
            if i == len(dims) - 2:
                w.mul_(0.1)  # small last layer: near-uniform initial outputs
            self.weights.append(nn.Parameter(w))
            self.biases.append(nn.Parameter(torch.zeros(n_nets, 1, b)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (N, B, d_in) -> (N, B, d_out)."""
        n = len(self.weights)
        for i, (w, b) in enumerate(zip(self.weights, self.biases)):
            x = torch.baddbmm(b, x, w)
            if i < n - 1:
                x = torch.relu(x)
        return x


@torch.no_grad()
def clip_grad_norm_per_net(module: BatchedMLP, max_norm: float) -> None:
    """Clip each network's gradient norm separately.

    torch.nn.utils.clip_grad_norm_ would compute one global norm across all
    networks, so a single exploding net would shrink every other net's update.
    """
    grads = [p.grad for p in module.parameters() if p.grad is not None]
    if not grads:
        return
    sq = sum(g.pow(2).flatten(1).sum(1) for g in grads)  # (N,)
    scale = (max_norm / (sq.sqrt() + 1e-6)).clamp(max=1.0)
    for g in grads:
        g.mul_(scale.view(-1, *([1] * (g.dim() - 1))))
