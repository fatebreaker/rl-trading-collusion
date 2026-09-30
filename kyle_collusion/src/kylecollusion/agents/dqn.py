"""Batched Double DQN with per-network replay buffers.

Observations are the continuous features from the market (current value,
previous value, previous residual flow), so the network sees the same
information as the tabular agent without binning.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from ..market import KyleMarket
from .batched_mlp import BatchedMLP


class DQN:
    name = "dqn"

    def __init__(
        self,
        env: KyleMarket,
        gamma: float = 0.95,
        lr: float = 1e-3,
        hidden: int = 64,
        buffer_size: int = 50_000,
        batch_size: int = 64,
        train_every: int = 4,
        target_every: int = 1_000,
        beta_decay: float = 2e-5,
        eps_min: float = 0.0,
        learning_starts: int = 1_000,
        seed: int = 0,
    ):
        torch.manual_seed(seed)
        self.env = env
        self.gamma = gamma
        self.batch_size = batch_size
        self.train_every = train_every
        self.target_every = target_every
        self.beta_decay = beta_decay
        self.eps_min = eps_min
        self.learning_starts = learning_starts
        self.rng = np.random.default_rng(seed)

        self.N = env.S * env.I
        d, A = env.feat_dim, env.n_actions
        self.q = BatchedMLP(self.N, d, A, hidden)
        self.q_target = BatchedMLP(self.N, d, A, hidden)
        self.q_target.load_state_dict(self.q.state_dict())
        self.opt = torch.optim.Adam(self.q.parameters(), lr=lr)
        # Profits are O(0.1-1); scale so Q-targets are O(1) for the optimiser.
        self.r_scale = 1.0 / max(env.bench.profit_nash, 1e-6)

        cap = buffer_size
        self.buf_s = torch.zeros(self.N, cap, d)
        self.buf_a = torch.zeros(self.N, cap, dtype=torch.long)
        self.buf_r = torch.zeros(self.N, cap)
        self.buf_s2 = torch.zeros(self.N, cap, d)
        self.cap, self.ptr, self.size = cap, 0, 0
        self._steps = 0

    def epsilon(self, t: int) -> float:
        return max(self.eps_min, float(np.exp(-self.beta_decay * t)))

    def _feat(self, obs) -> torch.Tensor:
        return torch.from_numpy(obs["feat"].reshape(self.N, 1, -1))

    @torch.no_grad()
    def act(self, obs, t, greedy: bool = False) -> np.ndarray:
        a = self.q(self._feat(obs)).argmax(-1).view(self.env.S, self.env.I).numpy()
        if greedy:
            return a
        explore = self.rng.random(a.shape) < self.epsilon(t)
        if explore.any():
            a = np.where(explore, self.rng.integers(self.env.n_actions, size=a.shape), a)
        return a

    def observe(self, obs, a, r, obs2, t) -> None:
        i = self.ptr
        self.buf_s[:, i] = self._feat(obs)[:, 0]
        self.buf_a[:, i] = torch.from_numpy(a.reshape(self.N))
        self.buf_r[:, i] = torch.from_numpy((r * self.r_scale).reshape(self.N).astype(np.float32))
        self.buf_s2[:, i] = self._feat(obs2)[:, 0]
        self.ptr = (i + 1) % self.cap
        self.size = min(self.size + 1, self.cap)
        self._steps += 1

        if self._steps >= self.learning_starts and self._steps % self.train_every == 0:
            self._train()
        if self._steps % self.target_every == 0:
            self.q_target.load_state_dict(self.q.state_dict())

    def _train(self) -> None:
        idx = torch.randint(0, self.size, (self.N, self.batch_size))
        g = lambda buf: torch.gather(buf, 1, idx)  # noqa: E731
        gd = lambda buf: torch.gather(buf, 1, idx[..., None].expand(-1, -1, buf.shape[-1]))  # noqa: E731
        s, s2 = gd(self.buf_s), gd(self.buf_s2)
        a, r = g(self.buf_a), g(self.buf_r)

        with torch.no_grad():
            a2 = self.q(s2).argmax(-1, keepdim=True)
            q2 = self.q_target(s2).gather(-1, a2).squeeze(-1)
            target = r + self.gamma * q2
        q = self.q(s).gather(-1, a[..., None]).squeeze(-1)
        # Sum over networks so each net gets the gradient of its own mean loss.
        loss = F.smooth_l1_loss(q, target, reduction="none").mean(1).sum()
        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        self.opt.step()
