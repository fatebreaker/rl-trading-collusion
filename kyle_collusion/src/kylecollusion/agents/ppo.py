"""Batched PPO (clipped objective, GAE) for the continuing Kyle game.

Each (session, trader) pair has its own actor-critic network: one BatchedMLP
whose output is n_actions policy logits plus one value. The entropy bonus is
annealed to zero so policies become near-deterministic and a converged
strategy can be read off, matching how the value-based agents are evaluated.
"""

from __future__ import annotations

import numpy as np
import torch
from torch.distributions import Categorical

from ..market import KyleMarket
from .batched_mlp import BatchedMLP, clip_grad_norm_per_net


class PPO:
    name = "ppo"

    def __init__(
        self,
        env: KyleMarket,
        gamma: float = 0.95,
        gae_lambda: float = 0.95,
        lr: float = 3e-4,
        hidden: int = 64,
        rollout_len: int = 256,
        epochs: int = 4,
        n_minibatches: int = 4,
        clip: float = 0.2,
        vf_coef: float = 0.5,
        ent_coef: float = 0.01,
        ent_anneal_steps: int = 300_000,
        seed: int = 0,
    ):
        torch.manual_seed(seed)
        self.env = env
        self.gamma = gamma
        self.lam = gae_lambda
        self.T = rollout_len
        self.epochs = epochs
        self.n_mb = n_minibatches
        self.clip = clip
        self.vf_coef = vf_coef
        self.ent_coef0 = ent_coef
        self.ent_anneal = ent_anneal_steps

        self.N = env.S * env.I
        d, A = env.feat_dim, env.n_actions
        self.A = A
        self.net = BatchedMLP(self.N, d, A + 1, hidden)
        self.opt = torch.optim.Adam(self.net.parameters(), lr=lr)
        self.r_scale = 1.0 / max(env.bench.profit_nash, 1e-6)

        self.s = torch.zeros(self.N, self.T, d)
        self.a = torch.zeros(self.N, self.T, dtype=torch.long)
        self.logp = torch.zeros(self.N, self.T)
        self.v = torch.zeros(self.N, self.T)
        self.r = torch.zeros(self.N, self.T)
        self.k = 0
        self._last_logp = None
        self._last_v = None

    def _feat(self, obs) -> torch.Tensor:
        return torch.from_numpy(obs["feat"].reshape(self.N, 1, -1))

    def _split(self, out):
        return out[..., : self.A], out[..., self.A]

    @torch.no_grad()
    def act(self, obs, t, greedy: bool = False) -> np.ndarray:
        logits, value = self._split(self.net(self._feat(obs)))
        logits, value = logits[:, 0], value[:, 0]
        if greedy:
            a = logits.argmax(-1)
        else:
            dist = Categorical(logits=logits)
            a = dist.sample()
            self._last_logp = dist.log_prob(a)
            self._last_v = value
        return a.view(self.env.S, self.env.I).numpy()

    def observe(self, obs, a, r, obs2, t) -> None:
        k = self.k
        self.s[:, k] = self._feat(obs)[:, 0]
        self.a[:, k] = torch.from_numpy(a.reshape(self.N))
        self.logp[:, k] = self._last_logp
        self.v[:, k] = self._last_v
        self.r[:, k] = torch.from_numpy((r * self.r_scale).reshape(self.N).astype(np.float32))
        self.k += 1
        if self.k == self.T:
            with torch.no_grad():
                _, v_last = self._split(self.net(self._feat(obs2)))
            self._update(v_last[:, 0], t)
            self.k = 0

    def _update(self, v_last: torch.Tensor, t: int) -> None:
        adv = torch.zeros_like(self.r)
        gae = torch.zeros(self.N)
        for k in reversed(range(self.T)):
            v_next = v_last if k == self.T - 1 else self.v[:, k + 1]
            delta = self.r[:, k] + self.gamma * v_next - self.v[:, k]
            gae = delta + self.gamma * self.lam * gae
            adv[:, k] = gae
        ret = adv + self.v
        adv = (adv - adv.mean(1, keepdim=True)) / (adv.std(1, keepdim=True) + 1e-8)

        ent_coef = self.ent_coef0 * max(0.0, 1.0 - t / self.ent_anneal)
        mb = self.T // self.n_mb
        for _ in range(self.epochs):
            perm = torch.randperm(self.T)
            for j in range(self.n_mb):
                idx = perm[j * mb : (j + 1) * mb]
                logits, value = self._split(self.net(self.s[:, idx]))
                dist = Categorical(logits=logits)
                logp = dist.log_prob(self.a[:, idx])
                ratio = torch.exp(logp - self.logp[:, idx])
                A = adv[:, idx]
                pg = -torch.min(ratio * A, ratio.clamp(1 - self.clip, 1 + self.clip) * A)
                vf = (value - ret[:, idx]) ** 2
                loss = (pg + self.vf_coef * vf - ent_coef * dist.entropy()).mean(1).sum()
                self.opt.zero_grad(set_to_none=True)
                loss.backward()
                clip_grad_norm_per_net(self.net, 0.5)
                self.opt.step()
