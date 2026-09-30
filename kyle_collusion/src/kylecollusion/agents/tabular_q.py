"""Batched tabular Q-learning, following Calvano et al. (2020) conventions.

Each (session, trader) pair owns an independent Q-table indexed by
(memory state, current value, order). Exploration is epsilon-greedy with
epsilon_t = exp(-beta_decay * t), and Q is initialised to the discounted
payoff of each order against uniformly random opponents, so no action
starts out artificially attractive.

Learning rate: "const" keeps alpha fixed as in Calvano et al. With noisy
profits a constant step keeps estimates jittering, so greedy choices among
near-equivalent orders never settle. "visits" decays the step per table entry,
alpha_n = max(alpha_min, alpha * (1 + n / kappa) ** -power), where n counts
updates of that entry (Robbins-Monro conditions hold for power in (0.5, 1]).
"""

from __future__ import annotations

import numpy as np

from ..market import KyleMarket


class TabularQ:
    name = "q"

    def __init__(
        self,
        env: KyleMarket,
        alpha: float = 0.15,
        gamma: float = 0.95,
        beta_decay: float = 4e-6,
        alpha_schedule: str = "const",
        alpha_kappa: float = 50.0,
        alpha_power: float = 0.7,
        alpha_min: float = 0.0,
        seed: int = 0,
    ):
        if alpha_schedule not in ("const", "visits"):
            raise ValueError("alpha_schedule must be 'const' or 'visits'")
        self.env = env
        self.alpha = alpha
        self.gamma = gamma
        self.beta_decay = beta_decay
        self.alpha_schedule = alpha_schedule
        self.alpha_kappa = alpha_kappa
        self.alpha_power = alpha_power
        self.alpha_min = alpha_min
        self.rng = np.random.default_rng(seed)

        S, I = env.S, env.I
        x = env.grid_v  # (n_values, n_actions): order sizes per value
        v = env.values[:, None]
        lam0 = env.bench.lam_nash
        # E[(v - lam*(x + others + passive*v + u)) x] with others zero-mean.
        stage = v * x - lam0 * (x * x + env.passive_beta * v * x)
        q0 = stage / (1.0 - gamma)  # (n_values, n_actions)
        self.Q = np.broadcast_to(
            q0, (S, I, env.n_states, env.n_values, env.n_actions)
        ).copy()
        self.visits = (
            np.zeros(self.Q.shape, dtype=np.uint32) if alpha_schedule == "visits" else None
        )
        self._si = np.arange(S)[:, None]
        self._ii = np.arange(I)[None, :]

    def epsilon(self, t: int) -> float:
        return float(np.exp(-self.beta_decay * t))

    def _rows(self, obs: dict) -> np.ndarray:
        return self.Q[self._si, self._ii, obs["s"], obs["v_idx"][:, None]]

    def act(self, obs: dict, t: int, greedy: bool = False) -> np.ndarray:
        a = self._rows(obs).argmax(axis=-1)
        if greedy:
            return a
        explore = self.rng.random(a.shape) < self.epsilon(t)
        if explore.any():
            a = np.where(explore, self.rng.integers(self.env.n_actions, size=a.shape), a)
        return a

    def observe(self, obs, a, r, obs2, t) -> None:
        target = r + self.gamma * self._rows(obs2).max(axis=-1)
        idx = (self._si, self._ii, obs["s"], obs["v_idx"][:, None], a)
        if self.visits is None:
            lr = self.alpha
        else:
            n = self.visits[idx]
            lr = np.maximum(
                self.alpha_min, self.alpha * (1.0 + n / self.alpha_kappa) ** -self.alpha_power
            )
            self.visits[idx] = n + 1
        self.Q[idx] += lr * (target - self.Q[idx])

    def greedy_policy(self) -> np.ndarray:
        """(S, I, n_states, n_values) greedy order index, for convergence checks."""
        return self.Q.argmax(axis=-1)
