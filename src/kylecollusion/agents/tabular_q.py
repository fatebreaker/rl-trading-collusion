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

Update target: "taken" is standard Q-learning, which only learns about the
order actually submitted. "counterfactual" updates every order at once with
the profit it would have earned (Asker, Fershtman & Pakes's synchronous
learning). In a Kyle market this is exact, because the price is linear in the
trader's own order; it requires a memory state that does not depend on the
trader's own order (none or residual), so all orders share one next state.

shared=True gives all traders in a market one common Q-table, updated by each
trader's experience in turn ("shared matrix, sequential updates" in Esquinas
Coves's reimplementation of Dou et al. 2025). It is a single learner acting
for several traders, so its policy can coordinate them without any strategy.
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
        update: str = "taken",
        shared: bool = False,
        explore_by_value: bool = False,
        stop_unchanged: int = 0,
        seed: int = 0,
    ):
        if shared and update != "taken":
            raise ValueError("shared tables support only standard updates")
        if update not in ("taken", "counterfactual"):
            raise ValueError("update must be 'taken' or 'counterfactual'")
        if update == "counterfactual" and env.cfg.memory not in ("none", "residual"):
            raise ValueError("counterfactual updates need memory 'none' or 'residual'")
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
        self.update = update
        self.shared = shared
        # Dou et al. (2025): epsilon depends on how often the current value has
        # been visited, eps = exp(-beta * t(v)), instead of on calendar time.
        self.explore_by_value = explore_by_value
        # Dou et al.'s stopping rule (compiled engine only): a session stops
        # learning once its greedy strategies are unchanged for this many
        # consecutive periods. 0 = train for the full horizon.
        self.stop_unchanged = stop_unchanged
        self.conv_count = self.done = self.conv_time = None
        self.vcount = np.zeros((env.S, env.n_values), dtype=np.int64) if explore_by_value else None
        self.rng = np.random.default_rng(seed)

        S, I = env.S, env.I
        x = env.grid_v  # (n_values, n_actions): order sizes per value
        v = env.values[:, None]
        lam0 = env.bench.lam_nash
        # E[(v - lam*(x + others + passive*v + u)) x] with others zero-mean.
        stage = v * x - lam0 * (x * x + env.passive_beta * v * x)
        q0 = stage / (1.0 - gamma)  # (n_values, n_actions)
        n_tables = 1 if shared else I
        self.Q = np.broadcast_to(
            q0, (S, n_tables, env.n_states, env.n_values, env.n_actions)
        ).copy()
        self.visits = (
            np.zeros(self.Q.shape, dtype=np.uint32) if alpha_schedule == "visits" else None
        )
        self._si = np.arange(S)[:, None]
        self._ii = np.zeros((1, I), dtype=np.int64) if shared else np.arange(I)[None, :]

    def epsilon(self, t: int) -> float:
        return float(np.exp(-self.beta_decay * t))

    def _rows(self, obs: dict) -> np.ndarray:
        return self.Q[self._si, self._ii, obs["s"], obs["v_idx"][:, None]]

    def act(self, obs: dict, t: int, greedy: bool = False) -> np.ndarray:
        a = self._rows(obs).argmax(axis=-1)
        if greedy:
            return a
        if self.explore_by_value:
            si = np.arange(self.env.S)
            eps = np.exp(-self.beta_decay * self.vcount[si, obs["v_idx"]])
            self.vcount[si, obs["v_idx"]] += 1
            explore = self.rng.random(a.shape) < eps[:, None]
        else:
            explore = self.rng.random(a.shape) < self.epsilon(t)
        if explore.any():
            a = np.where(explore, self.rng.integers(self.env.n_actions, size=a.shape), a)
        return a

    def observe(self, obs, a, r, obs2, t) -> None:
        if self.update == "counterfactual":
            cont = self.gamma * self._rows(obs2).max(axis=-1)  # (S, I)
            target = self.env.counterfactual_profits() + cont[..., None]  # (S, I, A)
            rows = (self._si, self._ii, obs["s"], obs["v_idx"][:, None])
            self.Q[rows] += self.alpha * (target - self.Q[rows])
            return
        if self.shared:
            # Sequential: each trader's update sees the previous trader's.
            si = np.arange(self.env.S)
            for i in range(self.env.I):
                nxt = self.Q[si, 0, obs2["s"][:, i], obs2["v_idx"]].max(axis=-1)
                idx = (si, 0, obs["s"][:, i], obs["v_idx"], a[:, i])
                self.Q[idx] += self.alpha * (r[:, i] + self.gamma * nxt - self.Q[idx])
            return
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
