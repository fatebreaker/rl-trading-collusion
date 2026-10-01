"""Calvano et al. (2020) repeated logit-Bertrand game with batched Q-learning.

Used to validate the diagnostics on the canonical case where algorithmic
collusion is held to be genuine (sustained by punishment). If the deviation
test and the placebos say "collusion" here and "pruning" in the Kyle market,
the diagnostics discriminate rather than always returning the same verdict.

Setup (baseline of Calvano et al. 2020): n = 2 firms, logit demand
    q_i = exp((a_i - p_i)/mu) / (sum_j exp((a_j - p_j)/mu) + exp(a_0/mu)),
a_i = 2, a_0 = 0, mu = 1/4, marginal cost c = 1, profit (p_i - c) q_i.
Prices on m = 15 points spanning the Nash-to-monopoly range extended by
xi = 0.1 on both sides. State: both firms' prices last period (memory 1).
Q initialised to the discounted payoff against a uniformly random rival;
epsilon_t = exp(-beta t); alpha = 0.15, beta = 4e-6, delta = 0.95.

Memory variants mirror the Kyle experiments: "full" (both last prices),
"none" (one state), "random" (an uninformative state with as many values as
"full"). Optional profit noise lets us ask whether the pruning bias appears
once payoffs are noisy.
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass

import numpy as np


@dataclass
class BertrandConfig:
    n_prices: int = 15
    a: float = 2.0
    a0: float = 0.0
    mu: float = 0.25
    cost: float = 1.0
    xi: float = 0.1
    memory: str = "full"  # full | none | random
    profit_noise: float = 0.0  # sd of additive noise on realised profit


def logit_demand(p: np.ndarray, cfg: BertrandConfig) -> np.ndarray:
    """p: (..., 2) prices -> (..., 2) quantities."""
    e = np.exp((cfg.a - p) / cfg.mu)
    return e / (e.sum(-1, keepdims=True) + math.exp(cfg.a0 / cfg.mu))


def profits(p: np.ndarray, cfg: BertrandConfig) -> np.ndarray:
    return (p - cfg.cost) * logit_demand(p, cfg)


def benchmarks(cfg: BertrandConfig) -> dict:
    """Symmetric static Nash price (best-response iteration) and the
    symmetric joint-profit-maximising (monopoly) price."""
    def br(p_other):
        grid = np.linspace(cfg.cost, cfg.cost + 3, 30001)
        pr = profits(np.stack([grid, np.full_like(grid, p_other)], -1), cfg)[:, 0]
        return grid[pr.argmax()]

    p = 1.5
    for _ in range(200):
        p_new = br(p)
        if abs(p_new - p) < 1e-9:
            break
        p = p_new
    p_n = p
    grid = np.linspace(cfg.cost, cfg.cost + 3, 30001)
    joint = profits(np.stack([grid, grid], -1), cfg).sum(-1)
    p_m = grid[joint.argmax()]
    pi_n = profits(np.array([p_n, p_n]), cfg)[0]
    pi_m = profits(np.array([p_m, p_m]), cfg)[0]
    return {"p_nash": float(p_n), "p_mono": float(p_m), "pi_nash": float(pi_n), "pi_mono": float(pi_m)}


class BertrandMarket:
    def __init__(self, cfg: BertrandConfig, n_sessions: int, seed: int = 0):
        self.cfg, self.S, self.I = cfg, n_sessions, 2
        self.rng = np.random.default_rng(seed)
        self.rng_mem = np.random.default_rng([seed, 7919])
        b = benchmarks(cfg)
        self.bench = b
        span = b["p_mono"] - b["p_nash"]
        self.grid = np.linspace(b["p_nash"] - cfg.xi * span, b["p_mono"] + cfg.xi * span, cfg.n_prices)
        m = cfg.n_prices
        pa, pb = np.meshgrid(self.grid, self.grid, indexing="ij")
        self.payoff = profits(np.stack([pa, pb], -1), cfg)  # (m, m, 2): own index first for firm 0
        self.n_states = 1 if cfg.memory == "none" else m * m
        self.n_actions = m

    def reset(self):
        self.prev = self.rng.integers(self.n_actions, size=(self.S, 2))
        return self._state()

    def _state(self):
        m = self.n_actions
        if self.cfg.memory == "none":
            return np.zeros((self.S, 2), dtype=np.int64)
        if self.cfg.memory == "random":
            return self.rng_mem.integers(m * m, size=(self.S, 2))
        s = self.prev[:, 0] * m + self.prev[:, 1]
        # each firm sees (own, rival) ordering so both tables are symmetric
        s1 = self.prev[:, 1] * m + self.prev[:, 0]
        return np.stack([s, s1], 1)

    def step(self, a: np.ndarray):
        """a: (S, 2) price indices -> profits (S, 2), next state."""
        r0 = self.payoff[a[:, 0], a[:, 1], 0]
        r1 = self.payoff[a[:, 0], a[:, 1], 1]
        r = np.stack([r0, r1], 1)
        # profit each own price would have earned against the rival's actual price
        self.last_cf = np.stack([self.payoff[:, a[:, 1], 0].T, self.payoff[a[:, 0], :, 1]], 1)
        if self.cfg.profit_noise > 0:
            noise = self.rng.normal(0, self.cfg.profit_noise, size=r.shape)
            r = r + noise
            self.last_cf = self.last_cf + noise[..., None]
        self.prev = a.copy()
        return r, self._state()


class BertrandQ:
    def __init__(self, env: BertrandMarket, alpha=0.15, gamma=0.95, beta_decay=4e-6, seed=0,
                 update="taken"):
        """update="counterfactual" (synchronous learning, Asker et al.) updates
        every own price with the profit it would have earned against the
        rival's actual price, using the next state that price would have led
        to. Needs env.last_cf (S, 2, m) of realised counterfactual profits."""
        self.env, self.alpha, self.gamma, self.beta = env, alpha, gamma, beta_decay
        self.update = update
        self.rng = np.random.default_rng(seed)
        m = env.n_actions
        # discounted payoff of each own price against a uniformly random rival
        own = env.payoff[:, :, 0].mean(1)  # (m,)
        q0 = own / (1 - gamma) if gamma < 1 else own
        self.Q = np.broadcast_to(q0, (env.S, 2, env.n_states, m)).copy()
        self._si = np.arange(env.S)[:, None]
        self._ii = np.arange(2)[None, :]

    def epsilon(self, t):
        return math.exp(-self.beta * t)

    def act(self, s, t, greedy=False):
        a = self.Q[self._si, self._ii, s].argmax(-1)
        if greedy:
            return a
        explore = self.rng.random(a.shape) < self.epsilon(t)
        if explore.any():
            a = np.where(explore, self.rng.integers(self.env.n_actions, size=a.shape), a)
        return a

    def observe(self, s, a, r, s2):
        if self.update == "counterfactual":
            m = self.env.n_actions
            cf = self.env.last_cf  # (S, 2, m)
            mem = self.env.cfg.memory
            if mem == "full":
                rival = a[:, ::-1]  # (S, 2): each firm's rival price index
                nxt = np.arange(m)[None, None, :] * m + rival[..., None]  # (S, 2, m), own first
            else:  # none / random: the next state does not depend on the own price
                nxt = np.broadcast_to(s2[..., None], cf.shape)
            cont = self.Q[self._si[..., None], self._ii[..., None], nxt].max(-1)  # (S, 2, m)
            rows = (self._si, self._ii, s)
            self.Q[rows] += self.alpha * (cf + self.gamma * cont - self.Q[rows])
            return
        target = r + self.gamma * self.Q[self._si, self._ii, s2].max(-1)
        idx = (self._si, self._ii, s, a)
        self.Q[idx] += self.alpha * (target - self.Q[idx])

    def greedy_policy(self):
        return self.Q.argmax(-1)


def train(env, agent, steps, conv_window=100_000):
    s = env.reset()
    snap = None
    for t in range(steps):
        if t == steps - conv_window:
            snap = agent.greedy_policy().copy()
        a = agent.act(s, t)
        r, s2 = env.step(a)
        agent.observe(s, a, r, s2)
        s = s2
    change = (agent.greedy_policy() != snap).reshape(env.S, -1).mean(1) if snap is not None else None
    return s, change


def evaluate(env, agent, s, periods=1000):
    """Greedy play; deterministic payoffs reach a cycle quickly. Returns
    per-session average profit (both firms) and average price."""
    prof = np.zeros(env.S)
    price = np.zeros(env.S)
    burn = 200
    for t in range(periods):
        a = agent.act(s, 0, greedy=True)
        r, s = env.step(a)
        if t >= burn:
            # evaluate on expected profits, not noisy realised ones
            exp_r = env.payoff[a[:, 0], a[:, 1]]
            prof += exp_r.mean(1)
            price += env.grid[a].mean(1)
    n = periods - burn
    return prof / n, price / n, s


def deviation_response(env, agent, s, horizon=15, deviator=0):
    """Calvano impulse response: from the greedy path, firm `deviator` plays
    its static best response to the rival's current greedy price for one
    period; then both follow their strategies. Paired with an undeviated copy.
    Returns per-lag mean price changes for deviator and rival (deviated minus
    baseline) and the deviator's discounted profit gain."""
    rival = 1 - deviator
    e_dev = copy.deepcopy(env)
    s_b, s_d = s.copy(), s.copy()
    dp_dev = np.zeros((horizon + 1, env.S))
    dp_riv = np.zeros((horizon + 1, env.S))
    dpi = np.zeros((horizon + 1, env.S))
    deviated = None
    for k in range(horizon + 1):
        a_b = agent.act(s_b, 0, greedy=True)
        a_d = agent.act(s_d, 0, greedy=True)
        if k == 0:
            # static best response to rival's price in the deviating copy
            pay = env.payoff[:, a_d[:, rival], deviator] if deviator == 0 else env.payoff[a_d[:, rival], :, deviator].T
            br = pay.argmax(0)
            a_d = a_d.copy()
            a_d[:, deviator] = br
            deviated = br != a_b[:, deviator]
        pb = env.payoff[a_b[:, 0], a_b[:, 1]]
        pd = env.payoff[a_d[:, 0], a_d[:, 1]]
        dp_dev[k] = env.grid[a_d[:, deviator]] - env.grid[a_b[:, deviator]]
        dp_riv[k] = env.grid[a_d[:, rival]] - env.grid[a_b[:, rival]]
        dpi[k] = pd[:, deviator] - pb[:, deviator]
        _, s_b = env.step(a_b)
        _, s_d = e_dev.step(a_d)
    m = deviated
    n = int(m.sum())
    disc = agent.gamma ** np.arange(horizon + 1) if agent.gamma > 0 else 0.95 ** np.arange(horizon + 1)
    gains = (dpi * disc[:, None]).sum(0)[m]

    def prof(x):
        v = x[:, m]
        return v.mean(1).tolist(), (1.96 * v.std(1, ddof=1) / math.sqrt(max(n, 2))).tolist()

    out = {"n_events": n, "share_deviated": float(m.mean())}
    out["d_price_dev"], out["d_price_dev_ci95"] = prof(dp_dev)
    out["d_price_rival"], out["d_price_rival_ci95"] = prof(dp_riv)
    out["cum_gain_dev"] = float(gains.mean()) if n else float("nan")
    out["cum_gain_dev_ci95"] = float(1.96 * gains.std(ddof=1) / math.sqrt(n)) if n > 1 else float("nan")
    return out
