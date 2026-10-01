"""Competing Q-learning dealers under adverse selection.

A stylised version of the dealer market studied by Colliard, Foucault and
Lovo: two market makers post ask quotes on a grid; each period one buyer
arrives. With probability pi the buyer is informed and buys only when the
asset is worth v = +sigma (v = +-sigma with equal probability). Otherwise it
is a liquidity buyer with private valuation w ~ U[0, 1] who buys when w
exceeds the best ask. The dealer with the lowest ask serves the order (ties
split at random) and earns ask - v.

Expected profit of the best ask a:
    pi(a) = (1 - pi_inf) (1 - a) a - pi_inf / 2 * (sigma - a).
Benchmarks: the competitive (Glosten-Milgrom zero-profit) ask, the smallest a
with pi(a) = 0, and the monopoly ask maximising pi(a). The collusion index is
measured on the best ask, Delta = (a - a^N) / (a^M - a^N).

Feedback is noisy (the buyer's type and v are random), which is exactly what
the pruning mechanism needs; and the profit every ask would have earned
against the same buyer and the same rival quote is known, so the
counterfactual (synchronous) update applies exactly.

The class reuses the Bertrand machinery: `payoff` holds expected profits
(used for evaluation and the deviation test), `step` samples realised ones.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .bertrand import BertrandMarket


@dataclass
class QuoteConfig:
    n_prices: int = 15
    pi_inf: float = 0.3
    sigma: float = 0.5
    xi: float = 0.1  # grid extension beyond [a^N, a^M], as in Calvano et al.
    memory: str = "full"  # full | none | random
    profit_noise: float = 0.0  # unused; noise comes from the buyer


def quote_benchmarks(cfg: QuoteConfig) -> dict:
    p, s = cfg.pi_inf, cfg.sigma
    # (1-p) a (1-a) = p/2 (s - a)  ->  (1-p) a^2 - (1-p + p/2) a + p s / 2 = 0
    A, B, C = 1 - p, -(1 - p + p / 2), p * s / 2
    a_n = (-B - math.sqrt(B * B - 4 * A * C)) / (2 * A)
    a_m = 0.5 + p / (4 * (1 - p))
    prof = lambda a: (1 - p) * (1 - a) * a - p / 2 * (s - a)
    return {"p_nash": a_n, "p_mono": a_m, "pi_nash": 0.0, "pi_mono": prof(a_m) / 2}


class QuoteMarket(BertrandMarket):
    def __init__(self, cfg: QuoteConfig, n_sessions: int, seed: int = 0):
        self.cfg, self.S, self.I = cfg, n_sessions, 2
        self.rng = np.random.default_rng(seed)
        self.rng_mem = np.random.default_rng([seed, 7919])
        b = quote_benchmarks(cfg)
        self.bench = b
        span = b["p_mono"] - b["p_nash"]
        self.grid = np.linspace(b["p_nash"] - cfg.xi * span, b["p_mono"] + cfg.xi * span, cfg.n_prices)
        m = cfg.n_prices
        self.n_states = 1 if cfg.memory == "none" else m * m
        self.n_actions = m
        p, s = cfg.pi_inf, cfg.sigma
        a = self.grid
        best = (1 - p) * (1 - a) * a - p / 2 * (s - a)  # expected profit of serving at ask a
        i, j = np.meshgrid(np.arange(m), np.arange(m), indexing="ij")
        win0 = np.where(i < j, 1.0, np.where(i == j, 0.5, 0.0))
        lo = np.minimum(i, j)
        self.payoff = np.stack([win0 * best[lo], (1 - win0) * best[lo]], -1)
        self.last_cf = None

    def step(self, a: np.ndarray):
        cfg, S, g = self.cfg, self.S, self.grid
        informed = self.rng.random(S) < cfg.pi_inf
        v = np.where(self.rng.random(S) < 0.5, cfg.sigma, -cfg.sigma)
        w = self.rng.random(S)
        coin = self.rng.random(S) < 0.5  # tie-break: firm 0 wins if True

        inf_, v_, w_, c_ = informed[:, None], v[:, None], w[:, None], coin[:, None]

        def outcome(own, riv, firm):
            """Realised profit of `firm` quoting grid indices own (S, K) when
            the rival quotes riv (S, 1), against this period's buyer."""
            ask = g[own]
            buys = np.where(inf_, v_, w_) > ask
            tie_win = c_ if firm == 0 else ~c_
            wins = (own < riv) | ((own == riv) & tie_win)
            return np.where(buys & wins, ask - v_, 0.0)

        r = np.concatenate(
            [outcome(a[:, 0:1], a[:, 1:2], 0), outcome(a[:, 1:2], a[:, 0:1], 1)], 1
        )
        # profit of every own quote against the same buyer and rival quote
        allq = np.broadcast_to(np.arange(self.n_actions), (S, self.n_actions))
        self.last_cf = np.stack([outcome(allq, a[:, 1:2], 0), outcome(allq, a[:, 0:1], 1)], 1)
        self.prev = a.copy()
        return r, self._state()
