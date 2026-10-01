"""Over-pruning in isolation: one informed trader, a fixed price rule, no rivals.

With I = 1 and lambda frozen at the monopoly value, the optimal order is
v / (2 lambda) and nothing strategic can happen. Any shortfall in the learned
intensity is pure learning bias. We vary the step size and the update rule.

    PYTHONPATH=src python experiments/mechanism.py
"""

from __future__ import annotations

import json
import os

import numpy as np

from kylecollusion.agents.tabular_q import TabularQ
from kylecollusion.market import KyleMarket, MarketConfig

ALPHAS = (0.02, 0.05, 0.1, 0.2, 0.4)
UPDATES = ("taken", "counterfactual")
SESSIONS, STEPS, BETA = 100, 300_000, 2e-5


GAMMAS = (0.0, 0.5, 0.8, 0.95)


STATES = (1, 7, 35, 155)


def learned_intensity(alpha: float, update: str, seed: int = 0, gamma: float = 0.0,
                      n_states: int = 1) -> np.ndarray:
    mem = "none" if n_states == 1 else "random"
    cfg = MarketConfig(n_informed=1, memory=mem, mm_fixed=True, n_actions=41,
                       n_random_states=max(n_states, 1))
    env = KyleMarket(cfg, SESSIONS, seed=seed)
    agent = TabularQ(env, alpha=alpha, gamma=gamma, beta_decay=BETA, update=update, seed=seed + 1)
    obs = env.reset()
    for t in range(STEPS):
        a = agent.act(obs, t)
        r, obs2, _ = env.step(a)
        agent.observe(obs, a, r, obs2, t)
        obs = obs2
    # greedy order per (state, value), averaged over the irrelevant states
    greedy = env.grid[agent.greedy_policy()[:, 0]].mean(1)  # (S, n_values)
    v = env.values
    slope = (greedy * v).sum(1) / (v * v).sum()  # per-session intensity
    return slope * 2 * env.bench.lam_nash  # 1.0 = optimal intensity


def main_gamma():
    """Same single-trader setting, alpha fixed at 0.15, discount factor varied."""
    out = {"gammas": GAMMAS, "alpha": 0.15, "sessions": SESSIONS, "steps": STEPS, "results": {}}
    for g in GAMMAS:
        rel = learned_intensity(0.15, "taken", gamma=g)
        out["results"][str(g)] = {
            "mean": float(rel.mean()),
            "ci95": float(1.96 * rel.std(ddof=1) / np.sqrt(len(rel))),
        }
        print(f"gamma={g:<5} learned/optimal intensity = {rel.mean():.3f}", flush=True)
    os.makedirs("results/mechanism", exist_ok=True)
    json.dump(out, open("results/mechanism/single_trader_gamma.json", "w"), indent=1)


def main_states():
    """Single trader with an irrelevant random state of varying size."""
    out = {"states": STATES, "alpha": 0.15, "results": {}}
    for g in (0.95, 0.0):
        for n in STATES:
            rel = learned_intensity(0.15, "taken", gamma=g, n_states=n)
            out["results"][f"{g}_{n}"] = {
                "mean": float(rel.mean()),
                "ci95": float(1.96 * rel.std(ddof=1) / np.sqrt(len(rel))),
            }
            print(f"gamma={g:<5} states={n:<4} learned/optimal = {rel.mean():.3f}", flush=True)
    os.makedirs("results/mechanism", exist_ok=True)
    json.dump(out, open("results/mechanism/single_trader_states.json", "w"), indent=1)


def main():
    out = {"alphas": ALPHAS, "sessions": SESSIONS, "steps": STEPS, "beta_decay": BETA, "results": {}}
    for upd in UPDATES:
        for a in ALPHAS:
            rel = learned_intensity(a, upd)
            out["results"][f"{upd}_{a}"] = {
                "mean": float(rel.mean()),
                "ci95": float(1.96 * rel.std(ddof=1) / np.sqrt(len(rel))),
                "per_session": rel.tolist(),
            }
            print(f"{upd:>15} alpha={a:<5} learned/optimal intensity = {rel.mean():.3f} ± "
                  f"{1.96 * rel.std(ddof=1) / np.sqrt(len(rel)):.3f}", flush=True)
    os.makedirs("results/mechanism", exist_ok=True)
    json.dump(out, open("results/mechanism/single_trader.json", "w"), indent=1)


if __name__ == "__main__":
    import sys

    if "--gamma" in sys.argv:
        main_gamma()
    elif "--states" in sys.argv:
        main_states()
    else:
        main()
