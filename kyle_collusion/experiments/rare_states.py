"""Do rarely visited price states trade more aggressively?

Trains a myopic (gamma = 0) shared-table Q-learner in the Dou et al. xi = 500
regime, then relates each price-surprise state's visit frequency (during
greedy play) to the greedy trading intensity in that state. If large noise
shocks land in rarely visited states where estimates have been pruned less,
the noise-shock "response" is a learning artifact.

    PYTHONPATH=src python experiments/rare_states.py
"""

from __future__ import annotations

import json
import os

import numpy as np

from kylecollusion.agents.tabular_q import TabularQ
from kylecollusion.market import KyleMarket, MarketConfig
from kylecollusion.run import train

SESSIONS, STEPS = 20, 15_000_000


def main():
    cfg = MarketConfig(sigma_u=0.1, n_values=10, n_actions=15, grid_mode="bracket",
                       memory="price", xi=500.0)
    env = KyleMarket(cfg, SESSIONS, seed=11)
    agent = TabularQ(env, alpha=0.05, gamma=0.0, beta_decay=4e-7, shared=True, seed=12)
    os.makedirs("checkpoints", exist_ok=True)
    env, agent, obs, _ = train(env, agent, STEPS, log_every=1_000_000,
                               checkpoint="checkpoints/rare_states.pkl", checkpoint_every=1_000_000)

    nb, nv = cfg.n_price_bins, cfg.n_values
    visits = np.zeros((SESSIONS, nv * nb))
    for _ in range(50_000):
        a = agent.act(obs, 0, greedy=True)
        np.add.at(visits, (np.repeat(np.arange(SESSIONS), env.I), obs["s"].ravel()), 1)
        _, obs, _ = env.step(a)

    greedy = agent.greedy_policy()[:, 0]  # (S, n_states, n_values) action index
    orders = env.grid_v[np.arange(nv)[None, None, :], greedy]  # (S, n_states, n_values)
    v = env.values
    beta = (orders * v).sum(-1) / (v * v).sum()  # intensity per (session, state)
    bn, bc = env.bench.beta_nash, env.bench.beta_coll
    rel = (beta - bc) / (bn - bc)  # 0 = cartel, 1 = Nash, per state

    pbin = np.arange(nv * nb) % nb  # price-surprise bin of each state
    freq = visits / visits.sum(1, keepdims=True)
    out = {"by_bin": {}}
    for b in range(nb):
        m = pbin == b
        out["by_bin"][int(b)] = {
            "visit_share": float(freq[:, m].sum(1).mean()),
            "nash_share_of_intensity": float(rel[:, m].mean()),
        }
    common = freq > np.median(freq[freq > 0])
    rare = (freq > 0) & ~common
    out["common_states_nash_share"] = float(rel[common].mean())
    out["rare_states_nash_share"] = float(rel[rare].mean())
    out["never_visited_nash_share"] = float(rel[freq == 0].mean()) if (freq == 0).any() else None
    print(json.dumps({k: v for k, v in out.items() if k != "by_bin"}, indent=1))
    for b, d in out["by_bin"].items():
        print(f"bin {b:>2}: visit share {d['visit_share']:.4f}  intensity (0=cartel,1=Nash) {d['nash_share_of_intensity']:.3f}")
    os.makedirs("results/mechanism", exist_ok=True)
    json.dump(out, open("results/mechanism/rare_states.json", "w"), indent=1)


if __name__ == "__main__":
    main()
