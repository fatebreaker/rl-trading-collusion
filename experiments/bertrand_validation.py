"""Apply the punishment/pruning diagnostics outside the Kyle market.

--market bertrand: the logit-Bertrand game of Calvano et al. (2020), where
collusion is held to be sustained by punishment. Runs the baseline and the
placebos used in the Kyle market (myopic gamma = 0, no memory, uninformative
random memory) plus a profit-noise variant.

--market quotes: competing dealers under adverse selection (quotes.py), with
the same placebos and counterfactual (synchronous) updates.

Each variant gets the paired deviation test. One JSON per variant is written
to results/<market>/.

    PYTHONPATH=src python experiments/bertrand_validation.py [--market quotes] [--steps N] [--only a,b]
"""

import argparse
import json
import math
import os
import time

import numpy as np

from kylecollusion.bertrand import BertrandConfig, BertrandMarket, BertrandQ, deviation_response, evaluate, train
from kylecollusion.quotes import QuoteConfig, QuoteMarket

VARIANTS = {
    "baseline": dict(cfg={}, gamma=0.95),
    "myopic": dict(cfg={}, gamma=0.0),
    "nomemory": dict(cfg={"memory": "none"}, gamma=0.95),
    "random": dict(cfg={"memory": "random"}, gamma=0.95),
    "noise": dict(cfg={"profit_noise": 0.1}, gamma=0.95),
    "noise_myopic": dict(cfg={"profit_noise": 0.1}, gamma=0.0),
    "counterfactual": dict(cfg={}, gamma=0.95, update="counterfactual"),
    # Calvano et al.'s own memoryless specification (online appendix A4.1:
    # k = 0, delta = 0, beta = 1e-4, alpha = 0.25), and the same with delta = 0.95
    "nomemory_calvano_spec": dict(cfg={"memory": "none"}, gamma=0.0, alpha=0.25, beta=1e-4, steps=200_000),
    "nomemory_calvano_spec_g095": dict(cfg={"memory": "none"}, gamma=0.95, alpha=0.25, beta=1e-4, steps=200_000),
}
QUOTE_VARIANTS = {
    "baseline": dict(cfg={}, gamma=0.95),
    "myopic": dict(cfg={}, gamma=0.0),
    "nomemory": dict(cfg={"memory": "none"}, gamma=0.95),
    "random": dict(cfg={"memory": "random"}, gamma=0.95),
    "counterfactual": dict(cfg={}, gamma=0.95, update="counterfactual"),
    "counterfactual_nomemory": dict(cfg={"memory": "none"}, gamma=0.95, update="counterfactual"),
}
MARKETS = {"bertrand": (BertrandConfig, BertrandMarket, VARIANTS),
           "quotes": (QuoteConfig, QuoteMarket, QUOTE_VARIANTS)}


def ci(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    return float(x.mean()), float(1.96 * x.std(ddof=1) / math.sqrt(len(x)))


def run(name, spec, steps, sessions, seed, out_dir, market="bertrand"):
    path = os.path.join(out_dir, f"{name}.json")
    if os.path.exists(path):
        print("skip", name)
        return
    Cfg, Market, _ = MARKETS[market]
    cfg = Cfg(**spec["cfg"])
    env = Market(cfg, sessions, seed=seed)
    ag = BertrandQ(env, gamma=spec["gamma"], seed=seed, update=spec.get("update", "taken"),
                   alpha=spec.get("alpha", 0.15), beta_decay=spec.get("beta", 4e-6))
    steps = spec.get("steps", steps)
    t0 = time.time()
    s, change = train(env, ag, steps)
    prof, price, s = evaluate(env, ag, s)
    b = env.bench
    delta = (prof - b["pi_nash"]) / (b["pi_mono"] - b["pi_nash"])
    extra = {}
    if market == "quotes":
        # the price that matters is the best ask; average it over the greedy cycle
        best, s_e = np.zeros(sessions), s
        for _ in range(200):
            a = ag.act(s_e, 0, greedy=True)
            best += env.grid[a].min(1)
            _, s_e = env.step(a)
        best /= 200
        extra["delta_best_ask"] = ci((best - b["p_nash"]) / (b["p_mono"] - b["p_nash"]))
        extra["best_ask"] = ci(best)
        s = s_e
    devs = [deviation_response(env, ag, s, deviator=d) for d in (0, 1)]
    res = {
        "name": name, "gamma": spec["gamma"], "config": spec["cfg"], "steps": steps,
        "alpha": ag.alpha, "beta": ag.beta,
        "sessions": sessions, "seed": seed, "bench": b, "minutes": (time.time() - t0) / 60,
        "delta": ci(delta), "price": ci(price),
        "policy_change_last_window": float(change.mean()),
        "deviation": devs, "market": market, "update": spec.get("update", "taken"), **extra,
    }
    with open(path, "w") as f:
        json.dump(res, f, indent=1)
    d = devs[0]
    if "delta_best_ask" in res:
        print(f"{name}: best-ask Delta {res['delta_best_ask'][0]:.3f}+-{res['delta_best_ask'][1]:.3f}", flush=True)
    print(f"{name}: Delta {res['delta'][0]:.3f}+-{res['delta'][1]:.3f} "
          f"rival dp1 {d['d_price_rival'][1]:+.3f} gain {d['cum_gain_dev']:+.4f}+-{d['cum_gain_dev_ci95']:.4f}",
          flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=2_500_000)
    ap.add_argument("--sessions", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--only", default="")
    ap.add_argument("--market", choices=list(MARKETS), default="bertrand")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    out = a.out or f"results/{a.market}"
    os.makedirs(out, exist_ok=True)
    variants = MARKETS[a.market][2]
    names = a.only.split(",") if a.only else list(variants)
    for n in names:
        run(n, variants[n], a.steps, a.sessions, a.seed, out, a.market)
