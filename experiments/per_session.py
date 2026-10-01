"""Session-level classification in the Dou et al. (2025) protocol.

Dou et al. (online appendix 4.4-4.5) classify each session as price-trigger
collusion if, one period after a noise shock that moves the price by 1.2%
(their "medium deviation"), both speculators' order flows rise significantly;
as over-pruning if neither changes. Averages over sessions can hide a minority
of punishing sessions, so we apply the same per-session test.

Trains the baseline (gamma = 0.95) and the myopic placebo (gamma = 0) with the
faithful protocol, keeps the trained state, and for each session measures each
trader's change in order flow at lag 1 after shocks sized as price moves of
0.12%, 1.2%, 5.5% and 7.2% of the average sign-adjusted price (their small,
medium, large and ultra-large deviations), averaged over many paired replays.

    PYTHONPATH=src NUMBA_NUM_THREADS=4 python experiments/per_session.py
"""

import copy
import json
import math
import os
import pickle
import time

import numpy as np

from kylecollusion.agents.tabular_q import TabularQ
from kylecollusion.fast import fast_train
from kylecollusion.market import KyleMarket, MarketConfig

STEPS = int(os.environ.get("PS_STEPS", 400_000_000))
SESSIONS = 100
REPS = int(os.environ.get("PS_REPS", 200))
PRICE_DEVS = (0.0012, 0.012, 0.055, 0.072)
OUT = os.environ.get("PS_OUT", "results/per_session")


def build(gamma, seed=0):
    cfg = MarketConfig(n_informed=2, sigma_u=0.1, n_values=10, n_actions=15, grid_mode="bracket",
                       memory="price", price_bins="dou", n_price_bins=31, xi=500.0)
    env = KyleMarket(cfg, SESSIONS, seed=seed)
    ag = TabularQ(env, alpha=0.01, gamma=gamma, beta_decay=5e-7, explore_by_value=True, seed=seed + 1)
    env.reset()
    return env, ag


def per_session_response(env, ag, obs, shock, reps=REPS, gap=50):
    """Per session and trader: mean and s.e. of the relative change in the
    sign-adjusted order x * sign(v) at lag 1 (in % of its mean), and the
    price move at lag 0."""
    S, I = env.S, env.I
    dx = np.zeros((reps, S, I))
    dp0 = np.zeros((reps, S))
    for r in range(reps):
        for _ in range(gap):
            _, obs, _ = env.step(ag.act(obs, 0, greedy=True))
        e_s = copy.deepcopy(env)
        o_s = {k: v.copy() for k, v in obs.items()}
        o_b = obs
        e_s.u_shock = shock * np.sign(env.values[env.v_idx])
        for k in range(2):
            a_b, a_s = ag.act(o_b, 0, greedy=True), ag.act(o_s, 0, greedy=True)
            _, o_b, i_b = env.step(a_b)
            _, o_s, i_s = e_s.step(a_s)
            sv = np.sign(i_b["v"])
            if k == 0:
                dp0[r] = sv * (i_s["p"] - i_b["p"])
            else:
                dx[r] = sv[:, None] * (i_s["x"] - i_b["x"])
        obs = o_b
    return dx, dp0, obs


def main():
    os.makedirs(OUT, exist_ok=True)
    os.makedirs("checkpoints/per_session", exist_ok=True)
    summary = {}
    for name, gamma in (("baseline_g095", 0.95), ("myopic_g0", 0.0)):
        ck = f"checkpoints/per_session/{name}_{STEPS}.pkl"
        if os.path.exists(ck):
            env, ag = pickle.load(open(ck, "rb"))
        else:
            env, ag = build(gamma)
            t0 = time.time()
            chunk = 20_000_000
            for t in range(0, STEPS, chunk):
                fast_train(env, ag, t, t + chunk)
                print(f"{name}: {t + chunk:,} steps ({time.time() - t0:.0f}s)", flush=True)
            pickle.dump((env, ag), open(ck, "wb"))
        obs = env._obs()
        # long-run means of sign-adjusted order and price under greedy play
        xs, ps = [], []
        for _ in range(5000):
            a = ag.act(obs, 0, greedy=True)
            _, obs, info = env.step(a)
            sv = np.sign(info["v"])
            xs.append(sv[:, None] * info["x"])
            ps.append(sv * info["p"])
        mean_x = np.mean(xs, 0)  # (S, I)
        mean_p = np.mean(ps, 0)  # (S,)
        lam = float(env.lam.mean())
        res = {"gamma": gamma, "steps": STEPS, "sessions": SESSIONS, "reps": REPS, "shocks": []}
        for dev in PRICE_DEVS:
            shock = dev * float(mean_p.mean()) / lam  # flow shock that moves the price by dev
            dx, dp0, obs = per_session_response(env, ag, obs, shock)
            rel = 100 * dx / mean_x[None]  # % of each trader's mean order
            m = rel.mean(0)  # (S, I)
            se = rel.std(0, ddof=1) / math.sqrt(REPS)
            sig_up = (m - 1.96 * se) > 0
            sig_flat = np.abs(m) <= 1.96 * se
            trigger = sig_up.all(1)
            flat = sig_flat.all(1)
            res["shocks"].append({
                "price_dev": dev, "shock_flow": shock,
                "price_move_pct": float(100 * dp0.mean() / mean_p.mean()),
                "mean_resp_pct": float(m.mean()), "mean_resp_ci": float(1.96 * m.std(ddof=1) / math.sqrt(SESSIONS * 2)),
                "share_trigger": float(trigger.mean()), "share_flat": float(flat.mean()),
                "resp_quantiles": np.quantile(m.mean(1), [0.05, 0.25, 0.5, 0.75, 0.95]).tolist(),
                "per_session_resp": m.tolist(),
            })
            print(f"{name} price dev {100*dev:.2f}%: move {res['shocks'][-1]['price_move_pct']:.2f}%, "
                  f"resp {m.mean():+.3f}%, trigger {trigger.mean():.0%}, flat {flat.mean():.0%}, "
                  f"quantiles {np.round(res['shocks'][-1]['resp_quantiles'], 2)}", flush=True)
        json.dump(res, open(f"{OUT}/{name}.json", "w"), indent=1)
        summary[name] = res
    return summary


if __name__ == "__main__":
    main()
