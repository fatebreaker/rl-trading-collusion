"""Train learning traders in the repeated Kyle market, then evaluate greedy play.

    python -m kylecollusion.run --algo q --sessions 200 --steps 1500000 --out results/q_base.json

Training and evaluation share one market: after training, exploration is
switched off and learning is frozen while the market maker keeps adapting, so
the evaluated outcome is the learned strategies facing a rational pricer.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import asdict

import numpy as np

from .market import MEMORY_MODES, KyleMarket, MarketConfig
from .metrics import session_metrics, summarize

ALGOS = ("q", "dqn", "ppo")


def make_agent(algo: str, env: KyleMarket, seed: int, **kw):
    if algo == "q":
        from .agents.tabular_q import TabularQ

        return TabularQ(env, seed=seed, **kw)
    if algo == "dqn":
        from .agents.dqn import DQN

        return DQN(env, seed=seed, **kw)
    if algo == "ppo":
        from .agents.ppo import PPO

        return PPO(env, seed=seed, **kw)
    raise ValueError(algo)


def evaluate(env: KyleMarket, agent, obs: dict, steps: int) -> tuple[dict, dict]:
    keys = ("v", "p", "y", "lam")
    log = {k: np.empty((steps, env.S)) for k in keys}
    log["x"] = np.empty((steps, env.S, env.I))
    log["profit"] = np.empty((steps, env.S, env.I))
    for t in range(steps):
        a = agent.act(obs, t, greedy=True)
        r, obs, info = env.step(a)
        for k in keys:
            log[k][t] = info[k]
        log["x"][t] = info["x"]
        log["profit"][t] = r
    return log, obs


def train(env: KyleMarket, agent, steps: int, log_every: int = 0, conv_window: int = 0):
    """Returns (final obs, per-session share of the greedy strategy that changed
    over the last `conv_window` steps, or None if the agent has no table)."""
    obs = env.reset()
    t0 = time.time()
    snap = None
    snap_at = steps - conv_window if conv_window and hasattr(agent, "greedy_policy") else -1
    for t in range(steps):
        if t == snap_at:
            snap = agent.greedy_policy().copy()
        a = agent.act(obs, t)
        r, obs2, _ = env.step(a)
        agent.observe(obs, a, r, obs2, t)
        obs = obs2
        if log_every and (t + 1) % log_every == 0:
            eps = agent.epsilon(t) if hasattr(agent, "epsilon") else float("nan")
            print(
                f"  step {t + 1:>9,}  eps={eps:.4f}  mean lam={env.lam.mean():.4f}  "
                f"({time.time() - t0:.0f}s)",
                flush=True,
            )
    change = None
    if snap is not None:
        now = agent.greedy_policy()
        change = (now != snap).reshape(env.S, -1).mean(1)
    return obs, change


def run(args) -> dict:
    cfg = MarketConfig(
        n_informed=args.n_informed,
        n_passive=args.n_passive,
        sigma_u=args.sigma_u,
        sigma_v=args.sigma_v,
        n_values=args.n_values,
        n_actions=args.n_actions,
        order_cap=args.order_cap,
        tick=args.tick,
        disclosure_noise=args.disclosure_noise,
        memory=args.memory,
        n_flow_bins=args.n_flow_bins,
        mm_halflife=args.mm_halflife,
        mm_fixed=args.mm_fixed,
    )
    env = KyleMarket(cfg, args.sessions, seed=args.seed)
    agent_kw = json.loads(args.agent_kwargs) if args.agent_kwargs else {}
    if args.gamma is not None:
        agent_kw["gamma"] = args.gamma
    agent = make_agent(args.algo, env, seed=args.seed + 1, **agent_kw)

    t0 = time.time()
    obs, change = train(env, agent, args.steps, log_every=args.log_every, conv_window=args.conv_window)
    train_s = time.time() - t0

    log, _ = evaluate(env, agent, obs, args.eval_steps)
    per_session = session_metrics(log, env.bench)
    if change is not None:
        per_session["policy_change"] = change
    summary = summarize(per_session)

    result = {
        "algo": args.algo,
        "config": asdict(cfg),
        "agent_kwargs": agent_kw,
        "steps": args.steps,
        "eval_steps": args.eval_steps,
        "sessions": args.sessions,
        "seed": args.seed,
        "train_seconds": round(train_s, 1),
        "benchmarks": env.bench.as_dict(),
        "summary": summary,
        "per_session": {k: np.asarray(v).tolist() for k, v in per_session.items()},
    }
    if hasattr(agent, "epsilon"):
        result["final_epsilon"] = agent.epsilon(args.steps)
    return result


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--algo", choices=ALGOS, default="q")
    ap.add_argument("--sessions", type=int, default=100)
    ap.add_argument("--steps", type=int, default=1_000_000)
    ap.add_argument("--eval-steps", type=int, default=20_000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-informed", type=int, default=2)
    ap.add_argument("--n-passive", type=int, default=0)
    ap.add_argument("--sigma-u", type=float, default=1.0)
    ap.add_argument("--sigma-v", type=float, default=1.0)
    ap.add_argument("--n-values", type=int, default=5)
    ap.add_argument("--n-actions", type=int, default=31)
    ap.add_argument("--order-cap", type=float, default=None)
    ap.add_argument("--tick", type=float, default=0.0)
    ap.add_argument("--disclosure-noise", type=float, default=0.0)
    ap.add_argument("--memory", choices=MEMORY_MODES, default="residual")
    ap.add_argument("--n-flow-bins", type=int, default=7)
    ap.add_argument("--mm-halflife", type=float, default=2000.0)
    ap.add_argument("--mm-fixed", action="store_true")
    ap.add_argument("--gamma", type=float, default=None)
    ap.add_argument("--agent-kwargs", type=str, default="", help='JSON, e.g. \'{"alpha": 0.1}\'')
    ap.add_argument("--log-every", type=int, default=0)
    ap.add_argument("--conv-window", type=int, default=100_000,
                    help="measure greedy-strategy changes over the last N training steps (tabular only)")
    ap.add_argument("--out", type=str, default="")
    return ap.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    res = run(args)
    b, s = res["benchmarks"], res["summary"]
    print(f"\n{args.algo}  I={args.n_informed} P={args.n_passive} sigma_u={args.sigma_u} "
          f"memory={args.memory}  sessions={args.sessions}  train {res['train_seconds']}s")
    print(f"{'metric':<18}{'learned':>12}{'±95%':>9}{'nash':>10}{'collusive':>11}")
    rows = [
        ("profit", "profit_nash", "profit_coll"),
        ("agg_intensity", "agg_nash", "agg_coll"),
        ("lam", "lam_nash", "lam_coll"),
        ("informativeness", "info_nash", "info_coll"),
    ]
    for k, n, c in rows:
        print(f"{k:<18}{s[k]['mean']:>12.4f}{s[k]['ci95']:>9.4f}{b[n]:>10.4f}{b[c]:>11.4f}")
    for k in ("delta_profit", "delta_intensity", "delta_info"):
        print(f"{k:<18}{s[k]['mean']:>12.3f}{s[k]['ci95']:>9.3f}   (0 = Nash, 1 = collusion)")
    print(f"{'order_r2':<18}{s['order_r2']['mean']:>12.3f}{s['order_r2']['ci95']:>9.3f}   (1 = orders depend on v only)")
    if "policy_change" in s:
        print(f"{'policy_change':<18}{s['policy_change']['mean']:>12.4f}{s['policy_change']['ci95']:>9.4f}"
              f"   (share of strategy changed in last {args.conv_window:,} steps)")
    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w") as fh:
            json.dump(res, fh, indent=1)
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
