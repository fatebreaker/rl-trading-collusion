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
import pickle
import time
from dataclasses import asdict

import numpy as np

from .diagnostics import convergence_stats, impulse_response, noise_shock_response
from .theory import deviation_gap
from .market import GRID_MODES, MEMORY_MODES, KyleMarket, MarketConfig
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


def evaluate(env: KyleMarket, agent, obs: dict, steps: int):
    """Greedy play with learning frozen. Returns (log, final obs, on-path mask).

    The on-path mask marks (session, trader, state, value) entries visited
    during greedy play; it is None for agents without a table."""
    keys = ("v", "p", "y", "lam")
    log = {k: np.empty((steps, env.S)) for k in keys}
    log["x"] = np.empty((steps, env.S, env.I))
    log["profit"] = np.empty((steps, env.S, env.I))
    onpath = None
    if hasattr(agent, "greedy_policy"):
        onpath = np.zeros((env.S, env.I, env.n_states, env.n_values), dtype=bool)
        si, ii = np.arange(env.S)[:, None], np.arange(env.I)[None, :]
    for t in range(steps):
        if onpath is not None:
            onpath[si, ii, obs["s"], obs["v_idx"][:, None]] = True
        a = agent.act(obs, t, greedy=True)
        r, obs, info = env.step(a)
        for k in keys:
            log[k][t] = info[k]
        log["x"][t] = info["x"]
        log["profit"][t] = r
    return log, obs, onpath


def _save_checkpoint(path: str, state: dict) -> None:
    tmp = path + ".tmp"
    with open(tmp, "wb") as fh:
        pickle.dump(state, fh, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, path)  # atomic: a crash mid-write never corrupts the checkpoint


def train(
    env: KyleMarket,
    agent,
    steps: int,
    log_every: int = 0,
    conv_window: int = 0,
    checkpoint: str = "",
    checkpoint_every: int = 0,
):
    """Returns (env, agent, final obs, greedy table snapshot taken
    `conv_window` steps before the end or None if the agent has no table).

    With `checkpoint`, the full training state (market, agent, RNGs, step) is
    saved every `checkpoint_every` steps and training resumes from it if the
    file exists, so an interrupted run continues bit-for-bit where it stopped.
    env and agent are returned because resuming replaces them.
    """
    start, snap = 0, None
    if checkpoint and os.path.exists(checkpoint):
        with open(checkpoint, "rb") as fh:
            st = pickle.load(fh)
        env, agent, obs, start, snap = st["env"], st["agent"], st["obs"], st["t"], st["snap"]
        print(f"  resumed from {checkpoint} at step {start:,}", flush=True)
    else:
        obs = env.reset()
    t0 = time.time()
    snap_at = steps - conv_window if conv_window and hasattr(agent, "greedy_policy") else -1
    for t in range(start, steps):
        if checkpoint and checkpoint_every and t > start and t % checkpoint_every == 0:
            _save_checkpoint(
                checkpoint, {"env": env, "agent": agent, "obs": obs, "t": t, "snap": snap}
            )
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
    return env, agent, obs, snap


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
        grid_mode=args.grid_mode,
        xi=args.xi,
        theta=args.theta,
        n_price_bins=args.n_price_bins,
        n_random_states=args.n_random_states,
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
    env, agent, obs, snap = train(
        env, agent, args.steps, log_every=args.log_every, conv_window=args.conv_window,
        checkpoint=args.checkpoint, checkpoint_every=args.checkpoint_every,
    )
    train_s = time.time() - t0

    log, obs, onpath = evaluate(env, agent, obs, args.eval_steps)
    per_session = session_metrics(log, env.bench)
    if snap is not None:
        if onpath is not None and onpath.shape[1] != snap.shape[1]:  # shared table
            onpath = onpath.any(axis=1, keepdims=True)
        per_session.update(convergence_stats(snap, agent.greedy_policy(), onpath))
    summary = summarize(per_session)

    impulse = None
    if args.impulse_reps > 0:
        impulse = impulse_response(
            env, agent, obs, horizon=args.impulse_horizon, reps=args.impulse_reps,
            gamma=getattr(agent, "gamma", 0.95),
        )
    shocks = []
    dev_unit = abs(deviation_gap(env.bench)) * float(np.mean(np.abs(env.values)))
    for f in [float(x) for x in args.shock_devs.split(",") if x.strip()]:
        sh = noise_shock_response(env, agent, obs, f * dev_unit, reps=args.impulse_reps or 20)
        sh["shock_devs"] = f
        sh["shock_over_sigma_u"] = f * dev_unit / env.cfg.sigma_u
        shocks.append(sh)

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
    if impulse is not None:
        result["impulse"] = impulse
    if shocks:
        result["noise_shocks"] = shocks
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
    ap.add_argument("--grid-mode", choices=GRID_MODES, default="wide")
    ap.add_argument("--xi", type=float, default=0.0)
    ap.add_argument("--theta", type=float, default=0.1)
    ap.add_argument("--n-price-bins", type=int, default=15)
    ap.add_argument("--n-random-states", type=int, default=35)
    ap.add_argument("--n-flow-bins", type=int, default=7)
    ap.add_argument("--mm-halflife", type=float, default=2000.0)
    ap.add_argument("--mm-fixed", action="store_true")
    ap.add_argument("--gamma", type=float, default=None)
    ap.add_argument("--agent-kwargs", type=str, default="", help='JSON, e.g. \'{"alpha": 0.1}\'')
    ap.add_argument("--log-every", type=int, default=0)
    ap.add_argument("--conv-window", type=int, default=100_000,
                    help="measure greedy-strategy changes over the last N training steps (tabular only)")
    ap.add_argument("--impulse-reps", type=int, default=0,
                    help="deviation events per session for the punishment test (0 = skip)")
    ap.add_argument("--impulse-horizon", type=int, default=15)
    ap.add_argument("--shock-devs", type=str, default="",
                    help="comma-separated noise shocks for the Dou et al. noise-shock test, "
                         "in units of a typical one-period best-response deviation "
                         "(1.0 moves flow as much as a deviation at |v| = E|v|), e.g. '0.25,1'")
    ap.add_argument("--checkpoint", type=str, default="",
                    help="pickle file for periodic training state; resumes from it if present")
    ap.add_argument("--checkpoint-every", type=int, default=500_000)
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
    for k, note in (
        ("policy_change", "all table entries"),
        ("policy_change_onpath", "entries used in greedy play"),
        ("order_shift_onpath", "mean |order change| on-path, grid steps"),
    ):
        if k in s:
            print(f"{k:<22}{s[k]['mean']:>8.4f}{s[k]['ci95']:>9.4f}   "
                  f"({note}, last {args.conv_window:,} steps)")
    imp = res.get("impulse")
    if imp:
        print(f"\npunishment test: {imp['n_events']} deviation events "
              f"({imp['share_deviated']:.0%} of draws changed the deviator's order)")
        print("lag   rival d_beta          deviator d_profit")
        for k in range(min(8, imp["horizon"] + 1)):
            print(f"{k:>3}   {imp['d_beta_rival'][k]:>+7.4f} ± {imp['d_beta_rival_ci95'][k]:.4f}"
                  f"   {imp['d_profit_dev'][k]:>+7.4f} ± {imp['d_profit_dev_ci95'][k]:.4f}")
        print(f"deviator gain: period 0 {imp['gain_dev_period0']:+.4f}, discounted over "
              f"{imp['horizon'] + 1} periods {imp['cum_gain_dev']:+.4f} ± {imp['cum_gain_dev_ci95']:.4f}")
    for sh in res.get("noise_shocks", []):
        print(f"\nnoise shock {sh['shock_devs']} deviations = {sh['shock_over_sigma_u']:.2f} sigma_u:"
              f"  lag   d_beta per trader      d_price(signed)")
        for k in range(min(4, sh["horizon"] + 1)):
            print(f"                          {k:>3}   {sh['d_beta_all'][k]:>+8.4f} ± {sh['d_beta_all_ci95'][k]:.4f}"
                  f"   {sh['d_price'][k]:>+9.5f} ± {sh['d_price_ci95'][k]:.5f}")
    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w") as fh:
            json.dump(res, fh, indent=1)
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
