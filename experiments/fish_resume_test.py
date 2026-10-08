"""Re-run a deviation test on chosen sessions of a saved LLM pricing run, from the
state at the end of its main run, keeping per-session results.

The state of an LLM pricer is its price history (``prices_sessions`` in the run's
JSON), the demand outcomes (deterministic, so recomputed here) and, under the
protocol of Fish et al., the PLANS and INSIGHTS files it last wrote (parsed from
the run's raw transcript). Rebuilding them lets us ask which sessions drive a
response that the original test reports only on average.

    OPENAI_API_KEY=... python experiments/fish_resume_test.py \\
        --run results/llm_fish/gpt52_high_duopoly_P1.json --sessions 2,3,7 \\
        --mode best_response --events 6 --total-budget 250 --out <file>.json
    python experiments/fish_resume_test.py --run ... --sessions 2,3,7 --dry-run   # rebuild only
"""
from __future__ import annotations

import argparse
import gzip
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from kylecollusion.llm_pricing import (LLMPricers, PricingConfig, PricingMarket,  # noqa: E402
                                       deviation_test, parse_fish, parse_price)
from kylecollusion.llm_traders import BudgetExceeded, OpenAIBackend  # noqa: E402


def rebuild(run: dict, raw_path: str, sessions: list[int], objective: str | None = None):
    """Market and pricers for `sessions` of `run`, at the end of its main run. `objective`
    ("long" or "myopic") overrides the run's objective: the same state, another goal."""
    ra = run["args"]
    fish = ra["style"] == "fish"
    cfg = PricingConfig(scale=ra["scale"], history=ra.get("history") or (100 if fish else 30),
                        notes=not ra.get("no_notes", False),
                        objective=objective or ("myopic" if ra["condition"] == "myopic" else "long"),
                        temperature=ra["temperature"], style=ra["style"], prefix=ra["prefix"],
                        max_tokens=ra.get("max_tokens") or (1200 if fish else 300))
    P = np.asarray(run["prices_sessions"], float)  # (S, T, 2)
    S_full, T = P.shape[0], P.shape[1]
    full = LLMPricers(PricingMarket(cfg, S_full), cfg, seed=ra["seed"])  # the sessions' draws
    env = PricingMarket(cfg, len(sessions))
    pricers = LLMPricers(env, cfg, seed=ra["seed"])
    pricers.wtp = full.wtp[sessions].copy()
    for t in range(T):
        p = P[sessions, t, :]
        q, pi = env.step(p)
        pricers.record(p, q, pi)
    # the files (or notes) each firm last wrote in the main run
    last = {}
    with gzip.open(raw_path, "rt") as fh:
        next(fh)
        for line in fh:
            x = json.loads(line)
            if x["t"] < T and x["s"] in sessions:
                last.setdefault((x["s"], x["i"]), []).append((x["t"], x["text"]))
    for (s, i), items in last.items():
        k = sessions.index(s)
        for t, text in sorted(items, reverse=True):
            if fish:
                price, plans, insights = parse_fish(text)
                if price is not None:
                    pricers.files[k][i] = (plans, insights)
                    break
            else:
                price, notes = parse_price(text)
                if price is not None:
                    pricers.notes[k][i] = notes
                    break
    return cfg, env, pricers


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--sessions", default=None, help="comma-separated session indices (default: all)")
    ap.add_argument("--mode", default="best_response", choices=["best_response", "cut", "hike"])
    ap.add_argument("--cut", type=float, default=0.10)
    ap.add_argument("--events", type=int, default=3)
    ap.add_argument("--gap", type=int, default=5)
    ap.add_argument("--horizon", type=int, default=6)
    ap.add_argument("--run-budget", type=float, default=20.0)
    ap.add_argument("--total-budget", type=float, default=140.0)
    ap.add_argument("--objective", default=None, choices=["long", "myopic"],
                    help="override the run's objective (placebo at the run's state)")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)

    run = json.load(open(a.run))
    ra = run["args"]
    S_full = len(run["prices_sessions"])
    sessions = list(range(S_full)) if a.sessions is None else [int(x) for x in a.sessions.split(",")]
    cfg, env, pricers = rebuild(run, a.run.replace(".json", "_raw.jsonl.gz"), sessions, a.objective)
    span = env.bench["p_mono"] - env.bench["p_nash"]
    last_idx = (pricers.last - env.bench["p_nash"]) / span
    print("rebuilt sessions", sessions, "at t =", pricers.t, "| last prices (index):",
          np.round(last_idx, 2).tolist(), flush=True)
    if a.dry_run:
        print(pricers.fish_prompt(0, 0)[:1500] if cfg.style == "fish" else pricers.user_prompt(0, 0)[:1500])
        return
    backend = OpenAIBackend(ra["model"], reasoning_effort=ra.get("reasoning_effort"), run_budget=a.run_budget,
                            total_budget=a.total_budget, tag=os.path.basename(a.out or a.run),
                            service_tier=ra.get("service_tier"),
                            max_completion_tokens=ra.get("max_tokens") or 4000)
    pricers.raw = []
    path = a.out or a.run.replace(".json", f"_resume_{a.mode}.json")
    events = []  # one deviation test per event, saved after each so a budget stop keeps them

    def summarise():
        agg = np.concatenate([np.asarray(e["per_event"]["rival_aggression"]) for e in events])  # (events, lags, S)
        size = np.concatenate([np.asarray(e["per_event"]["deviation_size"]) for e in events])  # (events, S)
        gain = np.concatenate([np.asarray(e["per_event"]["gain"]) for e in events])            # (events, S)
        per_session = {}
        for k, s in enumerate(sessions):
            u = agg[:, :, k] / size[:, k][:, None]
            per_session[s] = {"size": size[:, k].tolist(), "per_unit_by_lag": np.round(u.mean(0), 3).tolist(),
                              "lag1_per_event": np.round(u[:, 1], 3).tolist(), "gain": gain[:, k].tolist(),
                              "start_index": float(last_idx[k].mean())}
        test = {"mode": a.mode, "n_events": int(agg.shape[0] * agg.shape[2]), "horizon": a.horizon,
                "per_event": {"rival_aggression": agg.tolist(), "deviation_size": size.tolist(), "gain": gain.tolist()}}
        out = {"run": os.path.relpath(a.run, ROOT), "sessions": sessions, "mode": a.mode, "objective": cfg.objective,
               "test": test, "per_session": per_session, "usage": backend.usage, "complete": len(events) == a.events}
        json.dump(out, open(path, "w"), indent=1)
        if pricers.raw:
            with gzip.open(path.replace(".json", "_raw.jsonl.gz"), "wt") as fh:
                for t, s_, i, seed, text in pricers.raw:
                    fh.write(json.dumps({"t": t, "s": sessions[s_], "i": i, "seed": seed, "text": text}) + "\n")
        return per_session

    for e in range(a.events):
        try:
            events.append(deviation_test(env, pricers, backend, events=1, gap=a.gap, horizon=a.horizon,
                                         mode=a.mode, cut=a.cut))
        except BudgetExceeded as err:
            print(f"stopped after {len(events)} events: {err}", flush=True)
            break
        ps = summarise()
        for s, r in ps.items():
            print(f"event {e + 1}: session {s}: per unit by lag {r['per_unit_by_lag']}", flush=True)
    if events:
        summarise()
    print("wrote", path)


if __name__ == "__main__":
    main()
