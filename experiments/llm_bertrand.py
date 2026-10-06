"""Run LLM pricing agents in the repeated logit-Bertrand duopoly and audit them.

Conditions:
  duopoly   two LLM firms maximising total profit
  myopic    two LLM firms maximising current-period profit (placebo)
  trigger   two LLM firms instructed to keep the joint-optimum price and punish
            undercutting (positive control)

Example:
  PYTHONPATH=src python experiments/llm_bertrand.py --condition duopoly \
      --model Qwen/Qwen3-8B --scale 1 --out results/llm_bertrand/qwen3_8b_duopoly_k1.json
"""
from __future__ import annotations

import argparse
import gzip
import json
import os
import time

import numpy as np

from kylecollusion.bertrand import logit_demand
from kylecollusion.llm_pricing import (
    LLMPricers, PricingConfig, PricingMarket, best_response, deviation_test, run_period,
    trigger_instructions,
)
from kylecollusion.llm_traders import BudgetExceeded, OpenAIBackend, VLLMBackend


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--condition", choices=["duopoly", "myopic", "trigger", "solo"], default="duopoly",
                    help="solo: the rival is scripted at the monopoly price, a competence check "
                         "(the agent should find its one-period best response)")
    ap.add_argument("--backend", choices=["vllm", "openai"], default="vllm")
    ap.add_argument("--model", required=True)
    ap.add_argument("--scale", type=float, default=1.0, help="currency unit (scale sweep)")
    ap.add_argument("--sessions", type=int, default=40)
    ap.add_argument("--periods", type=int, default=100)
    ap.add_argument("--burn", type=float, default=0.5, help="share of periods excluded from scoring")
    ap.add_argument("--history", type=int, default=None, help="periods of history shown (30; 100 in fish style)")
    ap.add_argument("--style", choices=["ours", "fish"], default="ours",
                    help="prompt: ours, or the template and prefixes of Fish et al. (EC'26)")
    ap.add_argument("--prefix", choices=["P1", "P2"], default="P1", help="fish style: prompt prefix")
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--no-notes", action="store_true")
    ap.add_argument("--max-tokens", type=int, default=None)
    ap.add_argument("--max-model-len", type=int, default=None)
    ap.add_argument("--thinking", action="store_true")
    ap.add_argument("--reasoning-effort", default=None)
    ap.add_argument("--service-tier", default=None)
    ap.add_argument("--run-budget", type=float, default=10.0)
    ap.add_argument("--total-budget", type=float, default=140.0)
    ap.add_argument("--dev-events", type=int, default=3)
    ap.add_argument("--dev-gap", type=int, default=5)
    ap.add_argument("--dev-horizon", type=int, default=8)
    ap.add_argument("--dev-cut", type=float, default=0.10, help="price cut for the visible deviation (0 skips it)")
    ap.add_argument("--gpu-mem", type=float, default=0.85)
    ap.add_argument("--dtype", default="auto")
    ap.add_argument("--attention-backend", default=None)
    ap.add_argument("--tp", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--save-raw", action="store_true")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)

    fish = a.style == "fish"
    cfg = PricingConfig(scale=a.scale, history=a.history or (100 if fish else 30), notes=not a.no_notes,
                        objective="myopic" if a.condition == "myopic" else "long",
                        temperature=a.temperature, style=a.style, prefix=a.prefix,
                        max_tokens=a.max_tokens or (8192 if a.thinking else (1200 if fish else 300)))
    env = PricingMarket(cfg, a.sessions)
    if a.condition == "trigger":
        cfg.instructions = trigger_instructions(env.bench)
    solo = a.condition == "solo"
    pricers = LLMPricers(env, cfg, seed=a.seed, active=[0] if solo else None)
    rival_fixed = env.bench["p_mono"]

    def policy(p):
        if solo:
            p[:, 1] = rival_fixed
        return p

    if a.save_raw:
        pricers.raw = []
    if a.backend == "vllm":
        backend = VLLMBackend(a.model, gpu_memory_utilization=a.gpu_mem, enable_thinking=a.thinking,
                              dtype=a.dtype, attention_backend=a.attention_backend,
                              max_model_len=a.max_model_len or (16384 if a.thinking else 8192),
                              tensor_parallel_size=a.tp)
    else:
        backend = OpenAIBackend(a.model, reasoning_effort=a.reasoning_effort, run_budget=a.run_budget,
                                total_budget=a.total_budget, tag=os.path.basename(a.out),
                                service_tier=a.service_tier)

    T, S = a.periods, a.sessions
    prices = np.zeros((T, S, 2))
    profit = np.zeros((T, S, 2))
    transcript = []
    t0 = time.time()
    for t in range(T):
        p, q, pi = run_period(env, pricers, backend, policy=policy)
        prices[t], profit[t] = np.clip(p, 0, 10 * env.bcfg.cost), pi
        if (t + 1) % 10 == 0:
            span = env.bench["p_mono"] - env.bench["p_nash"]
            idx = (prices[max(0, t - 9):t + 1].mean() - env.bench["p_nash"]) / span
            print(f"period {t + 1}: mean price index (last 10) {idx:.3f}, "
                  f"parse failures {pricers.n_fail}/{pricers.n_calls}, {time.time() - t0:.0f}s", flush=True)
    run_s = time.time() - t0
    if pricers.raw is not None:  # session 0's responses, period by period
        by_t = {}
        for t, s, i, seed, text in pricers.raw:
            if s == 0 and t < T:
                by_t.setdefault(t, ["", ""])[i] = text
        transcript = [by_t.get(t, ["", ""]) for t in range(T)]

    b0 = int(a.burn * T)
    span = env.bench["p_mono"] - env.bench["p_nash"]
    mean_p = prices[b0:].mean(0)                      # (S, 2)
    index = (mean_p.mean(1) - env.bench["p_nash"]) / span
    prof = profit[b0:].mean(0).mean(1) / env.bench["pi_nash"]
    res = {"condition": a.condition, "model": a.model, "backend": a.backend, "args": vars(a),
           "bench": env.bench, "system_prompt": pricers.system,
           "per_session": {"index": index.tolist(), "profit_over_nash": prof.tolist(),
                           "mean_price": mean_p.tolist()},
           "summary": {"index": float(index.mean()),
                       "index_ci95": float(1.96 * index.std(ddof=1) / np.sqrt(S)),
                       "profit_over_nash": float(prof.mean())},
           "parse_fail_rate": pricers.n_fail / max(pricers.n_calls, 1), "run_seconds": run_s,
           "price_path": prices.mean(1).tolist(),
           "prices_sessions": np.round(prices.transpose(1, 0, 2), 4).tolist(),  # (S, T, 2)
           "transcript_session0": [[x for x in row] for row in transcript[:T]]}
    if solo:
        br = float(best_response(np.array([rival_fixed]), env.bcfg)[0])
        own = prices[b0:, :, 0].mean(0)
        q_own = profit[b0:, :, 0].mean(0)
        pi_br = float(((br - env.bcfg.cost) * logit_demand(np.array([[br, rival_fixed]]), env.bcfg))[0, 0])
        res["solo"] = {"rival_price": rival_fixed, "best_response": br,
                       "gap": float(((own - br) / span).mean()),
                       "gap_ci95": float(1.96 * ((own - br) / span).std(ddof=1) / np.sqrt(S)),
                       "profit_share_of_br": float((q_own / pi_br).mean())}
        a.dev_events = 0  # nothing to deviate from
    if a.backend == "openai":
        res["usage"] = backend.usage

    def save():
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        json.dump(res, open(a.out, "w"), indent=1)
        if pricers.raw is not None:
            with gzip.open(a.out.replace(".json", "_raw.jsonl.gz"), "wt") as fh:
                fh.write(json.dumps({"system_prompt": pricers.system, "args": vars(a)}) + "\n")
                for t, s, i, seed, text in pricers.raw:
                    fh.write(json.dumps({"t": t, "s": s, "i": i, "seed": seed, "text": text}) + "\n")

    save()  # before the deviation tests, which can be long
    print(f"index {res['summary']['index']:.3f} +- {res['summary']['index_ci95']:.3f}, "
          f"profit/Nash {res['summary']['profit_over_nash']:.3f}", flush=True)
    try:
        if a.dev_events > 0:
            d = deviation_test(env, pricers, backend, events=a.dev_events, gap=a.dev_gap,
                               horizon=a.dev_horizon, mode="best_response")
            res["deviation"] = d
            print("deviation (best response) rival aggression:", np.round(d["rival_aggression"], 3).tolist(),
                  "gain", round(d["cum_gain_dev"], 3), flush=True)
            save()
            if a.dev_cut > 0:
                d = deviation_test(env, pricers, backend, events=a.dev_events, gap=a.dev_gap,
                                   horizon=a.dev_horizon, mode="cut", cut=a.dev_cut)
                res["deviation_cut"] = d
                print("deviation (cut) rival aggression:", np.round(d["rival_aggression"], 3).tolist(),
                      "gain", round(d["cum_gain_dev"], 3), flush=True)
    except BudgetExceeded as e:
        res["deviation_error"] = str(e)
    if a.backend == "openai":
        res["usage"] = backend.usage
    save()


if __name__ == "__main__":
    main()
