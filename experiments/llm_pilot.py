"""Pilot: language-model informed traders in the repeated Kyle market.

The market maker's price impact is frozen at the Nash lambda (mm_fixed), so
the stage game has exact benchmarks over a horizon of a few hundred periods
(with the adaptive market maker, lambda would barely move in that time).
Outcomes are scored against both the frozen-lambda benchmarks (stage Nash vs
joint optimum at that lambda) and the usual Kyle benchmarks.

Conditions:
  duopoly    two traders, long-run objective, see own trades and the order flow
  monitor    duopoly with perfect monitoring (history shows the rival's orders)
  myopic     duopoly told to maximise the current period only
  solo       one trader, no rival: any shortfall from 1/(2 lambda) is bias

Example:
  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=2 PYTHONPATH=src \\
    python experiments/llm_pilot.py --condition duopoly --model Qwen/Qwen3-8B \\
    --sessions 40 --periods 200 --out results/llm_pilot/qwen3_8b_duopoly.json
"""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import asdict

import numpy as np

from kylecollusion.llm_traders import (
    LLMTraderConfig, LLMTraders, OpenAIBackend, VLLMBackend, deviation_test,
    fixed_lambda_benchmarks,
)
from kylecollusion.market import KyleMarket, MarketConfig
from kylecollusion.metrics import session_metrics, summarize
from kylecollusion.theory import collusion_index

CONDITIONS = {
    "duopoly": dict(n_informed=2, objective="long", show_rival=False),
    "monitor": dict(n_informed=2, objective="long", show_rival=True),
    "myopic": dict(n_informed=2, objective="myopic", show_rival=False),
    "solo": dict(n_informed=1, objective="long", show_rival=False),
    # framing x presence: identical prompt, with and without a real rival
    "duopoly_vague": dict(n_informed=2, objective="long", show_rival=False, rival_info="vague"),
    "solo_vague": dict(n_informed=1, objective="long", show_rival=False, rival_info="vague"),
    # positive control: an explicitly instructed trigger strategy under perfect
    # monitoring; the deviation test must detect it
    "punisher": dict(n_informed=2, objective="long", show_rival=True, instructions=(
        "Rule you must follow: you and the other trader have agreed that each of you orders "
        "0.35 x V every period. Look at the other trader's orders in the last 2 periods of "
        "the history. If in either period the other trader's order was larger in absolute "
        "size than 0.35 x |V| of that period plus 0.2, the other trader broke the agreement: "
        "punish by ordering 1.0 x V this period. Otherwise order 0.35 x V.")),
}


def intensity(x, v):
    """Per-session OLS slope of order on value: x (T, S[, I]), v (T, S)."""
    vc = v - v.mean(0)
    if x.ndim == 3:
        xc = x - x.mean(0)
        return (xc * vc[..., None]).mean(0) / (vc**2).mean(0)[:, None]
    xc = x - x.mean(0)
    return (xc * vc).mean(0) / (vc**2).mean(0)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--condition", choices=sorted(CONDITIONS), default="duopoly")
    ap.add_argument("--backend", choices=["vllm", "openai"], default="vllm")
    ap.add_argument("--model", default="Qwen/Qwen3-8B")
    ap.add_argument("--sessions", type=int, default=40)
    ap.add_argument("--periods", type=int, default=200)
    ap.add_argument("--burn", type=float, default=0.5, help="share of periods dropped before scoring")
    ap.add_argument("--history", type=int, default=30)
    ap.add_argument("--sigma-u", type=float, default=1.0, help="noise-trading sd (sigma_v = 1)")
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--no-notes", action="store_true")
    ap.add_argument("--prompt-variant", choices=["a", "b"], default="a")
    ap.add_argument("--dev-events", type=int, default=3)
    ap.add_argument("--dev-gap", type=int, default=5)
    ap.add_argument("--dev-horizon", type=int, default=8)
    ap.add_argument("--dev-shift", type=float, default=2.0,
                    help="size of the 'shift' deviation in noise sd (0 skips it)")
    ap.add_argument("--gpu-mem", type=float, default=0.85)
    ap.add_argument("--dtype", default="auto", help="'half' on RTX 6000 (no bf16)")
    ap.add_argument("--attention-backend", default=None, help="'triton_attn' on RTX 6000")
    ap.add_argument("--lora", default=None, help="trained LoRA adapter directory")
    ap.add_argument("--max-tokens", type=int, default=None)
    ap.add_argument("--thinking", action="store_true", help="Qwen3 thinking mode (slow)")
    ap.add_argument("--reasoning-effort", default=None, help="OpenAI reasoning models")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)

    cond = CONDITIONS[a.condition]
    mcfg = MarketConfig(n_informed=cond["n_informed"], mm_fixed=True, memory="flow",
                        sigma_u=a.sigma_u)
    env = KyleMarket(mcfg, a.sessions, seed=a.seed)
    env.reset()
    tcfg = LLMTraderConfig(history=a.history, objective=cond["objective"],
                           show_rival=cond["show_rival"], notes=not a.no_notes,
                           rival_info=cond.get("rival_info", "explicit"),
                           prompt_variant=a.prompt_variant,
                           instructions=cond.get("instructions", ""),
                           temperature=a.temperature,
                           max_tokens=a.max_tokens or (4096 if a.thinking else 300))
    traders = LLMTraders(env, tcfg, seed=a.seed)
    if a.backend == "vllm":
        backend = VLLMBackend(a.model, gpu_memory_utilization=a.gpu_mem,
                              enable_thinking=a.thinking, dtype=a.dtype,
                              attention_backend=a.attention_backend, lora_path=a.lora,
                              max_model_len=16384 if a.thinking else 8192)
    else:
        backend = OpenAIBackend(a.model, reasoning_effort=a.reasoning_effort)

    T, S, I = a.periods, env.S, env.I
    log = {k: np.zeros((T, S)) for k in ("v", "p", "lam", "y")}
    log["x"] = np.zeros((T, S, I))
    log["profit"] = np.zeros((T, S, I))
    transcript = []
    t0 = time.time()
    for t in range(T):
        x = traders.act(env, backend)
        transcript.append([traders.last_text[0][i] for i in range(I)])
        profit, _, info = env.step_orders(x)
        traders.record(info, profit)
        for k in ("v", "p", "lam", "y"):
            log[k][t] = info[k]
        log["x"][t], log["profit"][t] = x, profit
        if (t + 1) % 20 == 0:
            b = intensity(log["x"][max(0, t - 19):t + 1].sum(-1), log["v"][max(0, t - 19):t + 1])
            print(f"period {t + 1}: agg intensity (last 20) {np.nanmean(b):.3f}, "
                  f"parse failures {traders.n_fail}/{traders.n_calls}, "
                  f"{time.time() - t0:.0f}s", flush=True)
    run_seconds = time.time() - t0

    lo = int(a.burn * T)
    tail = {k: v[lo:] for k, v in log.items()}
    per = session_metrics(tail, env.bench)
    fb = fixed_lambda_benchmarks(I, env.bench.lam_nash, mcfg.sigma_v)
    agg = per["agg_intensity"]
    per["delta_intensity_fixed"] = collusion_index(agg, fb["agg_nash"], fb["agg_coll"])
    per["delta_profit_fixed"] = collusion_index(per["profit"], fb["profit_nash"], fb["profit_coll"])
    per["intensity_over_nash_fixed"] = agg / fb["agg_nash"]
    per["intensity_over_joint_fixed"] = agg / fb["agg_coll"]
    beta_i = intensity(tail["x"], tail["v"])  # (S, I)
    per["trader_intensity_sd"] = beta_i.std(1) if I > 1 else np.zeros(S)

    # Intensity path in blocks of 20 periods (learning dynamics).
    blocks = [float(np.nanmean(intensity(log["x"][b:b + 20].sum(-1), log["v"][b:b + 20])))
              for b in range(0, T, 20)]

    res = {
        "condition": a.condition, "model": a.model, "backend": a.backend, "seed": a.seed,
        "args": vars(a), "market": asdict(mcfg), "trader": asdict(tcfg),
        "bench": env.bench.as_dict(), "bench_fixed": fb,
        "summary": summarize(per), "per_session": {k: np.asarray(v).tolist() for k, v in per.items()},
        "intensity_blocks": blocks,
        "parse_fail_rate": traders.n_fail / max(traders.n_calls, 1),
        "run_seconds": run_seconds,
        "transcript_session0": transcript,
        # raw paths for session-level statistics: (T, S[, I])
        "log": {k: np.round(log[k], 5).tolist() for k in ("v", "p", "x")},
    }
    if I > 1 and a.dev_events > 0:
        t1 = time.time()
        res["deviation"] = deviation_test(env, traders, backend, events=a.dev_events,
                                          gap=a.dev_gap, horizon=a.dev_horizon)
        if a.dev_shift > 0:
            res["deviation_shift"] = deviation_test(
                env, traders, backend, events=a.dev_events, gap=a.dev_gap,
                horizon=a.dev_horizon, mode="shift", scale=a.dev_shift)
        res["deviation_seconds"] = time.time() - t1
    if isinstance(backend, OpenAIBackend):
        res["usage"] = backend.usage

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w") as fh:
        json.dump(res, fh, indent=1)
    sm = res["summary"]
    print(json.dumps({k: sm[k]["mean"] for k in ("agg_intensity", "delta_intensity_fixed",
                                                 "delta_profit_fixed", "intensity_over_nash_fixed")}))
    for key in ("deviation", "deviation_shift"):
        if key in res:
            d = res[key]
            print(key, "rival d_beta:", np.round(d["d_beta_rival"], 3).tolist(),
                  "| cum gain dev:", round(d["cum_gain_dev"], 4), "+-", round(d["cum_gain_dev_ci95"], 4))


if __name__ == "__main__":
    main()
