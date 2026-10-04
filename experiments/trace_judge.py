"""LLM-judge coding of trader notes and reasoning traces (complements the
keyword coding in trace_analysis.py).

A random sample of main-run responses per run is labelled by a judge model
with six yes/no questions about what the response reasons about. The
instructed trigger strategy is included as a positive control for the
cooperation and punishment labels.

Usage: OPENAI_API_KEY=... python experiments/trace_judge.py [--per-run 150]
       [--model gpt-5.4-mini-2026-03-17] [--run-budget 5]
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

import trace_analysis as ta  # noqa: E402
from kylecollusion.llm_traders import OpenAIBackend  # noqa: E402

LABELS = {
    "impact": "The trader reasons that its own order moves the price (price impact), "
              "e.g. that a larger order pushes the price toward V or reduces profit per unit.",
    "rival_flow": "The trader treats another INFORMED trader as a source of order flow or "
                  "price pressure, e.g. 'the other informed trader will also buy, so the "
                  "price will be higher, so I buy less'.",
    "rival_infer": "The trader tries to infer the other informed trader's past orders, "
                   "behaviour or strategy from the history.",
    "coop": "The trader considers cooperating or coordinating with another informed trader, "
            "e.g. both trading less for mutual benefit, sharing profit, or keeping an agreement.",
    "punish": "The trader considers punishing or retaliating against another informed trader, "
              "considers being punished by it, or reacts to it having broken an agreement or "
              "traded too aggressively.",
    "half": "The trader explicitly scales down its order because another informed trader is "
            "present, e.g. trading half of what it would trade alone.",
}

SYSTEM = (
    "You annotate the written reasoning of an automated trading agent in a market where "
    "informed traders privately learn an asset's value V, uninformed traders submit random "
    "orders, and a market maker sets the price from the total order flow. Only the "
    "informed traders know V; 'uninformed traders' and 'noise' do not count as informed "
    "traders. Answer each question about what the text below explicitly says or reasons "
    "about, not about what the agent might be doing implicitly.\n\nQuestions:\n"
    + "\n".join(f"- {k}: {v}" for k, v in LABELS.items())
    + "\n\nRespond with a single JSON object with these keys and true/false values, and "
    "nothing else: " + json.dumps({k: False for k in LABELS})
)


def strip(text: str, limit: int = 12000) -> str:
    """Long thinking traces are truncated in the middle to bound cost."""
    if len(text) <= limit:
        return text
    return text[: limit // 2] + "\n[...]\n" + text[-limit // 2:]


def parse(text: str) -> dict | None:
    m = re.search(r"\{[^{}]*\}", text or "", re.S)
    if not m:
        return None
    try:
        d = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    return {k: bool(d.get(k, False)) for k in LABELS}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-run", type=int, default=150)
    ap.add_argument("--model", default="gpt-5.4-mini-2026-03-17")
    ap.add_argument("--run-budget", type=float, default=5.0)
    ap.add_argument("--total-budget", type=float, default=100.0)
    ap.add_argument("--only", default=None, help="regex on run labels")
    ap.add_argument("--out", default=os.path.join(ta.ROOT, "results", "trace_judge.json"))
    a = ap.parse_args(argv)

    backend = OpenAIBackend(a.model, reasoning_effort="none", max_completion_tokens=200,
                            run_budget=a.run_budget, total_budget=a.total_budget,
                            ledger=os.path.join(ta.ROOT, "results", "openai_spend.jsonl"),
                            tag="trace_judge")
    old = json.load(open(a.out)) if os.path.exists(a.out) else {}
    for label, group, path in ta.RUNS + ta.model_runs():
        if a.only and not re.search(a.only, label):
            continue
        if not os.path.exists(os.path.join(ta.ROOT, "results", path)) or path in old:
            continue
        tx, src = ta.texts(path)
        rng = random.Random(0)
        idx = sorted(rng.sample(range(len(tx)), min(a.per_run, len(tx))))
        convs = [[{"role": "system", "content": SYSTEM},
                  {"role": "user", "content": "Text:\n<<<\n" + strip(tx[i]) + "\n>>>"}] for i in idx]
        outs = backend.generate(convs, [0] * len(convs), 0.0, 200)
        labs = [parse(o) for o in outs]
        ok = [l for l in labs if l is not None]
        rates = {k: sum(l[k] for l in ok) / max(len(ok), 1) for k in LABELS}
        old[path] = {"label": label, "group": group, "source": src, "n": len(ok),
                     "failed": len(labs) - len(ok), "rates": rates,
                     "items": [{"i": i, **(l or {})} for i, l in zip(idx, labs)]}
        print(f"{label:42s} n={len(ok):4d} " + " ".join(f"{k} {v:.2f}" for k, v in rates.items()),
              f"| spent {backend.usage['cost']:.2f}", flush=True)
        json.dump(old, open(a.out, "w"), indent=1)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
