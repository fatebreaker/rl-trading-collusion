"""Cross-check the trace judge with a stronger model on the 200-item validation
sample (results/annotation_items.json), using the same instructions as
experiments/trace_judge.py. Writes results/annotation_ai_<tag>.json in the
format of the blind Claude annotation (a list of {id, impact, rival_flow, ...}).

Usage: OPENAI_API_KEY=... python experiments/annotation_ai_judge.py
       [--model gpt-5.5-2026-04-23] [--effort low] [--tier flex] [--tag gpt55]
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

import trace_judge as tj  # noqa: E402
from kylecollusion.llm_traders import OpenAIBackend  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gpt-5.5-2026-04-23")
    ap.add_argument("--effort", default="low")
    ap.add_argument("--tier", default="flex")
    ap.add_argument("--tag", default="gpt55")
    ap.add_argument("--run-budget", type=float, default=4.0)
    ap.add_argument("--total-budget", type=float, default=140.0)
    a = ap.parse_args(argv)
    items = json.load(open(os.path.join(ROOT, "results", "annotation_items.json")))
    backend = OpenAIBackend(a.model, reasoning_effort=a.effort, max_completion_tokens=1000,
                            run_budget=a.run_budget, total_budget=a.total_budget,
                            ledger=os.path.join(ROOT, "results", "openai_spend.jsonl"),
                            tag=f"annotation_{a.tag}", service_tier=a.tier)
    convs = [[{"role": "system", "content": tj.SYSTEM},
              {"role": "user", "content": "Text:\n<<<\n" + tj.strip(it["text"]) + "\n>>>"}]
             for it in items]
    outs = []  # small batches keep the backend's worst-case budget check meaningful
    for i in range(0, len(convs), 25):
        outs += backend.generate(convs[i:i + 25], [0] * len(convs[i:i + 25]), 0.0, 1000)
    rows, failed = [], 0
    for it, o in zip(items, outs):
        lab = tj.parse(o)
        if lab is None:
            failed += 1
        rows.append({"id": it["id"], **(lab or {k: None for k in tj.LABELS})})
    out = os.path.join(ROOT, "results", f"annotation_ai_{a.tag}.json")
    json.dump(rows, open(out, "w"), indent=1)
    print("wrote", out, "failed", failed, "usage", backend.usage)


if __name__ == "__main__":
    main()
