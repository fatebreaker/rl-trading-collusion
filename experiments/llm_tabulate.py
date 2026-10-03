"""Tabulate LLM-trader runs (results/llm_pilot/*.json) as a markdown table.

Columns: aggregate intensity and its ratio to the frozen-lambda stage Nash
(for a solo trader: the optimum), per-trader intensity, profit relative to
Nash, the rival's lag-1 reaction to the best-response and shift deviations,
and, when raw paths were saved, the escalation slope: per-session OLS slope of
aggregate intensity across 20-period blocks (intensity units per 100 periods).
"""

from __future__ import annotations

import glob
import json
import math
import os
import sys

import numpy as np


def ci(a):
    a = np.asarray(a, float)
    a = a[np.isfinite(a)]
    if len(a) < 2:
        return float("nan"), float("nan")
    return a.mean(), 1.96 * a.std(ddof=1) / math.sqrt(len(a))


def escalation(d, block=20):
    if "log" not in d:
        return float("nan"), float("nan")
    x = np.asarray(d["log"]["x"]).sum(-1)  # (T, S)
    v = np.asarray(d["log"]["v"])
    T = x.shape[0]
    b = []
    for s in range(0, T - block + 1, block):
        xs, vs = x[s:s + block], v[s:s + block]
        vc = vs - vs.mean(0)
        b.append(((xs - xs.mean(0)) * vc).mean(0) / np.maximum((vc**2).mean(0), 1e-12))
    b = np.array(b)  # (blocks, S)
    t = np.arange(len(b)) * block / 100.0
    tc = t - t.mean()
    slopes = (tc[:, None] * (b - b.mean(0))).sum(0) / (tc**2).sum()
    return ci(slopes)


def fmt(m, c, nd=2):
    if not np.isfinite(m):
        return "-"
    return f"{m:.{nd}f} ± {c:.{nd}f}" if np.isfinite(c) else f"{m:.{nd}f}"


def row(path):
    d = json.load(open(path))
    s, fb = d["summary"], d["bench_fixed"]
    I = d["market"]["n_informed"]
    agg = s["agg_intensity"]
    per = d["per_session"]
    ratio = ci(np.asarray(per["agg_intensity"]) / fb["agg_nash"])
    prof = ci(np.asarray(per["profit"]) / fb["profit_nash"])
    cells = {
        "run": os.path.basename(path)[:-5],
        "model": d["model"].split("/")[-1] + (" +LoRA" if d["args"].get("lora") else ""),
        "cond": d["condition"],
        "sigma_u": d["market"]["sigma_u"],
        "agg beta": fmt(agg["mean"], agg["ci95"]),
        "/ Nash": fmt(*ratio),
        "per trader": f"{agg['mean'] / I:.2f}",
        "profit / Nash": fmt(*prof),
        "escalation /100p": fmt(*escalation(d)),
        "parse fail": f"{d['parse_fail_rate']:.3f}",
    }
    for key, name in (("deviation", "rival lag1 (BR)"), ("deviation_shift", "rival lag1 (shift)")):
        if key in d:
            dv = d[key]
            cells[name] = fmt(dv["d_beta_rival"][1], dv["d_beta_rival_ci95"][1], 3)
            cells[name.replace("rival lag1", "dev gain")] = fmt(dv["cum_gain_dev"], dv["cum_gain_dev_ci95"])
        else:
            cells[name] = "-"
            cells[name.replace("rival lag1", "dev gain")] = "-"
    return cells


def main(argv=None):
    pattern = (argv or sys.argv[1:] or ["results/llm_pilot/*.json"])[0]
    rows = [row(p) for p in sorted(glob.glob(pattern))]
    if not rows:
        print("no runs")
        return
    cols = list(rows[0])
    for r in rows:
        cols += [c for c in r if c not in cols]
    print("| " + " | ".join(cols) + " |")
    print("|" + "---|" * len(cols))
    for r in rows:
        print("| " + " | ".join(str(r.get(c, "-")) for c in cols) + " |")


if __name__ == "__main__":
    main()
