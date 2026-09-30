"""Per-session outcome metrics, scored against the Nash / collusive benchmarks."""

from __future__ import annotations

import math

import numpy as np

from .theory import Benchmarks, collusion_index


def session_metrics(log: dict[str, np.ndarray], bench: Benchmarks) -> dict[str, np.ndarray]:
    """log arrays are (T, S) except x and profit which are (T, S, I)."""
    v, p = log["v"], log["p"]
    x_agg = log["x"].sum(-1)

    var_v = v.var(0)
    agg = ((x_agg - x_agg.mean(0)) * (v - v.mean(0))).mean(0) / var_v
    vc, pc = v - v.mean(0), p - p.mean(0)
    info = (vc * pc).mean(0) ** 2 / (var_v * pc.var(0) + 1e-12)

    profit = log["profit"].mean(axis=(0, 2))
    xc = x_agg - x_agg.mean(0)
    # Share of aggregate informed order variance explained by v. Below 1 means
    # orders also react to noise in the memory state (unconverged or mixed play).
    order_r2 = (xc * vc).mean(0) ** 2 / (var_v * xc.var(0) + 1e-12)
    return {
        "profit": profit,
        "agg_intensity": agg,
        "lam": log["lam"].mean(0),
        "informativeness": info,
        "mispricing_mse": ((v - p) ** 2).mean(0),
        "order_r2": order_r2,
        "delta_profit": collusion_index(profit, bench.profit_nash, bench.profit_coll),
        "delta_intensity": collusion_index(agg, bench.agg_nash, bench.agg_coll),
        "delta_info": collusion_index(info, bench.info_nash, bench.info_coll),
    }


def summarize(per_session: dict[str, np.ndarray]) -> dict[str, dict[str, float]]:
    out = {}
    for k, arr in per_session.items():
        arr = np.asarray(arr, dtype=float)
        arr = arr[np.isfinite(arr)]
        n = len(arr)
        if n == 0:
            out[k] = {"mean": float("nan"), "ci95": float("nan"), "n": 0}
            continue
        se = arr.std(ddof=1) / math.sqrt(n) if n > 1 else float("nan")
        out[k] = {
            "mean": float(arr.mean()),
            "median": float(np.median(arr)),
            "ci95": float(1.96 * se),
            "n": n,
        }
    return out
