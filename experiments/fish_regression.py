"""On-path regression of Fish et al. (EC'26, Table 1) on our LLM pricing runs.

    p_{i,t} = a_{i,r} + gamma p_{i,t-1} + delta p_{-i,t-1} + e

with firm-by-run fixed effects, on the second half of each run. Fish et al. read
delta > 0 as a reward-punishment scheme. Prices are strategic complements in
this game, so a firm that only best-responds to the rival's last price also has
delta > 0; we compare delta, and the long-run response delta / (1 - gamma), with
the exact slope of the static best response at the run's prices.

Writes results/fish_regression.json and paper/numbers_fishreg.tex.

    python experiments/fish_regression.py
"""
from __future__ import annotations

import glob
import gzip
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
sys.path.insert(0, os.path.join(ROOT, "src"))
from kylecollusion.llm_pricing import (best_response_exact, best_response_secant,  # noqa: E402
                                       best_response_slope, parse_fish, parse_price, scaled_config)


def prices_of(path: str, d: dict) -> np.ndarray | None:
    """(S, T, 2) prices of the main run; rebuilt from the raw responses for runs
    saved before per-session prices were stored (same fallback as the runner)."""
    if "prices_sessions" in d:
        return np.asarray(d["prices_sessions"], float)
    raw = path[:-5] + "_raw.jsonl.gz"
    if not os.path.exists(raw):
        return None
    a = d["args"]
    S, T, k = a["sessions"], a["periods"], a.get("scale", 1.0)
    fish = a.get("style", "ours") == "fish"
    p = np.full((T, S, 2), np.nan)
    with gzip.open(raw, "rt") as fh:
        next(fh)
        for line in fh:
            r = json.loads(line)
            if r["t"] < T:
                x = parse_fish(r["text"])[0] if fish else parse_price(r["text"])[0]
                p[r["t"], r["s"], r["i"]] = np.nan if x is None else x
    for t in range(T):  # unreadable answers repeat the last price (1.5 x cost at the start)
        prev = p[t - 1] if t else np.full((S, 2), 1.5 * k)
        p[t] = np.where(np.isfinite(p[t]), p[t], prev)
    return np.clip(p, 0.0, 10.0 * k).transpose(1, 0, 2)


def regress(P: np.ndarray, burn: float = 0.5) -> dict:
    """Within (firm x run) OLS of p_t on own and rival p_{t-1}; SEs clustered by run."""
    S, T, _ = P.shape
    t0 = max(1, int(burn * T))
    ys, xs, groups = [], [], []
    for s in range(S):
        for i in range(2):
            y = P[s, t0:, i]
            X = np.stack([P[s, t0 - 1:T - 1, i], P[s, t0 - 1:T - 1, 1 - i]], 1)
            ys.append(y - y.mean())
            xs.append(X - X.mean(0))
            groups.append(np.full(len(y), s))
    y, X, g = np.concatenate(ys), np.concatenate(xs), np.concatenate(groups)
    XtX = X.T @ X
    if np.linalg.matrix_rank(XtX) < 2:
        return {"gamma": np.nan, "delta": np.nan, "delta_se": np.nan, "n": int(len(y))}
    b = np.linalg.solve(XtX, X.T @ y)
    e = y - X @ b
    meat = sum(np.outer(X[g == s].T @ e[g == s], X[g == s].T @ e[g == s]) for s in np.unique(g))
    V = np.linalg.solve(XtX, np.linalg.solve(XtX, meat).T)
    se = np.sqrt(np.diag(V))
    return {"gamma": float(b[0]), "gamma_se": float(se[0]), "delta": float(b[1]),
            "delta_se": float(se[1]), "long_run": float(b[1] / (1 - b[0])) if b[0] < 1 else np.nan,
            "n": int(len(y))}


def br_slope(price: float, scale: float) -> float:
    """Exact slope of the static best response at a rival price (scale-free)."""
    return best_response_slope(price, scaled_config(scale))


def benchmarks(P: np.ndarray, scale: float) -> dict:
    """What a rival that only best-responds shows, at the run's prices (sessions x firms):
    br_slope, the local slope at each firm's mean price over the scoring window (the
    comparison for the on-path coefficient); cut_slope, the change in its best response per
    unit of a 10% cut from the deviator's final price; brdev_slope, the same for the
    deviator's move to its best response. Deviation tests start from the final state, and
    the deviator is firm 1 (index 0)."""
    cfg = scaled_config(scale)
    S, T, _ = P.shape
    local = [br_slope(P[s, T // 2:, i].mean(), scale) for s in range(S) for i in (0, 1)]
    cut, brd = [], []
    for p0, p1 in P[:, -1, :]:
        cut.append(best_response_secant(p0, 0.9 * p0, cfg))
        b = best_response_exact(p1, cfg)
        brd.append(best_response_secant(p0, b, cfg) if abs(p0 - b) > 1e-4 * scale else best_response_slope(p0, cfg))
    return {"br_slope": float(np.mean(local)), "cut_slope": float(np.mean(cut)), "brdev_slope": float(np.mean(brd))}


def main():
    out = {}
    files = (sorted(glob.glob(os.path.join(ROOT, "results", "llm_bertrand", "*.json")))
             + sorted(glob.glob(os.path.join(ROOT, "results", "llm_fish", "*.json"))))
    for f in files:
        if os.path.getsize(f) == 0:
            continue
        d = json.load(open(f))
        if "args" not in d or d.get("condition") == "solo":  # skip re-tests and the competence screen
            continue
        P = prices_of(f, d)
        if P is None:
            continue
        r = regress(P)
        k = d["args"].get("scale", 1.0)
        r.update(benchmarks(P, k), clusters=int(P.shape[0]))
        out[os.path.basename(f)[:-5]] = r
        print(f"{os.path.basename(f)[:-5]:34s} gamma {r['gamma']:.3f} delta {r['delta']:.3f} "
              f"({r['delta_se']:.3f}) long-run {r.get('long_run', np.nan):.3f} BR slope {r['br_slope']:.3f} "
              f"cut {r['cut_slope']:.3f} BR dev {r['brdev_slope']:.3f}")
    json.dump(out, open(os.path.join(ROOT, "results", "fish_regression.json"), "w"), indent=1)
    names = {"mistral7b_duopoly_k1": "Mistral", "mistral7b_myopic_k1": "MistralMyopic",
             "qwen3_8b_duopoly_k1": "Qwen", "qwen3_8b_myopic_k1": "QwenMyopic",
             "qwen3_8b_think_duopoly_k1": "QwenThink", "qwen3_8b_think_myopic_k1": "QwenThinkMyopic"}
    with open(os.path.join(ROOT, "paper", "numbers_fishreg.tex"), "w") as fh:
        fh.write("% generated by experiments/fish_regression.py\n")
        for key, nm in names.items():
            r = out.get(key)
            for m, v in (("", None if r is None else f"{r['delta']:.2f}"),
                         ("SE", None if r is None else f"{r['delta_se']:.2f}")):
                fh.write(f"\\newcommand{{\\FishReg{nm}{m}}}{{{'--' if v is None else v}}}\n")
        # benchmarks of our pricing runs: what a rival that only best-responds follows of the
        # best-response deviation (duopolies), and the local slope across all of these runs
        duo = [out[k]["brdev_slope"] for k in ("mistral7b_duopoly_k1", "qwen3_8b_duopoly_k1",
                                               "qwen3_8b_think_duopoly_k1") if k in out]
        ours = [r["br_slope"] for k, r in out.items() if k.startswith(("mistral7b_", "qwen3_8b_"))]
        if duo:
            fh.write(f"\\newcommand{{\\FishRegBRDevRange}}{{${min(duo):.2f}$--${max(duo):.2f}$}}\n")
        if ours:
            fh.write(f"\\newcommand{{\\FishRegBRRange}}{{${min(ours):.2f}$--${max(ours):.2f}$}}\n")


if __name__ == "__main__":
    main()
