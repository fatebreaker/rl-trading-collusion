"""Rebuild a run's main-period metrics from its raw responses (*_raw.jsonl.gz).

The market's value and noise draws do not depend on the orders (the price rule
is frozen), so replaying the market with the same seed and the logged orders
reproduces every period exactly. Deviation tests cannot be recovered.

Usage: python experiments/recover_from_raw.py results/llm_api/<run>_raw.jsonl.gz
"""

from __future__ import annotations

import gzip
import json
import sys

import numpy as np

from kylecollusion.llm_traders import fixed_lambda_benchmarks, parse_response
from kylecollusion.market import KyleMarket, MarketConfig
from kylecollusion.metrics import session_metrics, summarize
from kylecollusion.theory import collusion_index

CONDS = {"duopoly": 2, "monitor": 2, "myopic": 2, "solo": 1, "triopoly": 3}


def main(path):
    lines = [json.loads(l) for l in gzip.open(path, "rt")]
    head, rows = lines[0], lines[1:]
    a = head["args"]
    if a.get("adaptive_mm"):
        raise SystemExit("adaptive market maker: orders affect prices; replay not exact")
    I = CONDS[a["condition"]]
    S, T = a["sessions"], a["periods"]
    x = np.zeros((T, S, I))
    for r in rows:
        order, _ = parse_response(r["text"])
        x[r["t"], r["s"], r["i"]] = float(np.clip(order if order is not None else 0.0, -10, 10))
    env = KyleMarket(MarketConfig(n_informed=I, mm_fixed=True, memory="flow",
                                  sigma_u=a["sigma_u"]), S, seed=a["seed"])
    env.reset()
    log = {k: np.zeros((T, S)) for k in ("v", "p", "lam")}
    log["x"], log["profit"] = x, np.zeros((T, S, I))
    for t in range(T):
        pi, _, info = env.step_orders(x[t])
        log["v"][t], log["p"][t], log["lam"][t], log["profit"][t] = info["v"], info["p"], info["lam"], pi
    lo = T // 2
    per = session_metrics({k: v[lo:] for k, v in log.items()}, env.bench)
    fb = fixed_lambda_benchmarks(I, env.bench.lam_nash, 1.0)
    per["delta_intensity_fixed"] = collusion_index(per["agg_intensity"], fb["agg_nash"], fb["agg_coll"])
    per["delta_profit_fixed"] = collusion_index(per["profit"], fb["profit_nash"], fb["profit_coll"])
    out = path.replace("_raw.jsonl.gz", ".json")
    res = {"condition": a["condition"], "model": a["model"], "backend": a["backend"], "seed": a["seed"],
           "args": a, "market": {"n_informed": I, "sigma_u": a["sigma_u"], "mm_fixed": True},
           "bench": env.bench.as_dict(), "bench_fixed": fb, "summary": summarize(per),
           "per_session": {k: np.asarray(v).tolist() for k, v in per.items()},
           "parse_fail_rate": float(np.mean([parse_response(r["text"])[0] is None for r in rows])),
           "recovered_from_raw": True, "deviation_error": "run stopped by the budget cap during the deviation tests",
           "log": {k: np.round(log[k], 5).tolist() for k in ("v", "p", "x")}}
    json.dump(res, open(out, "w"), indent=1)
    sm = res["summary"]
    print(out, {k: round(sm[k]["mean"], 3) for k in ("agg_intensity", "delta_intensity_fixed")})


if __name__ == "__main__":
    main(sys.argv[1])
