"""GRPO-trained LLM pricing agents (kylecollusion.grpo with game="pricing") audited with the
in-context pricing protocol (experiments/grpo_pricing_audit.sh): the price index and profit
at prices x1 and x10, the rival's response to a best-response deviation and to a visible 10%
cut against what a rival that only best-responds would follow (exact, at each session's
prices), and the training curves. Reads results/grpo/pricing_*/log.jsonl and
results/grpo_pricing_audit/*.json; writes results/grpo_pricing_summary.json and
paper/numbers_grpopricing.tex ("--" where a run is missing).

    python experiments/grpo_pricing_summary.py
"""
from __future__ import annotations

import glob
import json
import os
import sys

import numpy as np
from scipy import stats

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.join(ROOT, "experiments"))
from fish_regression import benchmarks  # noqa: E402

MIN_CUT = 0.1  # smallest mean deviation (index units) for a response per unit
# run directory -> (model, discount factor of the credit, macro key)
RUNS = {"pricing_0p6b_g09": ("Qwen3-0.6B", 0.9, "SmallFwd"), "pricing_0p6b_g0": ("Qwen3-0.6B", 0.0, "SmallMyo"),
        "pricing_1p7b_g09": ("Qwen3-1.7B", 0.9, "MidFwd"), "pricing_1p7b_g0": ("Qwen3-1.7B", 0.0, "MidMyo")}
FIELDS = ("Idx", "IdxCI", "Prof", "ProfIdx", "BRSize", "BRUnit", "BRBench", "Gain", "CutUnit", "CutBench",
          "CutGain", "TenIdx", "TenProf", "Anchor", "Markup", "TenMarkup", "TrainIdx", "TrainProf", "Iters")


def tci(a):
    a = np.asarray(a, float)
    return float(a.mean()), float(stats.t.ppf(0.975, len(a) - 1) * a.std(ddof=1) / np.sqrt(len(a)))


def latest_audit(run):
    fs = sorted(glob.glob(os.path.join(ROOT, "results", "grpo_pricing_audit", f"{run}_adapter_*_duopoly_k1.json")))
    fs = [f for f in fs if os.path.getsize(f) > 0]
    return fs[-1] if fs else None


def main():
    out, macros = {}, {}
    pi_ratio = 0.33749 / 0.22296  # monopoly / Nash profit
    for run, (model, gamma, key) in RUNS.items():
        r = {"model": model, "gamma": gamma}
        log = os.path.join(ROOT, "results", "grpo", run, "log.jsonl")
        if os.path.exists(log):
            L = [json.loads(x) for x in open(log) if x.strip()]
            tail = L[-10:]
            r["train"] = {"iterations": len(L), "final_index": float(np.mean([x["price_index_second_half"] for x in tail])),
                          "final_profit_over_nash": float(np.mean([x["profit_over_nash"] for x in tail])),
                          "curve": [[x["iteration"], x["price_index_second_half"], x["profit_over_nash"]] for x in L]}
        f1 = latest_audit(run)
        if f1:
            d = json.load(open(f1))
            P = np.asarray(d["prices_sessions"], float)
            k = d["args"]["scale"]
            bench = benchmarks(P, k)
            idx = tci(d["per_session"]["index"])
            prof = tci(d["per_session"]["profit_over_nash"])
            half = P[:, P.shape[1] // 2:]  # markup over cost in the second half of each session
            r["markup"] = float(half.mean() - d["bench"]["cost"])
            r.update(audit=os.path.relpath(f1, ROOT), index=idx, profit_over_nash=prof,
                     profit_index=((prof[0] - 1) / (pi_ratio - 1), prof[1] / (pi_ratio - 1)), bench=bench,
                     parse_fail=d["parse_fail_rate"])
            dv = d.get("deviation")
            if dv:
                r["br"] = {"size": dv["deviation_size"], "gain": (dv["cum_gain_dev"], dv["cum_gain_dev_ci95"])}
                if dv["deviation_size"] >= MIN_CUT:
                    r["br"]["per_unit"] = (dv["rival_aggression"][1] / dv["deviation_size"],
                                           dv["rival_aggression_ci95"][1] / dv["deviation_size"])
                else:
                    r["br"]["raw"] = (dv["rival_aggression"][1], dv["rival_aggression_ci95"][1])
            dc = d.get("deviation_cut")
            if dc and dc["deviation_size"] >= MIN_CUT:
                r["cut"] = {"per_unit": (dc["rival_aggression"][1] / dc["deviation_size"],
                                         dc["rival_aggression_ci95"][1] / dc["deviation_size"]),
                            "by_lag": [x / dc["deviation_size"] for x in dc["rival_aggression"]],
                            "gain": (dc["cum_gain_dev"], dc["cum_gain_dev_ci95"])}
            f10 = f1.replace("_duopoly_k1.json", "_duopoly_k10.json")
            if os.path.exists(f10) and os.path.getsize(f10) > 0:
                d10 = json.load(open(f10))
                P10 = np.asarray(d10["prices_sessions"], float)
                r["markup_k10"] = float(P10[:, P10.shape[1] // 2:].mean() - d10["bench"]["cost"])
                r["index_k10"] = tci(d10["per_session"]["index"])
                r["profit_k10"] = tci(d10["per_session"]["profit_over_nash"])
                r["anchoring"] = bool(abs(idx[0] - r["index_k10"][0]) > np.hypot(idx[1], r["index_k10"][1]))
        out[run] = r
        m = {f: "--" for f in FIELDS}
        if "train" in r:
            m.update(TrainIdx=f"{r['train']['final_index']:.2f}", TrainProf=f"{r['train']['final_profit_over_nash']:.2f}",
                     Iters=str(r["train"]["iterations"]))
        if "index" in r:
            m.update(Idx=f"{r['index'][0]:.2f}", IdxCI=f"{r['index'][1]:.2f}", Prof=f"{r['profit_over_nash'][0]:.2f}",
                     ProfIdx=f"{r['profit_index'][0]:.2f}", BRBench=f"{r['bench']['brdev_slope']:.2f}",
                     CutBench=f"{r['bench']['cut_slope']:.2f}")
        if "br" in r:
            m.update(BRSize=f"{r['br']['size']:.2f}", Gain=f"{r['br']['gain'][0]:+.2f}\\pm{r['br']['gain'][1]:.2f}")
            if "per_unit" in r["br"]:
                m["BRUnit"] = f"{r['br']['per_unit'][0]:.2f}\\pm{r['br']['per_unit'][1]:.2f}"
        if "cut" in r:
            m.update(CutUnit=f"{r['cut']['per_unit'][0]:.2f}\\pm{r['cut']['per_unit'][1]:.2f}",
                     CutGain=f"{r['cut']['gain'][0]:+.2f}\\pm{r['cut']['gain'][1]:.2f}")
        if "index_k10" in r:
            m.update(TenIdx=f"{r['index_k10'][0]:.2f}", Anchor="yes" if r["anchoring"] else "no",
                     TenMarkup=f"{r['markup_k10']:.2f}", TenProf=f"{r['profit_k10'][0]:.2f}")
        if "markup" in r:
            m["Markup"] = f"{r['markup']:.2f}"
        for f, v in m.items():
            macros[f"GrpoPr{key}{f}"] = v[1:] if v.startswith("-0.00") else v  # no "-0.00"
    json.dump(out, open(os.path.join(ROOT, "results", "grpo_pricing_summary.json"), "w"), indent=1)
    os.makedirs(os.path.join(ROOT, "paper"), exist_ok=True)
    with open(os.path.join(ROOT, "paper", "numbers_grpopricing.tex"), "w") as fh:
        fh.write("% generated by experiments/grpo_pricing_summary.py\n")
        for k, v in macros.items():
            fh.write(f"\\newcommand{{\\{k}}}{{{v}}}\n")
    print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk != "train"} for k, v in out.items()}, indent=1,
                     default=str)[:3000])


if __name__ == "__main__":
    main()
