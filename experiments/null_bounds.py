"""How strongly do the deviation tests rule out punishment?

Collects the rival's lag-1 response in every uninstructed deviation test
(best-response and shift; in-context, reasoning, API and RL-trained LLM traders)
and compares it with the instructed trigger strategy's response (the positive
control). Reports the number of significant positive responses against the
number expected by chance, the share of tests whose 95% upper bound lies below
the positive control's response, and an inverse-variance pooled estimate.

Writes results/null_bounds.json, results/deviation_tests.csv (every test, for
the supplementary material) and paper/numbers_nulls.tex.
"""
from __future__ import annotations

import csv
import glob
import json
import math
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DIRS = ["llm_pilot", "llm_api", "llm_models", "grpo_audit", "llm_talk"]
CONTROLS = ("punisher",)  # instructed trigger strategies


def tests():
    out = []
    for d in DIRS:
        for f in sorted(glob.glob(os.path.join(ROOT, "results", d, "*.json"))):
            name = os.path.basename(f)
            if name.startswith("dry_") or name.endswith("_raw.json"):
                continue
            try:
                r = json.load(open(f))
            except (json.JSONDecodeError, UnicodeDecodeError):
                continue
            if not isinstance(r, dict) or "market" not in r:
                continue
            control = any(c in r.get("condition", "") for c in CONTROLS)
            # profit relative to the stage Nash profit: a cartel-like state has something to protect
            fb = r["bench_fixed"] if r["market"].get("mm_fixed", True) else r["bench"]
            prof = r.get("per_session", {}).get("profit")
            pn = float(sum(prof) / len(prof) / fb["profit_nash"]) if prof else float("nan")
            for key in ("deviation", "deviation_shift"):
                t = r.get(key)
                if not t or len(t.get("d_beta_rival", [])) < 2:
                    continue
                m, c = t["d_beta_rival"][1], t["d_beta_rival_ci95"][1]
                if m != m or c != c:
                    continue
                # summed over the periods after the deviation (up to six); the half-width
                # adds the per-lag half-widths, an upper bound under any correlation
                H = min(6, len(t["d_beta_rival"]) - 1)
                cm = float(sum(t["d_beta_rival"][1:H + 1]))
                cc = float(sum(t["d_beta_rival_ci95"][1:H + 1]))
                out.append({"file": f"{d}/{name}", "test": key, "control": control,
                            "condition": r.get("condition"), "model": r.get("model", "").split("/")[-1],
                            "sigma_u": r["market"]["sigma_u"], "n_informed": r["market"].get("n_informed"),
                            "profit_over_nash": pn,
                            "n": t.get("n_events"),
                            "mean": m, "ci": c, "cum_mean": cm, "cum_ci": cc, "cum_lags": H,
                            "gain": t.get("cum_gain_dev"), "gain_ci": t.get("cum_gain_dev_ci95")})
    return out


def duplicate(t, files):
    """An earlier checkpoint of an RL run whose final checkpoint is also tested."""
    f = t["file"]
    if "_adapter_" not in f:
        return False
    run, step = f.split("_adapter_")[0], int(f.split("_adapter_")[1][:4])
    return any(g.startswith(run + "_adapter_") and int(g.split("_adapter_")[1][:4]) > step for g in files)


def structure(t):
    """Information structure of a test: perfect monitoring (rival's orders shown),
    or order flow only with a best-response or a visible shift deviation."""
    if t["condition"] in ("monitor", "talk_monitor"):
        return "monitor"
    return "flow_shift" if t["test"] == "deviation_shift" else "flow_br"


def random_effects(ts, floor):
    """DerSimonian-Laird pooled mean and 95% half-width."""
    se = [max(t["ci"] / 1.96, floor) for t in ts]
    y = [t["mean"] for t in ts]
    w = [1 / s ** 2 for s in se]
    fe = sum(wi * yi for wi, yi in zip(w, y)) / sum(w)
    q = sum(wi * (yi - fe) ** 2 for wi, yi in zip(w, y))
    c = sum(w) - sum(wi ** 2 for wi in w) / sum(w)
    tau2 = max(0.0, (q - (len(ts) - 1)) / c) if c > 0 else 0.0
    w2 = [1 / (s ** 2 + tau2) for s in se]
    m = sum(wi * yi for wi, yi in zip(w2, y)) / sum(w2)
    return m, 1.96 * math.sqrt(1 / sum(w2))


def write_table(null, groups, ctrl_eff, res, effect, re_all, cum):
    """paper/table_nulls.tex: the pooled null by information structure and robustness subset.
    Chance is charged only to tests with sampling variance; "below" counts tests whose 95%
    upper bound lies below the instructed traders' response with the rival's orders shown."""
    f3 = lambda x: f"{abs(x) if abs(x) < 5e-4 else x:.3f}"  # noqa: E731
    pm = lambda m, c: f"${f3(m)}\\pm{c:.3f}$"  # noqa: E731
    var = lambda ts: [t for t in ts if t["ci"] / 1.96 > 0.001]  # noqa: E731

    def cells(ts, pooled, mde=True, key=("mean", "ci")):
        m, c = key
        nv = len(var(ts))
        sig = sum(t[m] - t[c] > 0 for t in ts)
        below = sum(t[m] + t[c] < (effect if m == "mean" else cum[3]) for t in ts)
        md = sorted(2.8 * t[c] / 1.96 for t in ts)
        return (f"{len(ts)} & {nv} & {sig} ({0.025 * nv:.1f}) & {below} & {pm(*pooled)} & "
                + (f"{md[len(md) // 2]:.2f}" if mde else "--"))

    st = {g: [t for t in null if structure(t) == g] for g in ("monitor", "flow_br", "flow_shift")}
    files = {t["file"] for t in null}
    indep = [t for t in null if not duplicate(t, files)]
    above = [t for t in indep if t["profit_over_nash"] >= 1]
    L = ["\\begin{tabular}{@{}lcccccc@{}}", "\\toprule",
         "Tests & $n$ & with variance & Significant & Below instructed & Pooled response & Median MDE \\\\",
         "\\midrule"]
    for name, g, c in (("Rival's orders shown", "monitor", ctrl_eff.get("monitor")),
                       ("Order flow, best-response deviation", "flow_br", ctrl_eff.get("flow_br")),
                       ("Order flow, shift deviation", "flow_shift", ctrl_eff.get("flow_shift"))):
        L.append(f"{name} (instructed: {c:.2f}) & "
                 + cells(st[g], (groups[g]["re_pooled"], groups[g]["re_pooled_ci"])) + " \\\\")
    L.append("\\midrule")
    L.append("All & " + cells(null, re_all) + " \\\\")
    L.append("Each run once & " + cells(indep, (res["indep_pooled"], res["indep_pooled_ci"]), mde=False) + " \\\\")
    L.append("Agents earning at least Nash profit & "
             + cells(above, (res["above_pooled"], res["above_pooled_ci"]), mde=False) + " \\\\")
    m_cum, c_cum, cum_sig, ctrl_cum = cum
    L.append(f"Summed over six periods (instructed: {ctrl_cum:.2f}) & "
             + cells(null, (m_cum, c_cum), mde=False, key=("cum_mean", "cum_ci")) + " \\\\")
    L += ["\\bottomrule", "\\end{tabular}"]
    open(os.path.join(ROOT, "paper", "table_nulls.tex"), "w").write("\n".join(L) + "\n")


def main():
    T = tests()
    ctrl = [t for t in T if t["control"]]
    null = [t for t in T if not t["control"]]
    # the positive-control effect we must be able to rule out: the instructed
    # Qwen3-8B best-response test (rival response one period later)
    ref = next(t for t in ctrl if t["file"].endswith("qwen3_8b_punisher.json") and t["test"] == "deviation")
    effect = ref["mean"]
    sig_pos = [t for t in null if t["mean"] - t["ci"] > 0]
    sig_neg = [t for t in null if t["mean"] + t["ci"] < 0]
    below = [t for t in null if t["mean"] + t["ci"] < effect]
    half = [t for t in null if t["mean"] + t["ci"] < effect / 2]
    # inverse-variance pooling (fixed effect); tests with zero spread (deterministic
    # trained policies) get the smallest positive standard error observed
    ses = [t["ci"] / 1.96 for t in null]
    floor = min(s for s in ses if s > 0)
    w = [1 / max(s, floor) ** 2 for s in ses]
    pooled = sum(wi * t["mean"] for wi, t in zip(w, null)) / sum(w)
    pooled_se = math.sqrt(1 / sum(w))
    # without the zero-spread tests, so they cannot dominate
    nz = [(t, s) for t, s in zip(null, ses) if s > 0.001]
    w2 = [1 / s ** 2 for _, s in nz]
    pooled2 = sum(wi * t["mean"] for wi, (t, _) in zip(w2, nz)) / sum(w2)
    pooled2_se = math.sqrt(1 / sum(w2))
    res = {"n_tests": len(null), "n_controls": len(ctrl), "control_effect": effect,
           "control_tests": ctrl, "sig_positive": sig_pos, "sig_negative": len(sig_neg),
           "expected_false_positive": 0.025 * len(null),
           "upper_below_control": len(below), "upper_below_half_control": len(half),
           "max_upper": max(t["mean"] + t["ci"] for t in null),
           "median_upper": sorted(t["mean"] + t["ci"] for t in null)[len(null) // 2],
           "pooled": pooled, "pooled_ci": 1.96 * pooled_se,
           "pooled_noisy_only": pooled2, "pooled_noisy_only_ci": 1.96 * pooled2_se,
           "n_noisy_only": len(nz)}
    # by information structure, against the instructed control in the same structure
    ctrl_eff = {}
    for t in ctrl:
        if t["file"].endswith("qwen3_8b_punisher.json"):
            ctrl_eff["monitor" if t["test"] == "deviation" else "monitor_shift"] = t["mean"]
        if t["file"].endswith("qwen3_8b_punisher_flow.json"):
            ctrl_eff["flow_br" if t["test"] == "deviation" else "flow_shift"] = t["mean"]
    groups = {}
    for g in ("monitor", "flow_shift", "flow_br"):
        ts = [t for t in null if structure(t) == g]
        mde = sorted(2.8 * t["ci"] / 1.96 for t in ts)
        # pooled over tests with sampling spread (deterministic trained policies would dominate)
        m, c = random_effects([t for t in ts if t["ci"] / 1.96 > 0.001], floor)
        groups[g] = {"n": len(ts), "sig_pos": sum(t["mean"] - t["ci"] > 0 for t in ts),
                     "re_pooled": m, "re_pooled_ci": c, "median_mde80": mde[len(mde) // 2],
                     "below_ctrl": sum(t["mean"] + t["ci"] < effect for t in ts)}
    m_all, c_all = random_effects([t for t in null if t["ci"] / 1.96 > 0.001], floor)
    # cumulative response over the following periods
    cum = [{"mean": t["cum_mean"], "ci": t["cum_ci"]} for t in null if t["cum_ci"] / 1.96 > 0.001]
    m_cum, c_cum = random_effects(cum, floor)
    cum_sig = sum(t["cum_mean"] - t["cum_ci"] > 0 for t in null)
    ctrl_cum = next((t["cum_mean"] for t in ctrl if t["file"].endswith("qwen3_8b_punisher.json")
                     and t["test"] == "deviation"), float("nan"))
    res.update(by_structure=groups, control_by_structure=ctrl_eff, re_pooled=m_all, re_pooled_ci=c_all,
               cum_pooled=m_cum, cum_pooled_ci=c_cum, cum_sig=cum_sig, ctrl_cum=ctrl_cum)
    # robustness: one entry per independent run (no earlier RL checkpoints) and
    # chance counted only over tests with sampling variance
    files = {t["file"] for t in null}
    indep = [t for t in null if not duplicate(t, files)]
    var = [t for t in indep if t["ci"] / 1.96 > 0.001]
    m_ind, c_ind = random_effects(var, floor)
    res.update(indep_n=len(indep), indep_var_n=len(var), indep_sig=sum(t["mean"] - t["ci"] > 0 for t in indep),
               indep_expected=0.025 * len(var), indep_pooled=m_ind, indep_pooled_ci=c_ind)
    # tests where the agents earn at least the Nash profit, i.e. have a cartel to protect
    above = [t for t in indep if t["profit_over_nash"] >= 1]
    above_var = [t for t in above if t["ci"] / 1.96 > 0.001]
    m_ab, c_ab = random_effects(above_var, floor) if above_var else (float("nan"), float("nan"))
    res.update(above_n=len(above), above_var_n=len(above_var), above_runs=len({t["file"] for t in above}),
               above_sig=sum(t["mean"] - t["ci"] > 0 for t in above), above_pooled=m_ab, above_pooled_ci=c_ab,
               above_below_ctrl=sum(t["mean"] + t["ci"] < effect for t in above))
    json.dump({**res, "tests": null}, open(os.path.join(ROOT, "results", "null_bounds.json"), "w"), indent=1)
    cols = ["file", "model", "condition", "sigma_u", "n_informed", "profit_over_nash", "test", "gain", "gain_ci", "structure", "control", "n",
            "mean", "ci", "cum_mean", "cum_ci", "cum_lags", "significant", "earlier_checkpoint"]
    with open(os.path.join(ROOT, "results", "deviation_tests.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for t in null + ctrl:
            w.writerow({**{k: t.get(k) for k in cols}, "structure": structure(t),
                        "significant": t["mean"] - t["ci"] > 0, "earlier_checkpoint": duplicate(t, files)})
    macros = {"NullTests": len(null), "NullSigPos": len(sig_pos),
              "NullExpectedFP": f"{0.025 * sum(t['ci'] / 1.96 > 0.001 for t in null):.1f}",
              "NullBelowCtrl": len(below), "NullBelowHalf": len(half),
              "NullCtrlEffect": f"{effect:.2f}",
              "NullMedianUpper": f"{res['median_upper']:.3f}",
              "NullPooled": f"{abs(pooled2) if abs(pooled2) < 5e-4 else pooled2:.3f}", "NullPooledCI": f"{1.96 * pooled2_se:.3f}",
              "NullREPooled": f"{abs(m_all) if abs(m_all) < 5e-4 else m_all:.3f}", "NullREPooledCI": f"{c_all:.3f}"}
    for g, nm in (("monitor", "Mon"), ("flow_shift", "Shift"), ("flow_br", "Flow")):
        r = groups[g]
        macros.update({f"Null{nm}N": r["n"], f"Null{nm}Sig": r["sig_pos"],
                       f"Null{nm}Pooled": f"{r['re_pooled']:.3f}", f"Null{nm}PooledCI": f"{r['re_pooled_ci']:.3f}",
                       f"Null{nm}MDE": f"{r['median_mde80']:.2f}", f"Null{nm}BelowCtrl": r["below_ctrl"]})
    macros.update({"NullCumPooled": f"{abs(m_cum) if abs(m_cum) < 5e-4 else m_cum:.3f}",
                   "NullCumPooledCI": f"{c_cum:.3f}", "NullCumSig": cum_sig, "NullCtrlCum": f"{ctrl_cum:.2f}"})
    br = [t for t in null if t["test"] == "deviation" and t.get("gain") is not None]
    macros.update({"NullBRN": len(br), "NullGainPos": sum(t["gain"] > 0 for t in br),
                   "NullGainSig": sum(t["gain"] - t["gain_ci"] > 0 for t in br)})
    macros.update({"NullIndepN": len(indep), "NullIndepVarN": len(var), "NullIndepSig": res["indep_sig"],
                   "NullIndepExpected": f"{res['indep_expected']:.1f}",
                   "NullIndepPooled": f"{abs(m_ind) if abs(m_ind) < 5e-4 else m_ind:.3f}",
                   "NullIndepPooledCI": f"{c_ind:.3f}",
                   "NullAboveN": len(above), "NullAboveRuns": res["above_runs"], "NullAboveSig": res["above_sig"],
                   "NullAbovePooled": f"{abs(m_ab) if abs(m_ab) < 5e-4 else m_ab:.3f}",
                   "NullAbovePooledCI": f"{c_ab:.3f}", "NullAboveBelowCtrl": res["above_below_ctrl"]})
    macros.update({"NullCtrlFlow": f"{ctrl_eff.get('flow_br', float('nan')):.2f}",
                   "NullCtrlShift": f"{ctrl_eff.get('monitor_shift', float('nan')):.2f}"})
    write_table(null, groups, ctrl_eff, res, effect, (m_all, c_all), (m_cum, c_cum, cum_sig, ctrl_cum))
    with open(os.path.join(ROOT, "paper", "numbers_nulls.tex"), "w") as fh:
        fh.write("% generated by experiments/null_bounds.py -- do not edit\n")
        for k, v in macros.items():
            fh.write(f"\\newcommand{{\\{k}}}{{{v}}}\n")
    print(json.dumps({k: v for k, v in res.items() if k not in ("control_tests",)}, indent=1, default=str)[:3000])
    print("controls:", [(t["file"], t["test"], round(t["mean"], 3), round(t["ci"], 3)) for t in ctrl])
    print(macros)


if __name__ == "__main__":
    main()
