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
          "CutGain", "TenIdx", "TenProf", "Anchor", "Markup", "TenMarkup", "TrainIdx", "TrainProf", "Iters",
          "PlaceboUnit")


def tci(a):
    a = np.asarray(a, float)
    return float(a.mean()), float(stats.t.ppf(0.975, len(a) - 1) * a.std(ddof=1) / np.sqrt(len(a)))


def latest_audit(run):
    """The latest audited adapter whose answers parse (an adapter whose outputs degenerated is not audited)."""
    fs = sorted(glob.glob(os.path.join(ROOT, "results", "grpo_pricing_audit", f"{run}_adapter_*_duopoly_k1.json")))
    fs = [f for f in fs if os.path.getsize(f) > 0 and json.load(open(f)).get("parse_fail_rate", 0) < 0.5]
    return fs[-1] if fs else None


def rivals_table(out):
    """Appendix table: the GRPO pricing agents trained against scripted rivals, one row per run."""
    rows = [("pricing_tft_0p6b_g09", "tit-for-tat", "0.6B", "0.9", "0"), ("pricing_tft_0p6b_g09_s1", "tit-for-tat", "0.6B", "0.9", "1"),
            ("pricing_tft_0p6b_g0", "tit-for-tat", "0.6B", "0", "0"),
            ("pricing_cheat_0p6b_g09", "cheater", "0.6B", "0.9", "0"), ("pricing_cheat_0p6b_g09_s1", "cheater", "0.6B", "0.9", "1"),
            ("pricing_cheat_0p6b_g09_s2", "cheater", "0.6B", "0.9", "2"), ("pricing_cheat_0p6b_g0", "cheater", "0.6B", "0", "0"),
            ("pricing_cheat_1p7b_g09", "cheater", "1.7B", "0.9", "0"),
            ("pricing_cheatnash_0p6b_g09", "cheaper war", "0.6B", "0.9", "0"),
            ("pricing_cheatnash_ent05_0p6b_g09", "cheaper war$^*$", "0.6B", "0.9", "0"),
            ("pricing_cheatnash_0p6b_g0", "cheaper war", "0.6B", "0", "0"),
            ("pricing_cheat2self_0p6b_g09", "cheater, then self", "0.6B", "0.9", "0"),
            ("pricing_cheat_0p6b_g09_s3", "cheater", "0.6B", "0.9", "3"),
            ("pricing_tft_0p6b_g09_s2", "tit-for-tat", "0.6B", "0.9", "2"),
            ("pricing_cheat_1p7b_g09_s1", "cheater", "1.7B", "0.9", "1"),
            ("pricing_cheat_4b_g09", "cheater", "4B", "0.9", "0"), ("pricing_cheat_4b_g09_s1", "cheater", "4B", "0.9", "1"),
            ("pricing_cheatp03_0p6b_g09", "cheats less$^\\dagger$", "0.6B", "0.9", "0"),
            ("pricing_cheatw09_0p6b_g09", "war $+9\\%^\\S$", "0.6B", "0.9", "0"),
            ("pricing_cheatw30_0p6b_g09", "war $+30\\%^\\S$", "0.6B", "0.9", "0"),
            ("pricing_cheatw30_0p6b_g09_s1", "war $+30\\%^\\S$", "0.6B", "0.9", "1"),
            ("pricing_cheatw30_0p6b_g0", "war $+30\\%^\\S$", "0.6B", "0", "0")]

    def f(x, nd=2):
        return "--" if x is None or not np.isfinite(x) else f"${x:.{nd}f}$".replace("$-0.00$", "$0.00$")

    def pm(v):
        return "--" if v is None else f"${v[0]:.2f}_{{\\pm{v[1]:.2f}}}$".replace("$-0.00", "$0.00")

    L = ["\\begin{tabular}{@{}lllcccccccccc@{}}", "\\toprule",
         " & & & \\multicolumn{3}{c}{training} & \\multicolumn{7}{c}{paired with itself} \\\\",
         "\\cmidrule(lr){4-6}\\cmidrule(l){7-13}",
         "rival & model & $\\gamma$/seed & index & answer & war & index & $\\times10$ & cut & lag 8 & cut gain & rise & placebo \\\\",
         "\\midrule"]
    for run, rival, model, gamma, seed in rows:
        r = out.get(run, {})
        tr, pr = r.get("train", {}), r.get("pair", {})
        if not tr and not pr:
            continue
        lag8 = pr.get("cut_by_lag", [None])[-1] if "cut_by_lag" in pr else None
        cells = [rival, model, f"{gamma}/{seed}", f(tr.get("final_index")), f(tr.get("answer_index")),
                 "--" if tr.get("war_share") is None else f"${100 * tr['war_share']:.0f}\\%$",
                 f(pr["index"][0]) if "index" in pr else "--", f(pr["index_k10"][0]) if "index_k10" in pr else "--",
                 pm(pr.get("cut_per_unit")), f(lag8), pm(pr.get("cut_gain")),
                 pm(pr["Rise"]["per_unit"]) if "Rise" in pr else "--",
                 pm(pr["Placebo"]["per_unit"]) if "Placebo" in pr else "--"]
        L.append(" & ".join(cells) + " \\\\")
    L += ["\\bottomrule", "\\end{tabular}"]
    open(os.path.join(ROOT, "paper", "table_grpo_rivals.tex"), "w").write("\n".join(L) + "\n")


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
            fp = f1.replace("_duopoly_k1.json", "_forced2_k1.json")  # state-matched placebo (twice cost)
            dp = json.load(open(fp)).get("deviation_cut") if os.path.exists(fp) and os.path.getsize(fp) > 0 else None
            if dp and dp["deviation_size"] >= MIN_CUT:
                r["placebo"] = (dp["rival_aggression"][1] / dp["deviation_size"],
                                abs(dp["rival_aggression_ci95"][1] / dp["deviation_size"]))
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
        if "placebo" in r:
            u = r["placebo"]
            m["PlaceboUnit"] = f"{abs(u[0]) if abs(u[0]) < 0.005 else u[0]:.2f}\\pm{u[1]:.2f}"
        for f, v in m.items():
            macros[f"GrpoPr{key}{f}"] = v[1:] if v.startswith("-0.00") else v  # no "-0.00"
    # training against the scripted tit-for-tat rival (positive control), and the trained policies paired
    # with themselves in the in-context audit
    # and against the scripted rival that cheats unless a price war deters it
    for run, key in (("pricing_tft_0p6b_g09", "TftFwd"), ("pricing_tft_0p6b_g0", "TftMyo"),
                     ("pricing_tft_1p7b_g09", "TftMidFwd"), ("pricing_tft_1p7b_g0", "TftMidMyo"),
                     ("pricing_tft_0p6b_g09_s1", "TftFwdSeed"),
                     ("pricing_cheat_0p6b_g09", "CheatFwd"), ("pricing_cheat_0p6b_g0", "CheatMyo"),
                     ("pricing_cheatnash_0p6b_g09", "CheatNashFwd"), ("pricing_cheatnash_0p6b_g0", "CheatNashMyo"),
                     ("pricing_cheat_0p6b_g09_s1", "CheatFwdSeed"), ("pricing_cheat_0p6b_g09_s2", "CheatFwdSeedTwo"),
                     ("pricing_cheatnash_ent05_0p6b_g09", "CheatNashEnt"), ("pricing_cheat_1p7b_g09", "CheatMidFwd"),
                     ("pricing_cheat2self_0p6b_g09", "CheatSelf"),
                     ("pricing_cheat_4b_g09", "CheatBigFwd"), ("pricing_cheat_4b_g09_s1", "CheatBigFwdSeed"),
                     ("pricing_cheat_1p7b_g09_s1", "CheatMidFwdSeed"), ("pricing_cheat_0p6b_g09_s3", "CheatFwdSeedThree"),
                     ("pricing_tft_0p6b_g09_s2", "TftFwdSeedTwo"), ("pricing_cheatp03_0p6b_g09", "CheatPThree"),
                     ("pricing_cheatw09_0p6b_g09", "CheatWarNine"), ("pricing_cheatw30_0p6b_g09", "CheatWarThirty"),
                     ("pricing_cheatw30_0p6b_g09_s1", "CheatWarThirtySeed"), ("pricing_cheatw30_0p6b_g0", "CheatWarThirtyMyo")):
        r, m = {}, {f: "--" for f in ("Idx", "Prof", "Iters", "PairIdx", "PairTenIdx", "PairProf", "PairCutUnit",
                                     "PairGain", "PairCutGain", "War", "WarFirst", "AnsIdx", "Cheats",
                                     "PairBRUnit", "PairCutBench", "PairLagEight", "PairStray", "PairTenAtCost")
                                     + tuple(f"Pair{k}{f}" for k in ("Rise", "Shade", "Long", "TenCut", "Placebo")
                                             for f in ("Unit", "Gain", "Bench"))}
        log = os.path.join(ROOT, "results", "grpo", run, "log.jsonl")
        f1 = latest_audit(run)
        if os.path.exists(log):
            L = [json.loads(x) for x in open(log) if x.strip()]
            tail = L[-10:]
            if f1:  # training statistics over the ten iterations up to the audited checkpoint
                it = int(os.path.basename(f1).split("_adapter_")[1][:4])
                if it < L[-1]["iteration"]:
                    tail = [x for x in L if it - 9 <= x["iteration"] <= it]
            r["train"] = {"iterations": len(L),
                          "final_index": float(np.mean([x.get("policy_price_index_second_half", x.get("price_index_second_half")) for x in tail])),
                          "final_profit_over_nash": float(np.mean([x.get("policy_profit_over_nash", x.get("profit_over_nash")) for x in tail])),
                          "curve": [[x["iteration"], x.get("policy_price_index_second_half", x.get("price_index_second_half"))] for x in L]}
            m.update(Idx=f"{r['train']['final_index']:.2f}", Prof=f"{r['train']['final_profit_over_nash']:.2f}",
                     Iters=str(len(L)))
            if "war_share" in L[0]:  # cheater rival: how the policy answers cheats (last ten iterations)
                r["train"].update(war_share=float(np.mean([x["war_share"] for x in tail])),
                                  war_share_first=float(np.mean([x["war_share"] for x in L[:10]])),
                                  answer_index=float(np.nanmean([x["answer_price_index"] for x in tail])),
                                  cheats=float(np.mean([x["cheats_per_session"] for x in tail])),
                                  war_curve=[[x["iteration"], x["war_share"]] for x in L])
                m.update(War=f"{100 * r['train']['war_share']:.0f}", WarFirst=f"{100 * r['train']['war_share_first']:.0f}",
                         AnsIdx=f"{r['train']['answer_index']:.2f}", Cheats=f"{r['train']['cheats']:.1f}")
        if f1:
            d = json.load(open(f1))
            r["pair"] = {"index": tci(d["per_session"]["index"]), "profit_over_nash": tci(d["per_session"]["profit_over_nash"]),
                         "bench": benchmarks(np.asarray(d["prices_sessions"], float), d["args"]["scale"])}
            m.update(PairIdx=f"{r['pair']['index'][0]:.2f}", PairProf=f"{r['pair']['profit_over_nash'][0]:.2f}")
            if d.get("deviation"):
                g, gc = d["deviation"]["cum_gain_dev"], d["deviation"]["cum_gain_dev_ci95"]
                r["pair"]["br_gain"] = (g, gc)
                m["PairGain"] = f"{g:+.2f}\\pm{gc:.2f}"
                dv = d["deviation"]
                if dv["deviation_size"] >= MIN_CUT:  # the best-response deviation per unit of the cut
                    u = (dv["rival_aggression"][1] / dv["deviation_size"], dv["rival_aggression_ci95"][1] / dv["deviation_size"])
                    r["pair"]["br_per_unit"] = u
                    m["PairBRUnit"] = f"{abs(u[0]) if abs(u[0]) < 0.005 else u[0]:.2f}\\pm{u[1]:.2f}"
            m["PairCutBench"] = f"{r['pair']['bench']['cut_slope']:.2f}"
            dc = d.get("deviation_cut")
            if dc and dc["deviation_size"] >= MIN_CUT:
                u = (dc["rival_aggression"][1] / dc["deviation_size"], dc["rival_aggression_ci95"][1] / dc["deviation_size"])
                r["pair"]["cut_per_unit"] = u
                r["pair"]["cut_gain"] = (dc["cum_gain_dev"], dc["cum_gain_dev_ci95"])
                m["PairCutUnit"] = f"{abs(u[0]) if abs(u[0]) < 0.005 else u[0]:.2f}\\pm{u[1]:.2f}"
                m["PairCutGain"] = f"{dc['cum_gain_dev']:+.2f}\\pm{dc['cum_gain_dev_ci95']:.2f}"
            f10 = f1.replace("_duopoly_k1.json", "_duopoly_k10.json")
            if os.path.exists(f10) and os.path.getsize(f10) > 0:
                d10 = json.load(open(f10))
                r["pair"]["index_k10"] = tci(d10["per_session"]["index"])
                m["PairTenIdx"] = f"{r['pair']['index_k10'][0]:.2f}"
            if dc and dc["deviation_size"] >= MIN_CUT:
                r["pair"]["cut_by_lag"] = [x / dc["deviation_size"] for x in dc["rival_aggression"]]
                m["PairLagEight"] = f"{r['pair']['cut_by_lag'][-1]:.2f}"
                P1 = np.asarray(d["prices_sessions"], float)
                m["PairStray"] = str(int(((P1 <= 1.05 * d["bench"]["cost"]).any(axis=(1, 2))).sum()))
            if "index_k10" in r["pair"]:
                P10 = np.asarray(d10["prices_sessions"], float)
                half = P10[:, P10.shape[1] // 2:]
                r["pair"]["k10_sessions_at_cost"] = int(((half <= 1.05 * d10["bench"]["cost"]).mean(axis=(1, 2)) > 0.5).sum())
                m["PairTenAtCost"] = str(r["pair"]["k10_sessions_at_cost"])
            # extra tests (experiments/grpo_pricing_extra.sh), each a separate run of the same pair:
            # a visible 10% price rise (an agent that copies prices follows it, one that punishes cuts need
            # not), a 2% cut (shading), a 10% cut followed for 20 periods, the 10% cut at prices x10, and a
            # state-matched placebo (both firms forced to twice cost for 8 periods, then a 10% cut)
            for tag, test, tk, min_size in (("_duopoly_k1_hike", "deviation_hike", "Rise", MIN_CUT),
                                             ("_duopoly_k1_cut2", "deviation_cut", "Shade", 0.05),
                                             ("_duopoly_k1_h20", "deviation_cut", "Long", MIN_CUT),
                                             ("_duopoly_k10_cut", "deviation_cut", "TenCut", MIN_CUT),
                                             ("_forced2_k1", "deviation_cut", "Placebo", MIN_CUT)):
                fx = f1.replace("_duopoly_k1.json", tag + ".json")
                dx = json.load(open(fx)) if os.path.exists(fx) and os.path.getsize(fx) > 0 else {}
                t = dx.get(test)
                if not t or abs(t["deviation_size"]) < min_size:
                    continue
                u = (t["rival_aggression"][1] / t["deviation_size"], abs(t["rival_aggression_ci95"][1] / t["deviation_size"]))
                r["pair"][tk] = {"per_unit": u, "gain": (t["cum_gain_dev"], t["cum_gain_dev_ci95"]),
                                  "by_lag": [x / t["deviation_size"] for x in t["rival_aggression"]],
                                  "bench": benchmarks(np.asarray(dx["prices_sessions"], float), dx["args"]["scale"])["cut_slope"]}
                m[f"Pair{tk}Unit"] = f"{abs(u[0]) if abs(u[0]) < 0.005 else u[0]:.2f}\\pm{u[1]:.2f}"
                m[f"Pair{tk}Gain"] = f"{t['cum_gain_dev']:+.2f}\\pm{t['cum_gain_dev_ci95']:.2f}"
                m[f"Pair{tk}Bench"] = f"{r['pair'][tk]['bench']:.2f}"
        out[run] = r
        for f, v in m.items():
            macros[f"Grpo{key}{f}"] = v[1:] if v.startswith("-0.00") else v
    # the base model in the same state-matched placebo
    fb = os.path.join(ROOT, "results", "grpo_pricing_audit", "base_qwen3p0p6b_forced2_k1.json")
    db = json.load(open(fb)) if os.path.exists(fb) and os.path.getsize(fb) > 0 else {}
    t = db.get("deviation_cut")
    if t and t["deviation_size"] >= MIN_CUT:
        u = (t["rival_aggression"][1] / t["deviation_size"], abs(t["rival_aggression_ci95"][1] / t["deviation_size"]))
        macros["GrpoBasePlaceboUnit"] = f"{abs(u[0]) if abs(u[0]) < 0.005 else u[0]:.2f}\\pm{u[1]:.2f}"
        macros["GrpoBasePlaceboBench"] = f"{benchmarks(np.asarray(db['prices_sessions'], float), 1.0)['cut_slope']:.2f}"
        out["base_placebo"] = {"per_unit": u, "gain": (t["cum_gain_dev"], t["cum_gain_dev_ci95"])}
    else:
        macros["GrpoBasePlaceboUnit"] = macros["GrpoBasePlaceboBench"] = "--"
    # cross-play (firm 0 deviates): the tit-for-tat-trained firm's response to the cheater-trained one's cut,
    # and the cheater-trained firm's response to the tit-for-tat-trained one's cut
    for tag, key in (("cheat_vs_tft", "TftResp"), ("tft_vs_cheat", "CheatResp")):
        fc = os.path.join(ROOT, "results", "grpo_pricing_audit", f"cross_{tag}_duopoly_k1.json")
        dcx = json.load(open(fc)) if os.path.exists(fc) and os.path.getsize(fc) > 0 else {}
        t = dcx.get("deviation_cut")
        if t and t["deviation_size"] >= MIN_CUT:
            u = (t["rival_aggression"][1] / t["deviation_size"], abs(t["rival_aggression_ci95"][1] / t["deviation_size"]))
            macros[f"GrpoCross{key}Unit"] = f"{abs(u[0]) if abs(u[0]) < 0.005 else u[0]:.2f}\\pm{u[1]:.2f}"
            macros[f"GrpoCross{key}Gain"] = f"{t['cum_gain_dev']:+.2f}\\pm{t['cum_gain_dev_ci95']:.2f}"
            macros[f"GrpoCross{key}Idx"] = f"{dcx['summary']['index']:.2f}"
            out[f"cross_{tag}"] = {"per_unit": u, "gain": (t["cum_gain_dev"], t["cum_gain_dev_ci95"]),
                                   "index": dcx["summary"]["index"]}
        else:
            for f_ in ("Unit", "Gain", "Idx"):
                macros[f"GrpoCross{key}{f_}"] = "--"
    # the untrained base model, audited as the adapters (Figure 1's starting point)
    fb1 = os.path.join(ROOT, "results", "grpo_pricing_audit", "base_qwen3p0p6b_duopoly_k1.json")
    db1 = json.load(open(fb1)) if os.path.exists(fb1) and os.path.getsize(fb1) > 0 else {}
    if db1.get("deviation_cut"):
        t = db1["deviation_cut"]
        po = tci(db1["per_session"]["profit_over_nash"])
        out["base_k1"] = {"index": tci(db1["per_session"]["index"]), "profit_over_nash": po,
                          "profit_index": ((po[0] - 1) / (pi_ratio - 1), po[1] / (pi_ratio - 1)),
                          "cut_per_unit": (t["rival_aggression"][1] / t["deviation_size"],
                                           abs(t["rival_aggression_ci95"][1] / t["deviation_size"])),
                          "cut_by_lag": [x / t["deviation_size"] for x in t["rival_aggression"]],
                          "bench": benchmarks(np.asarray(db1["prices_sessions"], float), 1.0)}
        macros["GrpoBaseIdx"] = f"{out['base_k1']['index'][0]:.1f}"
        macros["GrpoBaseProf"] = f"{po[0]:.2f}"
        u = out["base_k1"]["cut_per_unit"]
        macros["GrpoBaseCutUnit"] = f"{abs(u[0]) if abs(u[0]) < 0.005 else u[0]:.2f}\\pm{u[1]:.2f}"
        fb10 = fb1.replace("_k1.json", "_k10.json")
        if os.path.exists(fb10):
            macros["GrpoBaseTenIdx"] = f"{json.load(open(fb10))['summary']['index']:.1f}"
    for k in ("GrpoBaseIdx", "GrpoBaseProf", "GrpoBaseCutUnit", "GrpoBaseTenIdx"):
        macros.setdefault(k, "--")
    rivals_table(out)
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
