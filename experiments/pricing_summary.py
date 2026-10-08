"""Summary of the LLM pricing agents in the logit-Bertrand duopoly
(results/llm_bertrand/*.json): price index at two currency scales, the myopic
placebo and the instructed trigger strategy, with the rival's response to a
forced deviation. Writes paper/numbers_pricing.tex (macros; "--" where a run
is missing) and results/pricing_summary.json.

    python experiments/pricing_summary.py
"""
from __future__ import annotations

import gzip
import json
import os
import re

import numpy as np
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results", "llm_bertrand")
MODELS = {"qwen3_8b": "Qwen", "qwen3_8b_think": "QwenThink", "mistral7b": "Mistral"}
MIN_CUT = 0.1  # smallest mean deviation (in units of p_mono - p_Nash) for a per-unit response
CONDS = {"duopoly_k1": "DuoOne", "duopoly_k10": "DuoTen", "myopic_k1": "Myopic", "trigger_k1": "Trigger",
         "duopoly_k1_s1": "DuoOneSeedOne", "myopic_k1_s1": "MyopicSeedOne"}


WAR = re.compile(r"price war|retaliat", re.I)  # stated fear of a rival's punishment


def war_share(tag, cond):
    """Share of main-run responses (notes and traces) that invoke a price war or retaliation."""
    f = os.path.join(RES, f"{tag}_{cond}_raw.jsonl.gz")
    if not os.path.exists(f):
        return None
    n = k = 0
    with gzip.open(f, "rt") as fh:
        T = json.loads(next(fh))["args"]["periods"]
        for line in fh:
            r = json.loads(line)
            if r["t"] < T:
                n, k = n + 1, k + bool(WAR.search(r["text"]))
    return k / n if n else None


def load(tag, cond):
    f = os.path.join(RES, f"{tag}_{cond}.json")
    return json.load(open(f)) if os.path.exists(f) and os.path.getsize(f) > 0 else None


def write_table(table):
    """Appendix table: every pricing condition with both deviation tests."""
    names = {"duopoly_k1": "duopoly", "duopoly_k1_s1": "duopoly, seed 1", "duopoly_k10": "duopoly, prices $\\times10$",
             "myopic_k1": "myopic objective", "myopic_k1_s1": "myopic, seed 1", "trigger_k1": "instructed trigger"}
    pm = lambda m, c, sgn="": "--" if m is None else f"${m:{sgn}.2f}_{{\\pm{c:.2f}}}$"  # noqa: E731
    reg = os.path.join(HERE, "..", "results", "fish_regression.json")
    reg = json.load(open(reg)) if os.path.exists(reg) else {}

    def onpath(key):  # on-path coefficient on the rival's lagged price (Fish et al., Table 1)
        r = reg.get(key)
        if not r or r["delta"] != r["delta"]:
            return "--"
        tq = stats.t.ppf(0.975, r["clusters"] - 1)  # standard errors clustered by session
        return f"${r['delta']:.2f}_{{\\pm{tq * r['delta_se']:.2f}}}$"

    L = ["\\begin{tabular}{@{}llcccccccc@{}}", "\\toprule",
         " & & price & profit & \\multicolumn{2}{c}{best-response deviation} & \\multicolumn{2}{c}{price cut} "
         "& on-path & price war \\\\",
         "\\cmidrule(lr){5-6}\\cmidrule(lr){7-8}",
         "Model & Condition & index & / Nash & rival & gain & rival & gain & coefficient & (\\%) \\\\", "\\midrule"]
    first = True
    for tag, mname in (("qwen3_8b", "Qwen3-8B"), ("qwen3_8b_think", "Qwen3-8B, thinking"), ("mistral7b", "Mistral-7B")):
        rows = [(c, table.get(f"{tag}_{c}")) for c in names if table.get(f"{tag}_{c}")]
        if not rows:
            continue
        if not first:
            L.append("\\midrule")
        first = False
        for k, (c, r) in enumerate(rows):
            L.append(" & ".join([mname if k == 0 else "", names[c], pm(r["index"], r["index_ci95"]),
                                 pm(r["profit_over_nash"], r["profit_ci95"]),
                                 pm(r.get("pass_through"), r.get("pass_through_ci95"))
                                 if r.get("pass_through") is not None or r.get("gain") is None else "n/a",
                                 pm(r.get("gain"), r.get("gain_ci95"), "+"),
                                 pm(r.get("cut_pass_through"), r.get("cut_pass_through_ci95")),
                                 pm(r.get("cut_gain"), r.get("cut_gain_ci95"), "+"), onpath(f"{tag}_{c}"),
                                 "--" if r.get("war_share") is None else f"{100 * r['war_share']:.0f}"]) + " \\\\")
    L += ["\\bottomrule", "\\end{tabular}"]
    os.makedirs(os.path.join(HERE, "..", "paper"), exist_ok=True)
    open(os.path.join(HERE, "..", "paper", "table_pricing.tex"), "w").write("\n".join(L) + "\n")


def main():
    macros, table = {}, {}
    for tag, mname in MODELS.items():
        for cond, cname in CONDS.items():
            d = load(tag, cond)
            key = mname + cname
            row = None
            if d is not None:
                s = d["summary"]
                pn = d["per_session"]["profit_over_nash"]
                tq = stats.t.ppf(0.975, len(pn) - 1)  # t intervals across sessions
                pci = tq * float(np.std(pn, ddof=1)) / np.sqrt(len(pn))
                s = dict(s, index_ci95=tq * float(np.std(d["per_session"]["index"], ddof=1)) / np.sqrt(len(pn)))
                row = {"index": s["index"], "index_ci95": s["index_ci95"],
                       "profit_over_nash": s["profit_over_nash"], "profit_ci95": pci,
                       "parse_fail_rate": d["parse_fail_rate"]}
                macros[f"PrProfCI{key}"] = f"{pci:.2f}"
                w = war_share(tag, cond)
                if w is not None:
                    row["war_share"] = w
                    macros[f"PrWar{key}"] = f"{100 * w:.0f}"
                macros[f"PrIdx{key}"] = f"{s['index']:.2f}"
                macros[f"PrIdxCI{key}"] = f"{s['index_ci95']:.2f}"
                macros[f"PrProf{key}"] = f"{s['profit_over_nash']:.2f}"
                dv = d.get("deviation")
                if dv:
                    row.update(rival=dv["rival_aggression"][1], rival_ci95=dv["rival_aggression_ci95"][1],
                               gain=dv["cum_gain_dev"], gain_ci95=dv["cum_gain_dev_ci95"],
                               deviation_size=dv["deviation_size"])
                    macros[f"PrAgg{key}"] = f"{dv['rival_aggression'][1]:.2f}\\pm{dv['rival_aggression_ci95'][1]:.2f}"
                    # per unit of the deviator's price cut, comparable with the Q-learners' pass-through;
                    # undefined when the best response is (on average) about the agent's own price
                    if dv["deviation_size"] >= MIN_CUT:
                        u, uc = dv["rival_aggression"][1] / dv["deviation_size"], dv["rival_aggression_ci95"][1] / dv["deviation_size"]
                        row.update(pass_through=u, pass_through_ci95=uc)
                        macros[f"PrPass{key}"] = f"{u:.2f}\\pm{uc:.2f}"
                        macros[f"PrPassMean{key}"] = f"{u:.2f}"
                    macros[f"PrGain{key}"] = f"{dv['cum_gain_dev']:+.2f}\\pm{dv['cum_gain_dev_ci95']:.2f}"
                dc = d.get("deviation_cut")
                if dc and dc["deviation_size"] >= MIN_CUT:
                    u, uc = dc["rival_aggression"][1] / dc["deviation_size"], dc["rival_aggression_ci95"][1] / dc["deviation_size"]
                    row.update(cut_pass_through=u, cut_pass_through_ci95=uc, cut_gain=dc["cum_gain_dev"],
                               cut_gain_ci95=dc["cum_gain_dev_ci95"])
                    macros[f"PrCutPass{key}"] = f"{u:.2f}\\pm{uc:.2f}"
                    macros[f"PrCutGain{key}"] = f"{dc['cum_gain_dev']:+.2f}\\pm{dc['cum_gain_dev_ci95']:.2f}"
            table[f"{tag}_{cond}"] = row
            for m in ("PrIdx", "PrIdxCI", "PrProf", "PrProfCI", "PrWar", "PrAgg", "PrPass", "PrPassMean", "PrGain", "PrCutPass", "PrCutGain"):
                macros.setdefault(f"{m}{key}", "--")
    os.makedirs(os.path.join(HERE, "..", "paper"), exist_ok=True)
    out = os.path.join(HERE, "..", "paper", "numbers_pricing.tex")
    with open(out, "w") as fh:
        fh.write("% generated by experiments/pricing_summary.py\n")
        for k in sorted(macros):
            fh.write(f"\\newcommand{{\\{k}}}{{{macros[k]}}}\n")
    json.dump(table, open(os.path.join(HERE, "..", "results", "pricing_summary.json"), "w"), indent=1)
    write_table(table)
    for k, v in table.items():
        print(k, "missing" if v is None else {a: round(b, 3) for a, b in v.items()})


if __name__ == "__main__":
    main()
