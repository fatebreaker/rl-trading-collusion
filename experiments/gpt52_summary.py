"""GPT-5.2 (high reasoning effort) under the protocol of Fish et al.: sessions,
what the agents write, and the deviation tests of the forward-looking run and its
myopic placebo. Reads results/llm_fish/gpt52_high_{duopoly,myopic}_P1*.json;
writes results/gpt52_summary.json, paper/numbers_gpt52.tex and
paper/table_gpt52.tex (one row per session).

    python experiments/gpt52_summary.py
"""
from __future__ import annotations

import gzip
import json
import os
import re
import sys

import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.join(ROOT, "src"))
from kylecollusion.llm_pricing import parse_fish  # noqa: E402

RES = os.path.join(ROOT, "results", "llm_fish")
WAR = re.compile(r"price war|retaliat", re.I)
COOP = re.compile(r"cooperat|collu|tacit|coordinat|cartel", re.I)
# a rule that answers a rival's price cut with a cut (prices contain dots, so lines, not sentences)
MATCH = re.compile(r"(match|follow)[^\n]{0,60}(cut|undercut|lower|drop|below|<)"
                   r"|(cut|undercut|lower|drop)s?[^\n]{0,80}(match|follow)"
                   r"|(<|below|less than|lower than)[^\n]{0,60}match", re.I)


def load(name):
    f = os.path.join(RES, f"{name}.json")
    return json.load(open(f)) if os.path.exists(f) else None


def texts(name, T):
    """Main-run responses and each firm's last readable PLANS file."""
    rows, last = [], {}
    with gzip.open(os.path.join(RES, f"{name}_raw.jsonl.gz"), "rt") as fh:
        next(fh)
        for line in fh:
            r = json.loads(line)
            if r["t"] < T:
                rows.append(r)
                price, plans, _ = parse_fish(r["text"])
                if price is not None:
                    last[(r["s"], r["i"])] = plans or ""
    return rows, last


def per_unit(dv):
    u = dv["deviation_size"]
    return [x / u for x in dv["rival_aggression"]], [x / u for x in dv["rival_aggression_ci95"]]


def main():
    out, macros = {}, {}
    fw, my = load("gpt52_high_duopoly_P1"), load("gpt52_high_myopic_P1")
    if fw is None:
        print("no GPT-5.2 run yet")
        return
    b = fw["bench"]
    span = b["p_mono"] - b["p_nash"]
    P = np.asarray(fw["prices_sessions"], float)  # (S, T, 2)
    S, T = P.shape[:2]
    idx = (P - b["p_nash"]) / span
    rows, last = texts("gpt52_high_duopoly_P1", T)
    sessions = []
    for s in range(S):
        # stable: over the last 70 periods both prices stay within two cents of their final
        # level and within a cent of each other
        tail = P[s, -70:, :]
        const = bool(np.abs(tail - P[s, -1:, :]).max() <= 0.0201
                     and np.abs(tail[:, 0] - tail[:, 1]).max() <= 0.0201)
        streak = 0  # final periods with both prices within a cent of each other
        for e in (np.abs(P[s, :, 0] - P[s, :, 1]) <= 0.0101)[::-1]:
            if not e:
                break
            streak += 1
        sessions.append({"session": s, "index_last10": idx[s, -10:, :].mean(0).round(2).tolist(),
                         "index_second_half": float(idx[s, T // 2:, :].mean()),
                         "profit_over_nash": fw["per_session"]["profit_over_nash"][s],
                         "equal_streak": int(streak), "stable": bool(const),
                         "match_rules": sum(bool(MATCH.search(last.get((s, i), ""))) for i in (0, 1))})
    stable = [x for x in sessions if x["stable"]]
    cartel = [x for x in stable if x["index_last10"][0] <= 1]
    above = [x for x in stable if x["index_last10"][0] > 1]
    n_plans = len(last)
    n_match = sum(bool(MATCH.search(p)) for p in last.values())
    war = sum(bool(WAR.search(r["text"])) for r in rows) / len(rows)
    coop = sum(bool(COOP.search(r["text"])) for r in rows) / len(rows)
    out["forward"] = {"sessions": sessions, "war_share": war, "coop_share": coop,
                      "match_plans": n_match, "plans": n_plans}
    macros.update({"GptS": S, "GptT": T, "GptStableN": len(stable), "GptCartelN": len(cartel),
                   "GptWar": f"{100 * war:.0f}", "GptCoop": f"{100 * coop:.0f}",
                   "GptMatchPlans": n_match, "GptPlans": n_plans})
    if cartel:
        lo, hi = min(x["index_last10"][0] for x in cartel), max(x["index_last10"][0] for x in cartel)
        plo = min(x["profit_over_nash"] for x in cartel)
        phi = max(x["profit_over_nash"] for x in cartel)
        # text-mode macros (math inside), used outside $...$
        macros.update({"GptCartelIdx": f"${lo:.2f}$" if lo == hi else f"${lo:.2f}$ and ${hi:.2f}$",
                       "GptCartelProf": (f"${plo:.2f}$" if f"{plo:.2f}" == f"{phi:.2f}"
                                         else f"${plo:.2f}$--${phi:.2f}$"),
                       "GptCartelStreak": min(x["equal_streak"] for x in cartel)})
    if above:
        macros.update({"GptAboveIdx": f"{above[0]['index_last10'][0]:.2f}",
                       "GptAboveProf": f"{above[0]['profit_over_nash']:.2f}"})
    dv = fw.get("deviation")
    if dv:
        u, c = per_unit(dv)
        H = min(6, len(u) - 1)
        out["forward"]["br_per_unit"] = {"mean": u, "ci95": c, "size": dv["deviation_size"]}
        macros.update({"GptBRLagOne": f"{u[1]:.2f}\\pm{c[1]:.2f}", "GptBRLagOneMean": f"{u[1]:.2f}",
                       "GptBRLaterMin": f"{min(u[2:H + 1]):.2f}", "GptBRLaterMax": f"{max(u[2:H + 1]):.2f}",
                       "GptBRSize": f"{dv['deviation_size']:.2f}",
                       "GptBRGain": f"{dv['cum_gain_dev']:+.2f}\\pm{dv['cum_gain_dev_ci95']:.2f}",
                       "GptBRLow": f"{u[1] - c[1]:.2f}"})
    dc = fw.get("deviation_cut")
    if dc:
        u, c = per_unit(dc)
        H = min(6, len(u) - 1)
        out["forward"]["cut_per_unit"] = {"mean": u, "ci95": c, "size": dc["deviation_size"]}
        macros.update({"GptCutLagOne": f"{u[1]:.2f}\\pm{c[1]:.2f}", "GptCutLaterMin": f"{min(u[2:H + 1]):.2f}",
                       "GptCutLaterMax": f"{max(u[2:H + 1]):.2f}",
                       "GptCutGain": f"{dc['cum_gain_dev']:+.2f}\\pm{dc['cum_gain_dev_ci95']:.2f}"})
    if my:
        Tm = my["args"]["periods"]
        mrows, mlast = texts("gpt52_high_myopic_P1", Tm)
        mwar = sum(bool(WAR.search(r["text"])) for r in mrows) / len(mrows)
        out["myopic"] = {"index": my["summary"]["index"], "index_ci95": my["summary"]["index_ci95"],
                         "war_share": mwar, "match_plans": sum(bool(MATCH.search(p)) for p in mlast.values()),
                         "plans": len(mlast)}
        macros.update({"GptMyoWar": f"{100 * mwar:.0f}", "GptMyoMatchPlans": out["myopic"]["match_plans"],
                       "GptMyoPlans": len(mlast)})
        dm = my.get("deviation_cut")
        if dm:
            u, c = per_unit(dm)
            H = min(6, len(u) - 1)
            out["myopic"]["cut_per_unit"] = {"mean": u, "ci95": c}
            macros.update({"GptMyoCutLagOne": f"{u[1]:.2f}\\pm{c[1]:.2f}",
                           "GptMyoCutLaterMax": f"{max(u[2:H + 1]):.2f}",
                           "GptMyoCutGain": f"{dm['cum_gain_dev']:+.2f}\\pm{dm['cum_gain_dev_ci95']:.2f}"})
    led = [json.loads(x) for x in open(os.path.join(ROOT, "results", "openai_spend.jsonl")) if x.strip()]
    cost = sum(x["cost"] for x in led if x.get("tag", "").startswith("gpt52_high_")
               and "pilot" not in x.get("tag", ""))
    out["api_cost"] = cost
    macros["GptCost"] = f"{cost:.0f}"
    json.dump(out, open(os.path.join(ROOT, "results", "gpt52_summary.json"), "w"), indent=1)
    with open(os.path.join(ROOT, "paper", "numbers_gpt52.tex"), "w") as fh:
        fh.write("% generated by experiments/gpt52_summary.py\n")
        for k, v in macros.items():
            fh.write(f"\\newcommand{{\\{k}}}{{{v}}}\n")
    L = ["\\begin{tabular}{@{}ccccccc@{}}", "\\toprule",
         " & \\multicolumn{2}{c}{price index, last 10} & index, & profit & within a cent & plans that \\\\",
         "\\cmidrule(lr){2-3}",
         "Session & firm 1 & firm 2 & periods 51--100 & / Nash & (final periods) & answer a cut \\\\", "\\midrule"]
    for x in sessions:
        L.append(f"{x['session'] + 1} & ${x['index_last10'][0]:.2f}$ & ${x['index_last10'][1]:.2f}$ & "
                 f"${x['index_second_half']:.2f}$ & ${x['profit_over_nash']:.2f}$ & {x['equal_streak']} & "
                 f"{x['match_rules']} of 2 \\\\")
    L += ["\\bottomrule", "\\end{tabular}"]
    open(os.path.join(ROOT, "paper", "table_gpt52.tex"), "w").write("\n".join(L) + "\n")
    print(json.dumps({k: v for k, v in macros.items()}, indent=1))


if __name__ == "__main__":
    main()
