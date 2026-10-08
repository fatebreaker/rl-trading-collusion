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
    # re-test of the cartel-like sessions from their saved state (fish_resume_test.py), per event
    rt = load("gpt52_high_duopoly_P1_sessions_cut")
    if rt:
        t = rt["test"]
        agg = np.asarray(t["per_event"]["rival_aggression"])   # (events, lags, sessions)
        size = np.asarray(t["per_event"]["deviation_size"])   # (events, sessions)
        gain = np.asarray(t["per_event"]["gain"]).ravel()
        per = (agg[:, 1, :] / size).ravel()                     # lag-1 response per unit, every event
        n = per.size
        rows_rt = {}
        for k, s_ in enumerate(rt["sessions"]):
            u = agg[:, :, k] / size[:, k][:, None]
            g = np.asarray(t["per_event"]["gain"])[:, k]
            rows_rt[s_] = {"per_unit_by_lag": u.mean(0).tolist(), "gain": float(g.mean()),
                           "gain_ci95": float(1.96 * g.std(ddof=1) / np.sqrt(len(g)))}
        out["retest"] = {"sessions": rt["sessions"], "n_events": int(n), "lag1_per_unit": float(per.mean()),
                         "lag1_ci95": float(1.96 * per.std(ddof=1) / np.sqrt(n)),
                         "gain": float(gain.mean()), "gain_ci95": float(1.96 * gain.std(ddof=1) / np.sqrt(n)),
                         "per_session": rows_rt, "cost": rt["usage"]["cost"]}
        r = out["retest"]
        macros.update({"GptRetestN": r["n_events"], "GptRetestUnit": f"{r['lag1_per_unit']:.2f}\\pm{r['lag1_ci95']:.2f}",
                       "GptRetestGain": f"{r['gain']:+.2f}\\pm{r['gain_ci95']:.2f}",
                       "GptRetestMatched": int(np.sum(np.abs(per - 1) < 0.05)),
                       "GptRetestCost": f"{r['cost']:.0f}"})
    if rt:  # per session (two sessions; events within a session are not independent)
        ps = out["retest"]["per_session"]
        units = [f"{ps[k]['per_unit_by_lag'][1]:.2f}" for k in sorted(ps, key=int)]
        gains = [f"{ps[k]['gain']:+.2f}" for k in sorted(ps, key=int)]
        macros.update({"GptRetestSessUnit": " and ".join(units), "GptRetestSessGain": " and ".join(gains)})
    # placebo at the cartel state: session 4 rebuilt with the myopic objective (same history and plans)
    mp = load("gpt52_high_duopoly_P1_session4_cut_myopic")
    if mp and rt:
        r = next(iter(mp["per_session"].values()))
        lag1 = np.asarray(r["lag1_per_event"], float)
        fw4 = out["retest"]["per_session"][mp["sessions"][0]] if mp["sessions"][0] in out["retest"]["per_session"] \
            else out["retest"]["per_session"][str(mp["sessions"][0])]
        out["myopic_at_cartel_state"] = {"session": mp["sessions"][0], "events": int(lag1.size),
                                          "lag1_mean": float(lag1.mean()), "lag1_min": float(lag1.min()),
                                          "lag1_max": float(lag1.max()), "per_unit_by_lag": r["per_unit_by_lag"],
                                          "forward_lag1": fw4["per_unit_by_lag"][1], "cost": mp["usage"]["cost"]}
        macros.update({"GptMyoStateN": int(lag1.size), "GptMyoStateUnit": f"{lag1.mean():.2f}",
                       "GptMyoStateRange": f"${lag1.min():.2f}$--${lag1.max():.2f}$",
                       "GptMyoStateSession": int(mp["sessions"][0]) + 1})
    if dc:  # conservative bound: intervals across events, three per session (design effect up to sqrt 3)
        u, c = per_unit(dc)
        macros["GptCutLowDE"] = f"{u[1] - np.sqrt(3) * c[1]:.2f}"
    # what a 10% cut from the cartel state earns against rivals that do not punish (analytic)
    sys.path.insert(0, os.path.join(ROOT, "experiments"))
    from fish_regression import br_slope  # noqa: E402
    from kylecollusion.bertrand import logit_demand  # noqa: E402
    from kylecollusion.llm_pricing import scaled_config  # noqa: E402
    bc = scaled_config(1.0)

    def prof(p0, p1):
        q = logit_demand(np.array([[p0, p1]]), bc)[0]
        return float((p0 - bc.cost) * q[0])
    p0 = 1.81  # session 4's price
    cut, sl = 0.9 * p0, br_slope(p0, 1.0)
    base = sum(0.95 ** k * prof(p0, p0) for k in range(7))

    def path(r1):
        return prof(cut, p0) + sum(0.95 ** k * prof(p0, r1 if k == 1 else p0) for k in range(1, 7))
    bench = {"none": (path(p0) - base) / b["pi_nash"], "best_response": (path(p0 - sl * (p0 - cut)) - base) / b["pi_nash"],
             "match": (path(cut) - base) / b["pi_nash"], "slope": sl}
    out["cut_benchmarks"] = bench
    macros.update({"GptBenchNone": f"{bench['none']:+.2f}", "GptBenchBR": f"{bench['best_response']:+.2f}",
                   "GptBenchMatch": f"{bench['match']:+.2f}", "GptBenchSlope": f"{sl:.2f}"})
    # the same plan coding for the open models under the protocol
    for tag, key in (("gptoss20b_duopoly_P1", "OssPOne"), ("gptoss20b_myopic_P1", "OssMyopic"),
                     ("gptoss120b_duopoly_P1", "OssBig")):
        d = load(tag)
        if d is None:
            continue
        _, lp = texts(tag, d["args"]["periods"])
        hits = [k for k, v in lp.items() if MATCH.search(v)]
        out[f"plans_{tag}"] = {"match": len(hits), "plans": len(lp), "hits": [list(h) for h in sorted(hits)]}
        macros.update({f"Gpt{key}MatchPlans": len(hits), f"Gpt{key}Plans": len(lp)})
    led = [json.loads(x) for x in open(os.path.join(ROOT, "results", "openai_spend.jsonl")) if x.strip()]
    cost = sum(x["cost"] for x in led if x.get("tag", "").startswith("gpt52_high_")
               and "pilot" not in x.get("tag", ""))  # main runs, their tests and the re-test
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
