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
        # stable for: final periods in which both prices stay within two cents of each other and
        # of their final level; a session is stable if this holds for the last 70 periods
        ok = (np.abs(P[s] - P[s, -1:]).max(1) <= 0.0201) & (np.abs(P[s, :, 0] - P[s, :, 1]) <= 0.0201)
        streak = 0
        for e in ok[::-1]:
            if not e:
                break
            streak += 1
        last10 = idx[s, -10:, :].mean(0)
        drop = float(idx[s, -1].mean() - idx[s, -20].mean())  # change over the last 20 periods
        if streak >= 70:
            pattern = "cartel-like" if last10.max() <= 1 else "above monopoly"
        elif drop < -0.2:
            pattern = "falling"
        elif abs(last10[0] - last10[1]) > 0.5:
            pattern = "asymmetric"
        else:
            pattern = "low"
        sessions.append({"session": s, "index_last10": last10.round(2).tolist(), "index_last10_mean": float(last10.mean()),
                         "index_second_half": float(idx[s, T // 2:, :].mean()),
                         "profit_over_nash": fw["per_session"]["profit_over_nash"][s],
                         "stable_streak": int(streak), "stable": streak >= 70, "change_last20": drop,
                         "pattern": pattern,
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
    # profit over periods 51-100, session by session: mean and t interval across sessions
    from scipy import stats
    prof = np.asarray(fw["per_session"]["profit_over_nash"], float)
    half = stats.t.ppf(0.975, S - 1) * prof.std(ddof=1) / np.sqrt(S)
    out["forward"]["profit"] = {"mean": float(prof.mean()), "t95": float(half), "above_nash": int((prof > 1).sum()),
                                "p_mean_above_nash": float(stats.ttest_1samp(prof, 1.0).pvalue)}
    macros.update({"GptProfMean": f"{prof.mean():.2f}",
                   "GptProfRange": f"${prof.mean() - half:.2f}$--${prof.mean() + half:.2f}$",
                   "GptProfAboveN": int((prof > 1).sum())})
    pat = {k: [x for x in sessions if x["pattern"] == k] for k in ("low", "falling", "asymmetric")}
    num = {1: "one", 2: "two", 3: "three", 4: "four"}
    if pat["low"]:
        v = sorted(x["index_last10_mean"] for x in pat["low"])
        macros.update({"GptLowN": num.get(len(v), len(v)), "GptLowIdx": " and ".join(f"${x:.2f}$" for x in v)})
    macros["GptFallingN"] = num.get(len(pat["falling"]), len(pat["falling"]))
    if pat["asymmetric"]:
        a0 = pat["asymmetric"][0]["index_last10"]
        macros["GptAsymIdx"] = f"${min(a0):.2f}$ and ${max(a0):.2f}$"
    if cartel:
        lo, hi = min(x["index_last10"][0] for x in cartel), max(x["index_last10"][0] for x in cartel)
        plo = min(x["profit_over_nash"] for x in cartel)
        phi = max(x["profit_over_nash"] for x in cartel)
        # text-mode macros (math inside), used outside $...$
        macros.update({"GptCartelIdx": f"${lo:.2f}$" if lo == hi else f"${lo:.2f}$ and ${hi:.2f}$",
                       "GptCartelProf": (f"${plo:.2f}$" if f"{plo:.2f}" == f"{phi:.2f}"
                                         else f"${plo:.2f}$--${phi:.2f}$"),
                       "GptCartelStreak": min(x["stable_streak"] for x in cartel)})
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
        units = [f"${ps[k]['per_unit_by_lag'][1]:.2f}$" for k in sorted(ps, key=int)]
        gains = [f"${ps[k]['gain']:.2f}$" for k in sorted(ps, key=int)]
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
                       "GptMyoStateList": ", ".join(f"${x:.2f}$" for x in lag1[:-1]) + f" and ${lag1[-1]:.2f}$",
                       "GptMyoStateSession": int(mp["sessions"][0]) + 1})
    if dc:  # conservative bound: the main run's tests keep only event-level summaries, so take the
        # worst case, events within a session identical (intra-class correlation 1): then the S session
        # means carry all the information, and the bound is a t interval on S - 1 degrees of freedom
        from scipy import stats
        u, c = per_unit(dc)
        n, m = dc["n_events"], dc["n_events"] / S
        sd_n = c[1] / 1.96 * np.sqrt(n)                    # sd across events
        sd_s = sd_n * np.sqrt((n - 1) / (m * (S - 1)))     # sd across session means if events repeat
        macros["GptCutLowDE"] = f"{u[1] - stats.t.ppf(0.975, S - 1) * sd_s / np.sqrt(S):.2f}"
    # what a 10% cut from the cartel state earns against rivals that do not punish (analytic)
    from kylecollusion.bertrand import logit_demand  # noqa: E402
    from kylecollusion.llm_pricing import best_response_secant, scaled_config  # noqa: E402
    bc = scaled_config(1.0)

    def prof(p0, p1):
        q = logit_demand(np.array([[p0, p1]]), bc)[0]
        return float((p0 - bc.cost) * q[0])
    p0 = 1.81  # session 4's price
    cut = 0.9 * p0
    sl = best_response_secant(p0, cut, bc)  # what a rival that only best-responds follows of this cut
    base = sum(0.95 ** k * prof(p0, p0) for k in range(7))

    def path(r1):
        return prof(cut, p0) + sum(0.95 ** k * prof(p0, r1 if k == 1 else p0) for k in range(1, 7))
    bench = {"none": (path(p0) - base) / b["pi_nash"], "best_response": (path(p0 - sl * (p0 - cut)) - base) / b["pi_nash"],
             "match": (path(cut) - base) / b["pi_nash"], "slope": sl}
    out["cut_benchmarks"] = bench
    if rt:  # the same benchmark at each re-tested session's price (the rival follows the deviator, firm 1)
        P0 = np.asarray(fw["prices_sessions"], float)[:, -1, 0]
        brs = {s_: best_response_secant(P0[s_], 0.9 * P0[s_], bc) for s_ in rt["sessions"]}
        out["retest"]["br_benchmark"] = {str(k): v for k, v in brs.items()}
        out["retest"]["br_benchmark_mean"] = float(np.mean(list(brs.values())))
    sg = lambda x: f"{0.0 if abs(x) < 0.005 else x:+.2f}"  # noqa: E731  (no negative zero)
    macros.update({"GptBenchNone": sg(bench["none"]), "GptBenchBR": sg(bench["best_response"]),
                   "GptBenchMatch": sg(bench["match"]), "GptBenchSlope": f"{sl:.2f}"})
    # the same plan coding for the open models under the protocol
    for tag, key in (("gptoss20b_duopoly_P1", "OssPOne"), ("gptoss20b_myopic_P1", "OssMyopic"),
                     ("gptoss120b_duopoly_P1", "OssBig")):
        d = load(tag)
        if d is None:
            continue
        rows_o, lp = texts(tag, d["args"]["periods"])
        hits = [k for k, v in lp.items() if MATCH.search(v)]
        war_o = sum(bool(WAR.search(r["text"])) for r in rows_o) / len(rows_o)  # same regex as for GPT-5.2
        out[f"plans_{tag}"] = {"match": len(hits), "plans": len(lp), "hits": [list(h) for h in sorted(hits)],
                               "war_share": war_o}
        macros.update({f"Gpt{key}MatchPlans": len(hits), f"Gpt{key}Plans": len(lp), f"Gpt{key}War": f"{100 * war_o:.0f}"})
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
    L = ["\\begin{tabular}{@{}cccccccl@{}}", "\\toprule",
         " & \\multicolumn{3}{c}{price index} & profit & stable & plans that & \\\\",
         "\\cmidrule(lr){2-4}",
         "Session & firm 1 & firm 2 & 51--100 & / Nash & for & answer a cut & pattern \\\\", "\\midrule"]
    for x in sessions:
        L.append(f"{x['session'] + 1} & ${x['index_last10'][0]:.2f}$ & ${x['index_last10'][1]:.2f}$ & "
                 f"${x['index_second_half']:.2f}$ & ${x['profit_over_nash']:.2f}$ & {x['stable_streak']} & "
                 f"{x['match_rules']} of 2 & {x['pattern']} \\\\")
    L += ["\\bottomrule", "\\end{tabular}"]
    open(os.path.join(ROOT, "paper", "table_gpt52.tex"), "w").write("\n".join(L) + "\n")
    print(json.dumps({k: v for k, v in macros.items()}, indent=1))


if __name__ == "__main__":
    main()
