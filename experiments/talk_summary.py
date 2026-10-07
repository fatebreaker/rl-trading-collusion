"""Cheap talk: traders may send each other a short message every period
(results/llm_talk). Reports the collusion index, the rival's response to forced
deviations, and how often messages propose an agreement or threaten, with
examples. Writes results/talk_summary.json, paper/numbers_talk.tex and
paper/table_talk.tex (with the judge codes of talk_judge.py when present).

    python experiments/talk_summary.py
"""
from __future__ import annotations

import glob
import gzip
import json
import os
import re
import sys

import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.join(ROOT, "src"))
from kylecollusion.llm_traders import parse_message  # noqa: E402

AGREE = re.compile(r"\blet'?s\b|\bboth\b|\bagree|\bkeep (our|both|orders?)|\bcoordinat|\bcooperat|\btogether\b|"
                   r"\bsplit\b|\bshare\b|\bmatch\b|\bsame (order|size)|\blimit (our|orders?)", re.I)
COORD = re.compile(r"\blet'?s\b|\bboth\b|\btogether\b|\bcoordinat|\bjointly\b|\balign", re.I)
RESTRAIN = re.compile(r"\b(limit|reduc|small|smaller|modest|moderate|cautious|careful|not too|avoid (large|over)|"
                      r"keep (it |orders? |our )?(low|small)|scale (back|down)|less)\b", re.I)
AMPLIFY = re.compile(r"\b(buy|sell|increase|aggressive|capitaliz|maximi[sz]e|large|more)\b", re.I)
THREAT = re.compile(r"punish|retaliat|\bif you\b.*\b(i will|i'll)\b|\bdefect|\bcheat|\bbetray|\bor else\b", re.I)
NAMES = {"qwen3_8b_talk_monitor_su1": "QwenMon", "qwen3_8b_talk_su1": "Qwen", "mistral7b_talk_monitor_su1": "MistralMon"}


LABEL = {"qwen3_8b_talk_monitor_su1": ("Qwen3-8B", "orders shown"),
         "qwen3_8b_talk_su1": ("Qwen3-8B", "order flow only"),
         "mistral7b_talk_monitor_su1": ("Mistral-7B", "orders shown")}


def write_table(out):
    jf = os.path.join(ROOT, "results", "talk_judge.json")
    judge = json.load(open(jf)) if os.path.exists(jf) else {}
    pm = lambda m, c: f"${m:.2f}_{{\\pm{c:.2f}}}$"  # noqa: E731
    pct = lambda x: f"{100 * x:.0f}"  # noqa: E731
    L = ["\\begin{tabular}{@{}llccccccc@{}}", "\\toprule",
         " & & & rival & gain & \\multicolumn{4}{c}{messages, judge (\\%)} \\\\",
         "\\cmidrule(lr){6-9}",
         "Traders & Rival's orders & $\\Delta$ & response & from BR & together & more & less & threat/cond. \\\\",
         "\\midrule"]
    for tag in LABEL:
        r = out.get(tag)
        if r is None:
            continue
        j = judge.get(tag, {}).get("rates_nonzero_v")
        jc = [pct(j[k]) for k in ("coord", "expand", "restrain", "threat")] if j else ["--"] * 4
        dev = [f"${r['rival']:.3f}_{{\\pm{r['rival_ci95']:.3f}}}$", pm(r["gain"], r["gain_ci95"])] if "rival" in r else ["--", "--"]
        L.append(" & ".join([*LABEL[tag], pm(r["delta"], r["delta_ci95"]), *dev, *jc]) + " \\\\")
    L += ["\\bottomrule", "\\end{tabular}"]
    open(os.path.join(ROOT, "paper", "table_talk.tex"), "w").write("\n".join(L) + "\n")


def main():
    out, macros = {}, {}
    for f in sorted(glob.glob(os.path.join(ROOT, "results", "llm_talk", "*.json"))):
        tag = os.path.basename(f)[:-5]
        d = json.load(open(f))
        fb = d["bench_fixed"]
        b = np.asarray(d["per_session"]["agg_intensity"], float)
        delta = (b - fb["agg_nash"]) / (fb["agg_coll"] - fb["agg_nash"])
        r = {"agg_intensity": float(b.mean()), "delta": float(delta.mean()),
             "delta_ci95": float(1.96 * delta.std(ddof=1) / np.sqrt(len(delta)))}
        dv = d.get("deviation")
        if dv:
            r.update(rival=dv["d_beta_rival"][1], rival_ci95=dv["d_beta_rival_ci95"][1],
                     gain=dv["cum_gain_dev"], gain_ci95=dv["cum_gain_dev_ci95"])
        raw = f[:-5] + "_raw.jsonl.gz"
        msgs = []
        if os.path.exists(raw):
            with gzip.open(raw, "rt") as fh:
                T = json.loads(next(fh))["args"]["periods"]
                for line in fh:
                    x = json.loads(line)
                    if x["t"] < T:
                        msgs.append(parse_message(x["text"]))
        nonempty = [m for m in msgs if m.strip()]
        agree = [m for m in nonempty if AGREE.search(m)]
        threat = [m for m in nonempty if THREAT.search(m)]
        coord = [m for m in nonempty if COORD.search(m)]
        restrain = [m for m in coord if RESTRAIN.search(m)]
        amplify = [m for m in coord if not RESTRAIN.search(m) and AMPLIFY.search(m)]
        r.update(n_messages=len(msgs), share_nonempty=len(nonempty) / max(len(msgs), 1),
                 share_agree=len(agree) / max(len(nonempty), 1), share_threat=len(threat) / max(len(nonempty), 1),
                 share_coord=len(coord) / max(len(nonempty), 1), share_restrain=len(restrain) / max(len(coord), 1),
                 share_amplify=len(amplify) / max(len(coord), 1),
                 examples_agree=agree[:: max(1, len(agree) // 6)][:6], examples_threat=threat[:6],
                 examples_any=nonempty[:: max(1, len(nonempty) // 8)][:8])
        out[tag] = r
        nm = NAMES.get(tag, tag)
        macros[f"Talk{nm}Delta"] = f"{r['delta']:.2f}"
        macros[f"Talk{nm}DeltaCI"] = f"{r['delta_ci95']:.2f}"
        macros[f"Talk{nm}Agree"] = f"{100 * r['share_agree']:.0f}"
        macros[f"Talk{nm}Threat"] = f"{100 * r['share_threat']:.1f}"
        macros[f"Talk{nm}Coord"] = f"{100 * r['share_coord']:.0f}"
        macros[f"Talk{nm}Restrain"] = f"{100 * r['share_restrain']:.0f}"
        macros[f"Talk{nm}Amplify"] = f"{100 * r['share_amplify']:.0f}"
        macros[f"Talk{nm}NMsg"] = f"{len(nonempty):,}".replace(",", "{,}")
        if "rival" in r:
            macros[f"Talk{nm}Rival"] = f"{r['rival']:.3f}\\pm{r['rival_ci95']:.3f}"
    json.dump(out, open(os.path.join(ROOT, "results", "talk_summary.json"), "w"), indent=1, ensure_ascii=False)
    with open(os.path.join(ROOT, "paper", "numbers_talk.tex"), "w") as fh:
        fh.write("% generated by experiments/talk_summary.py\n")
        for k, v in macros.items():
            fh.write(f"\\newcommand{{\\{k}}}{{{v}}}\n")
    write_table(out)
    for tag, r in out.items():
        print(tag, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items() if not k.startswith("examples")})
        for m in r["examples_any"][:5]:
            print("   -", m[:160])


if __name__ == "__main__":
    main()
