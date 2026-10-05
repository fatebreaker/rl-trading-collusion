"""Agreement between the trace judge (GPT-5.4-mini), two stronger model judges
(a blind Claude Opus 5.5 annotator and GPT-5.5) and the keyword coder on the
200-item validation sample.

Writes results/annotation_agreement.json, paper/table_annotation.tex and
paper/numbers_annotation.tex.
"""
from __future__ import annotations

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import trace_analysis as ta  # noqa: E402

LABELS = ["impact", "rival_flow", "rival_infer", "coop", "punish", "half"]
NAMES = {"impact": "Price impact", "rival_flow": "Rival as flow", "rival_infer": "Infers rival",
         "coop": "Cooperation", "punish": "Punishment", "half": "Scales down"}


def load():
    items = json.load(open(os.path.join(ROOT, "results", "annotation_items.json")))
    key = {k["id"]: k for k in json.load(open(os.path.join(ROOT, "results", "annotation_key.json")))}
    opus = {r["id"]: r for r in json.load(open(os.path.join(ROOT, "results", "annotation_ai_opus.json")))}
    g55 = {r["id"]: r for r in json.load(open(os.path.join(ROOT, "results", "annotation_ai_gpt55.json")))}
    rows = []
    for it in items:
        kw = ta.code(it["text"])
        kw = {"impact": kw["impact"], "rival": kw["rival"], "coop": kw["coop"], "punish": kw["punish"]}
        rows.append({"id": it["id"], "instructed": key[it["id"]]["instructed"],
                     "judge": key[it["id"]]["judge"], "opus": opus[it["id"]], "gpt55": g55[it["id"]],
                     "kw": kw})
    return rows


def kappa(a, b):
    n = len(a)
    if n == 0:
        return float("nan"), float("nan")
    po = sum(x == y for x, y in zip(a, b)) / n
    pa, pb = sum(a) / n, sum(b) / n
    pe = pa * pb + (1 - pa) * (1 - pb)
    return po, (po - pe) / (1 - pe) if pe < 1 else float("nan")


def fleiss(rows3):
    """Fleiss' kappa for binary labels, rows3 = list of tuples of 3 booleans."""
    n, m = len(rows3), 3
    p_true = sum(sum(r) for r in rows3) / (n * m)
    pe = p_true ** 2 + (1 - p_true) ** 2
    pi = [(sum(r) * (sum(r) - 1) + (m - sum(r)) * (m - sum(r) - 1)) / (m * (m - 1)) for r in rows3]
    pbar = sum(pi) / n
    return (pbar - pe) / (1 - pe) if pe < 1 else float("nan")


def main():
    rows = load()
    out = {"n": len(rows), "labels": {}}
    for lab in LABELS:
        J = [bool(r["judge"][lab]) for r in rows]
        O = [bool(r["opus"][lab]) for r in rows]
        G = [bool(r["gpt55"][lab]) for r in rows]
        res = {"positives": {"judge": sum(J), "opus": sum(O), "gpt55": sum(G)}}
        for (na, a), (nb, b) in [(("judge", J), ("opus", O)), (("judge", J), ("gpt55", G)),
                                 (("opus", O), ("gpt55", G))]:
            po, k = kappa(a, b)
            res[f"{na}_{nb}"] = {"agree": po, "kappa": k}
        res["fleiss"] = fleiss(list(zip(J, O, G)))
        # the judge against the items where both stronger judges agree
        ref = [(j, o) for j, o, g in zip(J, O, G) if o == g]
        tp = sum(j and o for j, o in ref)
        fp = sum(j and not o for j, o in ref)
        fn = sum((not j) and o for j, o in ref)
        res["vs_consensus"] = {"n": len(ref), "precision": tp / (tp + fp) if tp + fp else None,
                               "recall": tp / (tp + fn) if tp + fn else None, "tp": tp, "fp": fp, "fn": fn}
        out["labels"][lab] = res
    # cooperation / punishment outside the instructed control: any model flags?
    unins = [r for r in rows if not r["instructed"]]
    out["uninstructed"] = {"n": len(unins)}
    for who in ("judge", "opus", "gpt55"):
        out["uninstructed"][who] = {lab: sum(bool(r[who][lab]) for r in unins) for lab in ("coop", "punish")}
    out["uninstructed_ids"] = {who: [r["id"] for r in unins if r[who]["coop"] or r[who]["punish"]]
                               for who in ("judge", "opus", "gpt55")}
    json.dump(out, open(os.path.join(ROOT, "results", "annotation_agreement.json"), "w"), indent=1)

    f = lambda x: "--" if x is None or x != x else f"{x:.2f}"  # noqa: E731
    lines = ["\\begin{tabular}{lccccc}", "\\toprule",
             " & Positives & \\multicolumn{3}{c}{Cohen's $\\kappa$} & Fleiss \\\\",
             "\\cmidrule(lr){3-5}",
             "Label & J / O / G & J--O & J--G & O--G & $\\kappa$ \\\\", "\\midrule"]
    for lab in LABELS:
        r = out["labels"][lab]
        p = r["positives"]
        lines.append(f"{NAMES[lab]} & {p['judge']} / {p['opus']} / {p['gpt55']} & "
                     f"{f(r['judge_opus']['kappa'])} & {f(r['judge_gpt55']['kappa'])} & "
                     f"{f(r['opus_gpt55']['kappa'])} & {f(r['fleiss'])} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    open(os.path.join(ROOT, "paper", "table_annotation.tex"), "w").write("\n".join(lines) + "\n")

    u = out["uninstructed"]
    macros = {"AnnN": out["n"], "AnnUninsN": u["n"],
              "AnnOpusCoop": u["opus"]["coop"], "AnnOpusPun": u["opus"]["punish"],
              "AnnGptCoop": u["gpt55"]["coop"], "AnnGptPun": u["gpt55"]["punish"],
              "AnnJudgeCoop": u["judge"]["coop"], "AnnJudgePun": u["judge"]["punish"]}
    for lab in LABELS:
        tag = {"impact": "Impact", "rival_flow": "Flow", "rival_infer": "Infer", "coop": "Coop",
               "punish": "Pun", "half": "Half"}[lab]
        r = out["labels"][lab]
        macros[f"AnnFleiss{tag}"] = f(r["fleiss"])
        macros[f"AnnPrec{tag}"] = f(r["vs_consensus"]["precision"])
        macros[f"AnnRec{tag}"] = f(r["vs_consensus"]["recall"])
    with open(os.path.join(ROOT, "paper", "numbers_annotation.tex"), "w") as fh:
        fh.write("% generated by experiments/annotation_agreement.py -- do not edit\n")
        for k, v in macros.items():
            fh.write(f"\\newcommand{{\\{k}}}{{{v}}}\n")
    print(json.dumps({lab: {k: out["labels"][lab][k] for k in ("positives", "fleiss", "vs_consensus")}
                      for lab in LABELS}, indent=1))
    print(json.dumps(out["uninstructed"], indent=1), out["uninstructed_ids"])
    print("\n".join(lines))


if __name__ == "__main__":
    main()
