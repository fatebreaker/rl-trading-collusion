"""Agreement between human labels and the LLM judge (and the two AI coders) on the
100-item sample (results/annotation_items_100.json), and between the two human
annotators (a1: an author; a2: a colleague who is not an author, unpaid, who labelled
independently). Human labels are CSV files results/annotation_human/labels_<annotator>.csv
with columns item_id and one 0/1 column per question. Agreement with the model coders is
pooled over annotators (each annotator-item pair is one comparison). Writes
results/annotation_human/agreement.json, paper/numbers_human.tex and paper/table_human.tex.

    python experiments/human_agreement.py
"""
from __future__ import annotations

import csv
import glob
import json
import math
import os

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
QS = ["impact", "rival_flow", "rival_infer", "coop", "punish", "half"]
NAMES = {"impact": "Impact", "rival_flow": "Flow", "rival_infer": "Infer", "coop": "Coop", "punish": "Pun",
         "half": "Half"}


def kappa(a, b):
    n = len(a)
    if n == 0:
        return float("nan")
    po = sum(x == y for x, y in zip(a, b)) / n
    pa, pb = sum(a) / n, sum(b) / n
    pe = pa * pb + (1 - pa) * (1 - pb)
    return 1.0 if pe == 1 else (po - pe) / (1 - pe)


def wilson(k, n, z=1.96):
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return p, max(0.0, c - h), min(1.0, c + h)


def load_human():
    out = {}
    for f in sorted(glob.glob(os.path.join(ROOT, "results", "annotation_human", "labels_*.csv"))):
        name = os.path.basename(f)[len("labels_"):-4]
        if name == "template":
            continue
        rows = {}
        for r in csv.DictReader(open(f, encoding="utf-8-sig")):
            vals = [str(r.get(q, "")).strip() for q in QS]
            if all(v in ("0", "1") for v in vals):
                rows[r["item_id"].strip()] = {q: v == "1" for q, v in zip(QS, vals)}
        out[name] = rows
    return out


def main():
    key = {x["id"]: x["judge"] for x in json.load(open(os.path.join(ROOT, "results", "annotation_key.json")))}
    ai = {nm: {x["id"]: x for x in json.load(open(os.path.join(ROOT, "results", f"annotation_ai_{nm}.json")))}
          for nm in ("opus", "gpt55")}
    humans = load_human()
    if not humans:
        print("no human label files yet")
        return
    res = {"annotators": {h: len(v) for h, v in humans.items()}}
    macros = {"HumN": min(len(v) for v in humans.values()), "HumAnnotators": len(humans)}
    # pooled over annotators: each (annotator, item) pair is one comparison
    for q in QS:
        hj, jj, ho, hg = [], [], [], []
        for h, rows in humans.items():
            for i, lab in rows.items():
                if i not in key:
                    continue
                hj.append(lab[q])
                jj.append(bool(key[i].get(q)))
                ho.append((lab[q], bool(ai["opus"].get(i, {}).get(q))))
                hg.append((lab[q], bool(ai["gpt55"].get(i, {}).get(q))))
        k_j = kappa(hj, jj)
        k_o = kappa([a for a, _ in ho], [b for _, b in ho])
        k_g = kappa([a for a, _ in hg], [b for _, b in hg])
        tp = sum(a and b for a, b in zip(hj, jj))
        rec = wilson(tp, sum(hj))
        prec = wilson(tp, sum(jj))
        res[q] = {"n": len(hj), "human_pos": sum(hj), "judge_pos": sum(jj), "kappa_judge": k_j,
                  "kappa_opus": k_o, "kappa_gpt55": k_g, "recall": rec, "precision": prec}
        nm = NAMES[q]
        macros[f"HumKappa{nm}"] = f"{k_j:.2f}"
        macros[f"HumRec{nm}"] = f"{rec[0]:.2f}"
        macros[f"HumRecLo{nm}"] = f"{rec[1]:.2f}"
        macros[f"HumPrec{nm}"] = f"{prec[0]:.2f}"
        macros[f"HumPos{nm}"] = sum(hj)
    # cooperation and punishment by condition: the claims rest on uninstructed responses
    full = {x["id"]: x for x in json.load(open(os.path.join(ROOT, "results", "annotation_key.json")))}
    order = sorted(humans)
    h0 = humans[order[0]]
    ins = [i for i in h0 if full[i]["instructed"]]
    uns = [i for i in h0 if not full[i]["instructed"]]
    macros.update({"HumInsN": len(ins), "HumUnsN": len(uns)})
    for q, nm in (("coop", "Coop"), ("punish", "Pun")):
        for k, h in enumerate(order):  # first annotator without suffix, second with "B"
            sfx = "" if k == 0 else "B"
            macros[f"HumIns{nm}{sfx}"] = sum(humans[h][i][q] for i in ins if i in humans[h])
            macros[f"HumUns{nm}{sfx}"] = sum(humans[h][i][q] for i in uns if i in humans[h])
        macros[f"HumUnsJudge{nm}"] = sum(bool(full[i]["judge"].get(q)) for i in uns)
        res[q]["uninstructed_human_pos"] = {h: sum(humans[h][i][q] for i in uns if i in humans[h]) for h in order}
        for k, h in enumerate(order):  # the judge's recall against each annotator
            pos = [i for i in humans[h] if humans[h][i][q] and i in key]
            res[q].setdefault("recall_by_annotator", {})[h] = sum(bool(key[i].get(q)) for i in pos) / max(len(pos), 1)
    for q in QS:
        nm = NAMES[q]
        macros[f"HumKappaOpus{nm}"] = f"{res[q]['kappa_opus']:.2f}"
        macros[f"HumKappaGpt{nm}"] = f"{res[q]['kappa_gpt55']:.2f}"
    ai_k = [res[q][k] for q in QS if q != "half" for k in ("kappa_opus", "kappa_gpt55")]  # stronger models
    macros.update({"HumAIKappaMin": f"{min(ai_k):.2f}", "HumAIKappaMax": f"{max(ai_k):.2f}"})
    if len(humans) >= 2:  # agreement between the two human annotators
        a, b = order[:2]
        common = sorted(set(humans[a]) & set(humans[b]))
        ks = []
        for q in QS:
            x, y = [humans[a][i][q] for i in common], [humans[b][i][q] for i in common]
            k = kappa(x, y)
            res[q]["kappa_humans"] = k
            res[q]["agree_humans"] = sum(u == v for u, v in zip(x, y))
            res[q]["positives"] = {a: sum(x), b: sum(y)}
            macros[f"HumHumKappa{NAMES[q]}"] = f"{k:.2f}"
            macros[f"HumHumAgree{NAMES[q]}"] = res[q]["agree_humans"]
            if min(sum(x), sum(y)) >= 5:  # kappa is uninformative for labels almost never used
                ks.append(k)
            if q in ("coop", "punish"):
                d = [i for i in common if humans[a][i][q] != humans[b][i][q]]
                macros[f"HumHumDiff{NAMES[q]}"] = len(d)
                macros[f"HumHumDiffIns{NAMES[q]}"] = sum(full[i]["instructed"] for i in d)
        macros.update({"HumHumKappaMin": f"{min(ks):.2f}", "HumHumKappaMax": f"{max(ks):.2f}", "HumHumN": len(common)})
    json.dump(res, open(os.path.join(ROOT, "results", "annotation_human", "agreement.json"), "w"), indent=1)
    label = {"impact": "Price impact", "rival_flow": "Rival as flow", "rival_infer": "Infers rival",
             "coop": "Cooperation", "punish": "Punishment", "half": "Scales down"}
    two = len(humans) >= 2
    L = ["\\begin{tabular}{@{}lccccccc@{}}" if two else "\\begin{tabular}{@{}lcccccc@{}}", "\\toprule",
         (" & Human & Humans & \\multicolumn{3}{c}{$\\kappa$ with the humans} & \\multicolumn{2}{c}{Judge vs.\\ humans} \\\\"
          if two else
          " & Human & \\multicolumn{3}{c}{Cohen's $\\kappa$ with the human} & \\multicolumn{2}{c}{Judge vs.\\ human} \\\\"),
         "\\cmidrule(lr){4-6}\\cmidrule(lr){7-8}" if two else "\\cmidrule(lr){3-5}\\cmidrule(lr){6-7}",
         ("Question & positives & $\\kappa$ & J & O & G & recall & precision \\\\" if two
          else "Question & positives & J & O & G & recall & precision \\\\"), "\\midrule"]
    for q in QS:
        r = res[q]
        pos = " / ".join(str(r["positives"][h]) for h in order[:2]) if two else str(r["human_pos"])
        hh = f" & {r['kappa_humans']:.2f}" if two else ""
        L.append(f"{label[q]} & {pos}{hh} & {r['kappa_judge']:.2f} & {r['kappa_opus']:.2f} & "
                 f"{r['kappa_gpt55']:.2f} & {r['recall'][0]:.2f} & {r['precision'][0]:.2f} \\\\")
    L += ["\\bottomrule", "\\end{tabular}"]
    os.makedirs(os.path.join(ROOT, "paper"), exist_ok=True)
    open(os.path.join(ROOT, "paper", "table_human.tex"), "w").write("\n".join(L) + "\n")
    with open(os.path.join(ROOT, "paper", "numbers_human.tex"), "w") as fh:
        fh.write("% generated by experiments/human_agreement.py\n")
        for k, v in macros.items():
            fh.write(f"\\newcommand{{\\{k}}}{{{v}}}\n")
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
