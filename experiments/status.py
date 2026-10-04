"""One-line progress per running experiment; with --changes, print only lines
that changed since the last call (state kept in a file). Used for monitoring:

  while true; do python experiments/status.py --changes; sleep 120; done
"""

from __future__ import annotations

import glob
import json
import os
import re
import sys

ROOT = os.path.join(os.path.dirname(__file__), "..", "results")
STATE = os.path.join(ROOT, ".status_state.json")
ERR = re.compile(r"OutOfMemoryError|Engine core initialization failed|Killed")


def grpo_lines():
    out = {}
    for log in sorted(glob.glob(os.path.join(ROOT, "grpo", "*", "log.jsonl"))):
        run = os.path.basename(os.path.dirname(log))
        rows = [json.loads(l) for l in open(log) if l.strip()]
        if not rows:
            continue
        d = rows[-1]
        bucket = d["iteration"] // 25 * 25  # report every 25 iterations
        b = sum(r["agg_intensity"] for r in rows[-5:]) / len(rows[-5:])
        line = (f"grpo/{run}: it {d['iteration']} beta(last5) {b:.3f} "
                f"x Nash {d['intensity_over_nash']:.2f} zero-adv {d.get('zero_adv_share', float('nan')):.2f}")
        if "policy_intensity" in d:  # scripted trigger rival
            pb = sum(r["policy_intensity"] for r in rows[-5:]) / len(rows[-5:])
            ps = sum(r["punished_share"] for r in rows[-5:]) / len(rows[-5:])
            line += f" | policy {pb:.3f} (coop {d['coop']:.2f}, BR {d['policy_br_to_coop']:.2f}) punished {ps:.2f}"
        out[f"grpo/{run}"] = (bucket, line)
    return out


def pilot_lines():
    out = {}
    for log in sorted(glob.glob(os.path.join(ROOT, "llm_pilot", "*.log"))
                      + glob.glob(os.path.join(ROOT, "grpo_audit", "*.log"))
                      + glob.glob(os.path.join(ROOT, "llm_base", "*.log"))
                      + glob.glob(os.path.join(ROOT, "llm_api", "*.log"))
                      + glob.glob(os.path.join(ROOT, "llm_models", "*.log"))
                      + glob.glob(os.path.join(ROOT, "grpo", "*.log"))):
        run = os.path.relpath(log, ROOT)[:-4]
        text = open(log, errors="ignore").read()
        js = log[:-4] + ".json"
        if os.path.exists(js):
            m = re.search(r'^\{"agg_intensity": ([-0-9.e]+)', text, re.M)
            out[run] = ("done", f"{run}: DONE agg {float(m.group(1)):.3f}" if m else f"{run}: DONE")
        elif ERR.search(text):
            out[run] = ("error", f"{run}: ERROR {ERR.search(text).group(0)}")
    return out


def main():
    cur = {**grpo_lines(), **pilot_lines()}
    if "--changes" not in sys.argv:
        for _, line in cur.values():
            print(line)
        return
    prev = json.load(open(STATE)) if os.path.exists(STATE) else None
    if prev is not None:
        for k, (key, line) in cur.items():
            if k not in prev or prev[k] != key:
                print(line, flush=True)
    json.dump({k: v[0] for k, v in cur.items()}, open(STATE, "w"))


if __name__ == "__main__":
    main()
