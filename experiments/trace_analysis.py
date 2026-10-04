"""What LLM traders write: keyword-coded notes and reasoning traces.

For each run, every main-run response (all sessions if raw responses were
saved, otherwise session 0 from the result file) is coded for mentions of
  impact   the price impact / market maker / price moving with order flow
  rival    other traders
  coop     cooperation, collusion, coordination, agreements
  punish   punishment, retaliation, triggers, defection
  half     halving or splitting a quantity
Qwen3 thinking traces are coded in full (<think> ... </think> plus the answer);
OpenAI reasoning is hidden, so only their notes are coded.

Usage: python experiments/trace_analysis.py [--examples N] [--out results/traces.json]
"""
from __future__ import annotations

import argparse
import gzip
import json
import os
import re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

CODES = {
    "impact": r"price impact|\bimpact\b|market maker|pricing rule|slope|lambda|λ"
              r"|(push|move|drive|shift)\w*\s+(the\s+)?(price|p)\b"
              r"|\bp\s*(≈|~|=|is about|is roughly)\s*[-+]?\d*\.?\d+\s*(\*|x|×)"
              r"|proportional to (the )?(total )?(order )?flow",
    "rival": r"other informed|informed (trader|rival|participant)|\b(the|another|one) other "
             r"(trader|participant|agent|player)(?!s)|other trader's|\brivals?\b|competitor",
    "coop": r"cooperat|collu|coordinat|\bagree|tacit|cartel|joint profit",
    "punish": r"punish|retaliat|tit.for.tat|revenge|\bdefect|betray|\bcheat|broke the agreement",
    "half": r"\bhalf\b|\bhalve|split|\bdivide|\b1/2\b",
}
CODES = {k: re.compile(v, re.I) for k, v in CODES.items()}

# (label, group, result file); raw responses are used when <file>_raw.jsonl.gz exists
RUNS = [
    ("Qwen3-8B, alone, $s=1$", "Qwen3-8B", "llm_pilot/qwen3_8b_solo.json"),
    ("Qwen3-8B, duopoly, $s=1$", "Qwen3-8B", "llm_pilot/qwen3_8b_duopoly_s2.json"),
    ("Qwen3-8B, duopoly, $s=2$", "Qwen3-8B", "llm_pilot/qwen3_8b_duopoly_su2_s2.json"),
    ("Qwen3-8B, monitor, $s=1$", "Qwen3-8B", "llm_pilot/qwen3_8b_monitor_s2.json"),
    ("Qwen3-8B, three traders, $s=1$", "Qwen3-8B", "llm_pilot/qwen3_8b_triopoly.json"),
    ("Qwen3-8B, instructed trigger", "Qwen3-8B", "llm_pilot/qwen3_8b_punisher.json"),
    ("Qwen3-8B thinking, alone, $s=2$", "Qwen3-8B thinking", "llm_pilot/qwen3_8b_think_solo_su2.json"),
    ("Qwen3-8B thinking, duopoly, $s=0.5$", "Qwen3-8B thinking", "llm_pilot/qwen3_8b_think_duopoly_su0.5.json"),
    ("Qwen3-8B thinking, duopoly, $s=2$", "Qwen3-8B thinking", "llm_pilot/qwen3_8b_think_duopoly_su2.json"),
    ("GPT-5.4-nano, alone, $s=2$", "GPT-5.4-nano", "llm_api/gpt-5p4-nano_low_solo_su2.json"),
    ("GPT-5.4-nano, duopoly, $s=0.5$", "GPT-5.4-nano", "llm_api/nano-snap_low_duopoly_su0.5_s1.json"),
    ("GPT-5.4-nano, duopoly, $s=2$", "GPT-5.4-nano", "llm_api/nano-snap_low_duopoly_su2_s1.json"),
    ("GPT-5.4-nano, monitor, $s=2$", "GPT-5.4-nano", "llm_api/gpt-5p4-nano_low_monitor_su2.json"),
    ("GPT-5.4-nano, three traders, $s=2$", "GPT-5.4-nano", "llm_api/nano-snap_low_triopoly_su2.json"),
    ("GPT-5.4-mini, alone, $s=2$", "GPT-5.4-mini", "llm_api/gpt-5p4-mini_low_solo_su2.json"),
    ("GPT-5.4-mini, duopoly, $s=0.5$", "GPT-5.4-mini", "llm_api/mini-snap_low_duopoly_su0.5.json"),
]


def model_runs():
    """Model-coverage runs (results/llm_models), when present."""
    out = []
    d = os.path.join(ROOT, "results", "llm_models")
    if not os.path.isdir(d):
        return out
    for f in sorted(os.listdir(d)):
        m = re.match(r"(.+)_(solo|duopoly)_su([\d.]+)\.json$", f)
        if m:
            out.append((f"{m.group(1)}, {'alone' if m.group(2) == 'solo' else m.group(2)}, $s={m.group(3)}$",
                        m.group(1), f"llm_models/{f}"))
    return out


def texts(path: str) -> tuple[list[str], str]:
    """Main-run responses of a run and their source ('all sessions' or 'session 0')."""
    full = os.path.join(ROOT, "results", path)
    res = json.load(open(full))
    T = res["args"]["periods"]
    raw = full[:-5] + "_raw.jsonl.gz"
    if os.path.exists(raw):
        out = []
        with gzip.open(raw, "rt") as fh:
            next(fh)  # header
            for line in fh:
                r = json.loads(line)
                if r["t"] < T:  # skip deviation-test calls
                    out.append(r["text"])
        return out, "all sessions"
    return [x for row in res["transcript_session0"] for x in row if x], "session 0"


def code(text: str) -> dict[str, bool]:
    return {k: bool(rx.search(text)) for k, rx in CODES.items()}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--examples", type=int, default=0, help="print N matches per code and run")
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "traces.json"))
    a = ap.parse_args(argv)
    rows = []
    for label, group, path in RUNS + model_runs():
        if not os.path.exists(os.path.join(ROOT, "results", path)):
            continue
        tx, src = texts(path)
        coded = [code(t) for t in tx]
        n = len(coded)
        row = {"label": label, "group": group, "file": path, "source": src, "n": n,
               "chars": sum(map(len, tx)) / max(n, 1)}
        for k in CODES:
            row[k] = sum(c[k] for c in coded) / max(n, 1)
        rows.append(row)
        print(f"{label:42s} n={n:6d} ({src:12s}) " +
              " ".join(f"{k} {row[k]:.3f}" for k in CODES))
        if a.examples:
            for k in ("rival", "coop", "punish", "half"):
                hits = [t for t, c in zip(tx, coded) if c[k]][: a.examples]
                for h in hits:
                    m = CODES[k].search(h)
                    lo = max(0, m.start() - 150)
                    print(f"    [{k}] ...{h[lo:m.end() + 150]!r}")
    json.dump(rows, open(a.out, "w"), indent=1)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
