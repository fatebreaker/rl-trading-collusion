"""Print a Markdown table of results for one or more experiment folders.

    python experiments/tabulate.py results/exp3 results/exp4
"""

from __future__ import annotations

import glob
import json
import os
import sys


def fmt(summary: dict, key: str, digits: int = 2) -> str:
    s = summary.get(key)
    if not s or s.get("n", 0) == 0:
        return "–"
    return f"{s['mean']:+.{digits}f} ± {s['ci95']:.{digits}f}"


def main(dirs: list[str]) -> None:
    print("| run | memory | Δ intensity | Δ info | Δ profit | on-path shift | rival dβ lag 1 | deviator gain |")
    print("|---|---|---|---|---|---|---|---|")
    for d in dirs:
        for f in sorted(glob.glob(os.path.join(d, "*.json"))):
            r = json.load(open(f))
            s = r["summary"]
            imp = r.get("impulse")
            rival = gain = "–"
            if imp and imp["n_events"]:
                rival = f"{imp['d_beta_rival'][1]:+.3f} ± {imp['d_beta_rival_ci95'][1]:.3f}"
                gain = f"{imp['cum_gain_dev']:+.3f} ± {imp['cum_gain_dev_ci95']:.3f}"
            name = os.path.join(os.path.basename(d), os.path.basename(f)[:-5])
            print(
                f"| {name} | {r['config']['memory']} | {fmt(s, 'delta_intensity')} | "
                f"{fmt(s, 'delta_info')} | {fmt(s, 'delta_profit')} | "
                f"{fmt(s, 'order_shift_onpath')} | {rival} | {gain} |"
            )


if __name__ == "__main__":
    main(sys.argv[1:] or ["results"])
