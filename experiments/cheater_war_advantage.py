"""Would GRPO reward a price war if it sampled one? For the scripted rival that cheats unless a war deters it
(kylecollusion.grpo.CheaterRival), compare two answers to the same cheat, everything else equal (common random
numbers): matching the cheat, as the trained policies do, and a war. The policy matches every other cheat. Reports,
by the period of the answer, the mean difference in the answer's discounted return-to-go (gamma = 0.9, to the end
of the 16-period training episode), in units of a period's Nash profit: the sign of the advantage GRPO would give a
sampled war against the matching answers of its group.

    python experiments/cheater_war_advantage.py
"""
from __future__ import annotations

import numpy as np

from kylecollusion.grpo import CheaterRival
import cheater_payoffs as cp  # noqa: E402  (profit, best response and benchmarks of the training design)

PM, PN = cp.PM, cp.PN
PI_N = cp.profit(PN, PN)


def run(design, war_at, war, gamma=0.9, S=40000, T=16, seed=0):
    """Discounted return-to-go from period war_at, for sessions with a cheat to answer there; the answer in
    period war_at is a war if `war`, else a match; every other cheat is matched."""
    rv = CheaterRival(cp.B, S, 1, T, np.random.default_rng(seed), **design)
    disc = gamma ** np.arange(T)
    rtg = np.zeros(S)
    answered = None
    for t in range(T):
        r = rv.price(t)
        p = np.where(rv.cheat_prev, rv.r_prev, PM)  # match the last cheat, else the monopoly price
        if t == war_at:
            answered = rv.cheat_prev.copy()
            if war:
                p = np.where(rv.cheat_prev, np.minimum(PN, rv.r_prev) - 0.12, p)
        rv.observe(t, p, r)
        if t >= war_at:
            rtg += disc[t - war_at] * cp.profit(p, r)
    return rtg, answered


def main():
    designs = {"first design": dict(prob=0.5, cut=(0.10, 0.25), deter_len=10, margin=0.22),
               "cheats more (probability 0.8)": dict(prob=0.8, cut=(0.10, 0.25), deter_len=10, margin=0.22),
               "cheats more and deeper": dict(prob=0.8, cut=(0.20, 0.40), deter_len=14, margin=0.22)}
    for name, design in designs.items():
        print(f"{name}: advantage of a war over matching, by the period it answers a cheat (Nash profits)")
        row = []
        for t in range(3, 16):
            a, ans = run(design, t, True)
            b, _ = run(design, t, False)
            d = (a - b)[ans] / PI_N
            row.append(f"t={t}: {d.mean():+.2f}")
        print("  " + "  ".join(row))


if __name__ == "__main__":
    main()
