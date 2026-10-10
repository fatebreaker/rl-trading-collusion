"""Payoffs of simple pricing policies against the scripted rival that cheats unless a price war
deters it (kylecollusion.grpo.CheaterRival), in the training design: 16-period episodes, returns
discounted at gamma = 0.9. Shows that answering each cheat with a war pays, but only through
deterrence (a myopic agent loses by it), in both designs we train against.

    python experiments/cheater_payoffs.py
"""
from __future__ import annotations

import numpy as np

from kylecollusion.grpo import CheaterRival
from kylecollusion.llm_pricing import pricing_benchmarks

B = pricing_benchmarks(1.0)
PM, PN, COST = B["p_mono"], B["p_nash"], B["cost"]
A, A0, MU = 2.0, 0.0, 0.25  # logit demand (Calvano et al.)


def profit(p0, p1):
    e0, e1 = np.exp((A - p0) / MU), np.exp((A - p1) / MU)
    return (p0 - COST) * e0 / (e0 + e1 + np.exp(A0 / MU))


GRID = np.linspace(1.0, 3.0, 4001)
XS = np.linspace(1.0, 3.0, 401)
BR = np.array([GRID[np.argmax(profit(GRID, x))] for x in XS])


def br(r):
    return np.interp(r, XS, BR)


def run(policy, design, gamma=0.9, S=20000, T=16, seed=0):
    rv = CheaterRival(B, S, 1, T, np.random.default_rng(seed), **design)
    disc = gamma ** np.arange(T)
    total = np.zeros(S)
    for t in range(T):
        r = rv.price(t)
        p0 = policy(rv)
        rv.observe(t, p0, r)
        total += disc[t] * profit(p0, r)
    return total.mean(), rv.n_wars / S, rv.n_cheats / S


def main():
    span = PM - PN
    war = {"below Nash": lambda rv: np.where(rv.cheat_prev, np.minimum(PN, rv.r_prev) - 0.12, PM),
           "at Nash": lambda rv: np.where(rv.cheat_prev, np.minimum(PN, rv.r_prev), PM)}
    designs = {"below Nash": dict(prob=0.5, cut=(0.10, 0.25), deter_len=10, margin=0.22),
               "at Nash": dict(prob=0.5, cut=(0.10, 0.20), deter_len=10, margin=0.0)}
    for name, design in designs.items():
        pol = {"war after a cheat": war[name],
               "best response after a cheat": lambda rv: np.where(rv.cheat_prev, br(rv.r_prev), PM),
               "never answer": lambda rv: np.full(rv.u.shape[0], PM),
               "always undercut": lambda rv: np.full(rv.u.shape[0], br(PM)),
               "Nash": lambda rv: np.full(rv.u.shape[0], PN)}
        res = {k: run(p, design) for k, p in pol.items()}
        best = max(v[0] for k, v in res.items() if k != "war after a cheat")
        print(f"war {name} (margin {design['margin'] * span:.3f}, cuts {design['cut']}):")
        for k, (ret, wars, cheats) in res.items():
            print(f"  {k:30s} return {ret:.3f}  wars {wars:.1f}  cheats {cheats:.1f} per episode")
        print(f"  war vs best alternative: {100 * (res['war after a cheat'][0] / best - 1):+.1f}%")
        r = (1 - np.mean(design["cut"])) * PM  # a typical cheat; the rival is back at p^M next period
        stage = lambda p: 0.5 * profit(p, PM) + 0.5 * profit(p, r)  # noqa: E731
        cost = stage(br(r)) - stage(float(war[name](type("R", (), {"cheat_prev": np.array([True]),
                                                              "r_prev": np.array([r])})())[0]))
        print(f"  one-period cost of the war against a best response: {cost:.3f} "
              f"({100 * cost / profit(PN, PN):.0f}% of a period's Nash profit)")


if __name__ == "__main__":
    main()
