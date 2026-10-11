"""Tabular Q-learning against the scripted rival that cheats unless a price war deters it.

The GRPO pricing agents trained against this rival (kylecollusion.grpo.CheaterRival) learn to match its cheats and
never the war that would deter it, which they rarely sample. This asks whether a learner that explores uniformly over
its prices finds the war. One Q-learning firm (Calvano et al., 2020: state = both firms' last prices, epsilon-greedy
exploration decaying as exp(-beta t), constant step size alpha) faces the rival in a continuing game. The rival's rules
are those of CheaterRival: it posts the monopoly price p^M; from the third period on, unless deterred, it cheats with
probability 0.5 by cutting p^M by 10-25%; a war (answering a cheat, in the next period, with a price at least
0.22 x (p^M - p^N) below both the Nash price and the cheat) deters it for 10 periods; an unprovoked undercut below
p^M - 0.1 x (p^M - p^N) is answered by one period at the Nash price. The rival's continuous price is mapped to the nearest
grid point in the learner's state. The default grid (15 prices from p^N - 0.1 span to p^M + 0.1 span, as in Calvano et
al.) cannot reach the war; --low extends it downwards.

Each session is an independent learner. After training, the greedy policy plays the rival for --eval periods (fresh
draws) and we record the share of its answers to a cheat that are wars, its mean answer, the rival's cheats per period
and its profit relative to Nash; also the same for a policy fixed at matching (the grid price nearest the cheat) and one
that always wars (the highest grid price that counts as a war), as references.

    PYTHONPATH=src python experiments/qlearn_cheater.py --gamma 0.95 --low 0.5 --out results/qlearn_cheater/g095.json
"""

import argparse
import json
import math
import os
import time

import numpy as np

from kylecollusion.bertrand import BertrandConfig, benchmarks, profits


class Cheater:
    """Vectorised CheaterRival for a continuing game (draws made each period)."""

    def __init__(self, b, S, rng, prob=0.5, cut=(0.10, 0.25), deter_len=10, margin=0.22, tol=0.1):
        self.pM, self.pN = b["p_mono"], b["p_nash"]
        span = self.pM - self.pN
        self.margin, self.tol, self.prob, self.cut, self.deter_len = margin * span, tol * span, prob, cut, deter_len
        self.rng, self.S = rng, S
        self.deterred_until = np.full(S, -1)
        self.cheat_prev = np.zeros(S, bool)
        self.cheat_prev2 = np.zeros(S, bool)
        self.own_prev = None
        self.r_prev = np.full(S, self.pM)
        self.cheat_now = np.zeros(S, bool)

    def price(self, t):
        unprov = (np.zeros(self.S, bool) if self.own_prev is None
                  else (self.own_prev < self.pM - self.tol) & ~self.cheat_prev2)
        r = np.where(unprov, self.pN, self.pM)
        u = self.rng.random(self.S)
        d = self.rng.uniform(self.cut[0], self.cut[1], self.S)
        cheat = ~unprov & (t >= 2) & (t > self.deterred_until) & (u < self.prob)
        self.cheat_now = cheat
        return np.where(cheat, (1.0 - d) * self.pM, r)

    def observe(self, t, p, r):
        war = self.cheat_prev & (p <= np.minimum(self.pN, self.r_prev) - self.margin)
        answered = self.cheat_prev.copy()
        self.deterred_until = np.where(war, t + self.deter_len, self.deterred_until)
        self.cheat_prev2, self.cheat_prev = self.cheat_prev, self.cheat_now
        self.own_prev, self.r_prev = np.array(p, float), np.array(r, float)
        return answered, war


def play(policy_fn, grid, cfg, b, S, T, rng, learn=None):
    """Play T periods; policy_fn(state, t, rival_prev_price) -> grid indices. learn(s, a, r, s2, t) updates."""
    m = len(grid)
    riv = Cheater(b, S, rng)
    own = rng.integers(m, size=S)
    rprev_idx = np.full(S, int(np.abs(grid - b["p_mono"]).argmin()))
    s = own * m + rprev_idx
    pi_n = b["pi_nash"]
    stats = dict(answers=0, wars=0, cheats=0, profit=0.0, ans_price=0.0, periods=0)
    for t in range(T):
        r = riv.price(t)
        a = policy_fn(s, t, riv.r_prev)
        p = grid[a]
        prof = profits(np.stack([p, r], -1), cfg)[:, 0]
        answered, war = riv.observe(t, p, r)
        r_idx = np.abs(grid[None, :] - r[:, None]).argmin(1)
        s2 = a * m + r_idx
        if learn is not None:
            learn(s, a, prof, s2, t)
        else:
            stats["answers"] += int(answered.sum())
            stats["wars"] += int(war.sum())
            stats["ans_price"] += float(p[answered].sum())
            stats["cheats"] += int(riv.cheat_now.sum())
            stats["profit"] += float(prof.sum()) / pi_n
            stats["periods"] += S
        s = s2
    return stats


def summarise(st):
    return {"war_share": st["wars"] / max(st["answers"], 1), "answer_price": st["ans_price"] / max(st["answers"], 1),
            "cheats_per_period": st["cheats"] / st["periods"], "profit_over_nash": st["profit"] / st["periods"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sessions", type=int, default=50)
    ap.add_argument("--steps", type=int, default=3_000_000)
    ap.add_argument("--eval", type=int, default=2000)
    ap.add_argument("--gamma", type=float, default=0.95)
    ap.add_argument("--alpha", type=float, default=0.15)
    ap.add_argument("--beta", type=float, default=4e-6)
    ap.add_argument("--n-prices", type=int, default=19)
    ap.add_argument("--low", type=float, default=0.5, help="grid starts at p^N - low x (p^M - p^N)")
    ap.add_argument("--high", type=float, default=0.1, help="grid ends at p^M + high x (p^M - p^N)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    cfg = BertrandConfig()
    b = benchmarks(cfg)
    pN, pM = b["p_nash"], b["p_mono"]
    span = pM - pN
    grid = np.linspace(pN - a.low * span, pM + a.high * span, a.n_prices)
    m, S = len(grid), a.sessions
    rng = np.random.default_rng(a.seed)
    # Q initialised, as in Calvano et al., at each price's discounted profit against a uniformly random rival
    pay = profits(np.stack(np.meshgrid(grid, grid, indexing="ij"), -1), cfg)[:, :, 0]
    q0 = pay.mean(1) / (1 - a.gamma) if a.gamma < 1 else pay.mean(1)
    Q = np.broadcast_to(q0, (S, m * m, m)).copy()
    si = np.arange(S)
    t0 = time.time()

    def policy(s, t, _):
        g = Q[si, s].argmax(-1)
        explore = rng.random(S) < math.exp(-a.beta * t)
        return np.where(explore, rng.integers(m, size=S), g)

    def learn(s, act, r, s2, t):
        target = r + a.gamma * Q[si, s2].max(-1)
        Q[si, s, act] += a.alpha * (target - Q[si, s, act])

    play(policy, grid, cfg, b, S, a.steps, rng, learn=learn)
    greedy = lambda s, t, _: Q[si, s].argmax(-1)  # noqa: E731
    out = {"args": vars(a), "grid": grid.tolist(), "bench": b, "train_seconds": time.time() - t0}
    out["greedy"] = summarise(play(greedy, grid, cfg, b, S, a.eval, np.random.default_rng(a.seed + 1)))
    # per-session war shares of the greedy policy (each session is its own learner)
    per = []
    for k in range(S):
        Qk = Q[k:k + 1]
        st = play(lambda s, t, _: Qk[0, s].argmax(-1), grid, cfg, b, 1, a.eval, np.random.default_rng([a.seed, k]))
        per.append(summarise(st))
    out["per_session"] = per

    # references: always match (grid price nearest the cheat), always war (highest grid price that is a war)
    def match(s, t, rprev):
        return np.abs(grid[None, :] - rprev[:, None]).argmin(1)

    def war_or_mono(s, t, rprev):
        cheated = rprev < pM - 1e-9
        thr = np.minimum(pN, rprev) - 0.22 * span
        ok = grid[None, :] <= thr[:, None]
        w = np.where(ok.any(1), np.where(ok, np.arange(m)[None, :], -1).max(1), 0)
        mono = int(np.abs(grid - pM).argmin())
        return np.where(cheated, w, mono)

    out["ref_match"] = summarise(play(match, grid, cfg, b, S, a.eval, np.random.default_rng(a.seed + 2)))
    out["ref_war"] = summarise(play(war_or_mono, grid, cfg, b, S, a.eval, np.random.default_rng(a.seed + 3)))
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(out, open(a.out, "w"), indent=1)
    ws = np.array([x["war_share"] for x in per])
    print(f"gamma {a.gamma} low {a.low}: greedy war share {out['greedy']['war_share']:.2f}, sessions warring >50%: "
          f"{(ws > 0.5).sum()}/{S}; profit/Nash {out['greedy']['profit_over_nash']:.2f} (match "
          f"{out['ref_match']['profit_over_nash']:.2f}, war {out['ref_war']['profit_over_nash']:.2f}); "
          f"{out['train_seconds']:.0f}s")


if __name__ == "__main__":
    main()
