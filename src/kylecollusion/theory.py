"""Closed-form benchmarks for the one-period Kyle (1985) market with several
informed traders.

Setup (per period, v_bar = 0):
    v ~ (0, sigma_v^2)        asset value, seen by informed traders
    x_i = beta_i * v          informed order of trader i
    u ~ N(0, sigma_u^2)       noise-trader order
    y = sum_i x_i + u         aggregate order flow, seen by the market maker
    p = lambda * y            linear market-maker pricing rule
    profit_i = (v - p) x_i

The market maker sets lambda = Cov(v, y) / Var(y), i.e. the OLS / linear
Bayesian rule. Everything below depends only on second moments, so the
benchmarks hold exactly for any value distribution as long as the market
maker is restricted to linear pricing — which is what the simulator does.

Two reference points are used to score learned behaviour:
  * Nash: every informed trader best-responds (Cournot-style competition).
  * Collusive: the learning traders jointly act as a monopolist informed
    trader, facing a market maker that rationally prices their joint strategy.

Optional "passive" traders play the Nash strategy of the full game and never
learn. They model non-algorithmic informed traders and are one of the market
design interventions.

Information-insensitive investors (Dou, Goldstein & Ji 2025): a mass xi of
investors with demand z = -xi (p - v_bar) trades against the price without
learning from it. The market maker then trades off pricing error against
inventory, p = argmin theta (p - E[v|y])^2 + (y + z)^2, which gives the linear
rule p = v_bar + lambda y with

    lambda = (theta * lambda_B + xi) / (theta + xi^2),

where lambda_B = Cov(v, y) / Var(y) is the Bayesian price impact. xi = 0 is the
standard Kyle market (lambda = lambda_B) and keeps the closed forms below; for
xi > 0 the Nash and collusive intensities are solved numerically.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class Benchmarks:
    n_informed: int
    n_passive: int
    sigma_v: float
    sigma_u: float
    # per-learner trading intensity (order = beta * v)
    beta_nash: float
    beta_coll: float
    # aggregate learner intensity B_L = sum of learner betas
    agg_nash: float
    agg_coll: float
    lam_nash: float
    lam_coll: float
    # expected profit per learning trader per period
    profit_nash: float
    profit_coll: float
    # price informativeness R^2 = Corr(p, v)^2
    info_nash: float
    info_coll: float

    def as_dict(self) -> dict:
        return asdict(self)


def market_lambda(
    total_beta: float, sigma_v: float, sigma_u: float, xi: float = 0.0, theta: float = 0.1
) -> float:
    """Rational linear price impact given aggregate informed intensity."""
    lam_b = total_beta * sigma_v**2 / (total_beta**2 * sigma_v**2 + sigma_u**2)
    if xi == 0.0:
        return lam_b
    return (theta * lam_b + xi) / (theta + xi**2)


def informativeness(total_beta: float, sigma_v: float, sigma_u: float) -> float:
    s = total_beta**2 * sigma_v**2
    return s / (s + sigma_u**2)


def learner_profit(beta_i: float, total_beta: float, lam: float, sigma_v: float) -> float:
    """E[(v - lam*y) * beta_i * v] with y = total_beta * v + u."""
    return beta_i * sigma_v**2 * (1.0 - lam * total_beta)


def _nash_beta(n: int, sigma_v: float, sigma_u: float, xi: float, theta: float) -> float:
    """Symmetric Nash beta solving beta = 1 / ((n+1) lambda(n beta)).

    h(beta) = (n+1) beta lambda(n beta) - 1 is strictly increasing, so the root
    is unique and bisection is safe."""
    def h(b):
        return (n + 1) * b * market_lambda(n * b, sigma_v, sigma_u, xi, theta) - 1.0

    lo, hi = 0.0, 1.0
    while h(hi) < 0:
        hi *= 2.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if h(mid) < 0 else (lo, mid)
    return 0.5 * (lo + hi)


def _collusive_agg(
    passive_total: float, agg_nash: float, sigma_v: float, sigma_u: float, xi: float, theta: float
) -> float:
    """Learners' joint profit-maximising aggregate intensity (numerical)."""
    def joint(bl):
        tot = bl + passive_total
        return learner_profit(bl, tot, market_lambda(tot, sigma_v, sigma_u, xi, theta), sigma_v)

    top = 4.0 * max(agg_nash, 1e-12)
    grid = [top * k / 4000 for k in range(4001)]
    k = max(range(len(grid)), key=lambda i: joint(grid[i]))
    lo, hi = grid[max(k - 1, 0)], grid[min(k + 1, len(grid) - 1)]
    g = (math.sqrt(5) - 1) / 2
    for _ in range(200):  # golden-section refinement inside the best bracket
        a, b = hi - g * (hi - lo), lo + g * (hi - lo)
        if joint(a) > joint(b):
            hi = b
        else:
            lo = a
    return 0.5 * (lo + hi)


def kyle_benchmarks(
    n_informed: int,
    sigma_v: float,
    sigma_u: float,
    n_passive: int = 0,
    xi: float = 0.0,
    theta: float = 0.1,
) -> Benchmarks:
    if n_informed < 1:
        raise ValueError("need at least one learning informed trader")
    n = n_informed + n_passive
    if xi > 0.0:
        return _numeric_benchmarks(n_informed, n_passive, sigma_v, sigma_u, xi, theta)

    # Symmetric Nash of the n-trader game (Kyle with n informed insiders).
    beta_n = sigma_u / (math.sqrt(n) * sigma_v)
    lam_n = math.sqrt(n) * sigma_v / ((n + 1) * sigma_u)
    passive_total = n_passive * beta_n

    # Learners jointly maximise B_L * sigma_v^2 * sigma_u^2 / ((B_L+B_P)^2 sigma_v^2 + sigma_u^2)
    # First-order condition gives B_L = sqrt(B_P^2 + sigma_u^2 / sigma_v^2).
    agg_coll = math.sqrt(passive_total**2 + (sigma_u / sigma_v) ** 2)
    total_coll = agg_coll + passive_total
    lam_c = market_lambda(total_coll, sigma_v, sigma_u)

    total_nash = n * beta_n
    return Benchmarks(
        n_informed=n_informed,
        n_passive=n_passive,
        sigma_v=sigma_v,
        sigma_u=sigma_u,
        beta_nash=beta_n,
        beta_coll=agg_coll / n_informed,
        agg_nash=n_informed * beta_n,
        agg_coll=agg_coll,
        lam_nash=lam_n,
        lam_coll=lam_c,
        profit_nash=learner_profit(beta_n, total_nash, lam_n, sigma_v),
        profit_coll=learner_profit(agg_coll / n_informed, total_coll, lam_c, sigma_v),
        info_nash=informativeness(total_nash, sigma_v, sigma_u),
        info_coll=informativeness(total_coll, sigma_v, sigma_u),
    )


def _numeric_benchmarks(I, P, sigma_v, sigma_u, xi, theta) -> Benchmarks:
    n = I + P
    beta_n = _nash_beta(n, sigma_v, sigma_u, xi, theta)
    passive_total = P * beta_n
    total_nash = n * beta_n
    lam_n = market_lambda(total_nash, sigma_v, sigma_u, xi, theta)
    agg_coll = _collusive_agg(passive_total, I * beta_n, sigma_v, sigma_u, xi, theta)
    total_coll = agg_coll + passive_total
    lam_c = market_lambda(total_coll, sigma_v, sigma_u, xi, theta)
    return Benchmarks(
        n_informed=I,
        n_passive=P,
        sigma_v=sigma_v,
        sigma_u=sigma_u,
        beta_nash=beta_n,
        beta_coll=agg_coll / I,
        agg_nash=I * beta_n,
        agg_coll=agg_coll,
        lam_nash=lam_n,
        lam_coll=lam_c,
        profit_nash=learner_profit(beta_n, total_nash, lam_n, sigma_v),
        profit_coll=learner_profit(agg_coll / I, total_coll, lam_c, sigma_v),
        info_nash=informativeness(total_nash, sigma_v, sigma_u),
        info_coll=informativeness(total_coll, sigma_v, sigma_u),
    )


def collusion_index(value: float, nash: float, coll: float) -> float:
    """Calvano et al. (2020) style normalisation: 0 = Nash, 1 = full collusion.

    Works for any metric that moves monotonically from the Nash to the
    collusive benchmark (profit, aggregate intensity, informativeness).
    """
    if math.isclose(nash, coll, rel_tol=1e-4, abs_tol=1e-12):
        # Degenerate or ill-conditioned: no gain from collusion (e.g. I = P + 1
        # with passive Nash traders), or benchmarks practically equal (e.g.
        # informativeness ~ 1 under both when xi is large). Undefined.
        return value * float("nan")
    return (value - nash) / (coll - nash)


def deviation_gap(b: Benchmarks, xi: float = 0.0, theta: float = 0.1) -> float:
    """Intensity change of a one-period best-response deviation from collusion.

    With rivals at beta_coll and the market maker pricing at lam_coll, the
    myopic best response is beta_d = (1 - lam_coll * (others)) / (2 lam_coll).
    Returns beta_d - beta_coll (per trader). Multiplied by sigma_v / sigma_u it
    is the deviation's signal-to-noise ratio per unit of v / sigma_v.
    """
    others = (b.n_informed - 1) * b.beta_coll + b.n_passive * b.beta_nash
    beta_d = (1.0 - b.lam_coll * others) / (2.0 * b.lam_coll)
    return beta_d - b.beta_coll
