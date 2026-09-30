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


def market_lambda(total_beta: float, sigma_v: float, sigma_u: float) -> float:
    """Rational linear pricing given aggregate informed intensity."""
    return total_beta * sigma_v**2 / (total_beta**2 * sigma_v**2 + sigma_u**2)


def informativeness(total_beta: float, sigma_v: float, sigma_u: float) -> float:
    s = total_beta**2 * sigma_v**2
    return s / (s + sigma_u**2)


def learner_profit(beta_i: float, total_beta: float, lam: float, sigma_v: float) -> float:
    """E[(v - lam*y) * beta_i * v] with y = total_beta * v + u."""
    return beta_i * sigma_v**2 * (1.0 - lam * total_beta)


def kyle_benchmarks(
    n_informed: int, sigma_v: float, sigma_u: float, n_passive: int = 0
) -> Benchmarks:
    if n_informed < 1:
        raise ValueError("need at least one learning informed trader")
    n = n_informed + n_passive

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


def collusion_index(value: float, nash: float, coll: float) -> float:
    """Calvano et al. (2020) style normalisation: 0 = Nash, 1 = full collusion.

    Works for any metric that moves monotonically from the Nash to the
    collusive benchmark (profit, aggregate intensity, informativeness).
    """
    if math.isclose(nash, coll, rel_tol=1e-9, abs_tol=1e-12):
        # Degenerate market: no gain from collusion (e.g. I = P + 1 with
        # passive Nash traders), so the index is undefined.
        return value * float("nan")
    return (value - nash) / (coll - nash)
