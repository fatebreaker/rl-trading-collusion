"""Can trigger strategies sustain collusion in the repeated Kyle market?

A Green & Porter (1984) style check under imperfect public monitoring. On the
collusive path every learner trades b per unit of v and the market maker
prices at lambda(I b). A deviating trader plays its static best response for
one period. Rivals see last period's value v and the aggregate order flow
(equivalently the price), and start a punishment phase of T periods of static
Nash play when the order-flow surprise sign(v) (y - I b v) exceeds k sigma_u.

Order flow carries noise u ~ N(0, sigma_u^2), so the deviation is detected
with probability 1 - Phi(k - d(v)), where d(v) = (b_d - b)|v| / sigma_u, and
punishment is also triggered by mistake with probability 1 - Phi(k). The
deviation is unprofitable at value v iff

    g(v) <= delta (q1(v) - q0) (1 - delta^T) (V - pi^N / (1 - delta)),

with V the value of the cartel path including false punishments,
    V = (pi(b) + delta q0 pi^N w_T) / (1 - delta (1 - q0) - delta^{T+1} q0),
    w_T = (1 - delta^T) / (1 - delta).

Payoffs are computed with the market maker's lambda held at its on-path
value, since it re-estimates price impact slowly (it does not react to a
single period). T = 1 is the harshest phase a memory-one strategy can run.
The threshold k is optimised on a grid; the result is therefore a sufficient
condition for sustainability within this class of strategies.
"""

from __future__ import annotations

import math
from statistics import NormalDist

import numpy as np

from .market import value_grid
from .theory import kyle_benchmarks, market_lambda

_PHI = np.vectorize(NormalDist().cdf, otypes=[float])


def _path(b, I, sigma_v, sigma_u, xi, theta):
    lam = market_lambda(I * b, sigma_v, sigma_u, xi, theta)
    others = (I - 1) * b
    b_d = (1.0 - lam * others) / (2.0 * lam)
    prof = b * (1.0 - lam * I * b)  # per unit v^2
    prof_d = b_d * (1.0 - lam * (b_d + others))
    return lam, b_d, prof, prof_d


def ic_slack(b, delta, T, k, I=2, sigma_v=1.0, sigma_u=1.0, xi=0.0, theta=0.1, n_values=5, bm=None):
    """Minimum over values of (punishment loss - deviation gain), in units of
    sigma_v^2, for each threshold in k. Non-negative means b is sustainable."""
    bm = bm or kyle_benchmarks(I, sigma_v, sigma_u, xi=xi, theta=theta)
    k = np.atleast_1d(np.asarray(k, float))[:, None]
    v = value_grid(n_values, sigma_v)
    v = v[v != 0]
    _, b_d, prof, prof_d = _path(b, I, sigma_v, sigma_u, xi, theta)
    pi_c = prof * sigma_v**2  # E[v^2] = sigma_v^2 on the grid
    pi_n = bm.profit_nash
    q0 = 1.0 - _PHI(k)
    q1 = 1.0 - _PHI(k - abs(b_d - b) * np.abs(v)[None, :] / sigma_u)
    if math.isinf(T):
        w, dT = 1.0 / (1.0 - delta), 0.0
    else:
        w, dT = (1.0 - delta**T) / (1.0 - delta), delta**T
    V = (pi_c + delta * q0 * pi_n * w) / (1.0 - delta * (1.0 - q0) - delta * dT * q0)
    loss = delta * (q1 - q0) * (1.0 - dT) * (V - pi_n / (1.0 - delta))
    gain = (prof_d - prof) * v**2
    return (loss - gain[None, :]).min(1)


def max_sustainable(delta, T, I=2, sigma_v=1.0, sigma_u=1.0, xi=0.0, theta=0.1, n_values=5,
                    n_b=400, ks=np.linspace(-3, 6, 181)):
    """Most collusive per-learner intensity sustainable by a trigger strategy.

    Scans b from the collusive to the Nash intensity and returns the smallest
    sustainable one with its collusion index (intensity and profit). Nash
    itself is always sustainable."""
    bm = kyle_benchmarks(I, sigma_v, sigma_u, xi=xi, theta=theta)
    bs = np.linspace(bm.beta_coll, bm.beta_nash, n_b)
    for b in bs:
        if ic_slack(b, delta, T, ks, I, sigma_v, sigma_u, xi, theta, n_values, bm).max() >= 0:
            break
    _, _, prof, _ = _path(b, I, sigma_v, sigma_u, xi, theta)
    d_beta = (b - bm.beta_nash) / (bm.beta_coll - bm.beta_nash)
    d_prof = (prof * sigma_v**2 - bm.profit_nash) / (bm.profit_coll - bm.profit_nash)
    return {"beta": float(b), "delta_beta": float(d_beta), "delta_profit": float(d_prof)}


def necessary_slack(b, delta, I=2, sigma_v=1.0, sigma_u=1.0, xi=0.0, theta=0.1, n_values=5, bm=None):
    """Necessary condition for any punishment scheme.

    Whatever the strategies, a deviation at value v changes the probability of
    any punishment event by at most the total-variation distance between the
    order-flow distributions, 2 Phi(d(v)/2) - 1, and the deviator's
    continuation value can fall by at most (I pi^C - 0) / (1 - delta): it can
    never get more than the whole cartel profit (with the market maker
    pricing rationally), nor less than zero (it can stop trading). The
    deviation must not pay at every v:
        g(v) <= delta / (1 - delta) * (2 Phi(d(v)/2) - 1) * I * pi^C.
    Returns the minimum slack over values; negative means b cannot be an
    equilibrium outcome under any punishment."""
    v = value_grid(n_values, sigma_v)
    v = v[v != 0]
    _, b_d, prof, prof_d = _path(b, I, sigma_v, sigma_u, xi, theta)
    d = abs(b_d - b) * np.abs(v) / sigma_u
    tv = 2.0 * _PHI(d / 2.0) - 1.0
    bm = bm or kyle_benchmarks(I, sigma_v, sigma_u, xi=xi, theta=theta)
    loss = delta / (1.0 - delta) * tv * I * bm.profit_coll
    return float((loss - (prof_d - prof) * v**2).min())


def max_sustainable_any(delta, I=2, sigma_v=1.0, sigma_u=1.0, xi=0.0, theta=0.1, n_values=5, n_b=400):
    """Most collusive intensity that passes the necessary condition."""
    bm = kyle_benchmarks(I, sigma_v, sigma_u, xi=xi, theta=theta)
    for b in np.linspace(bm.beta_coll, bm.beta_nash, n_b):
        if necessary_slack(b, delta, I, sigma_v, sigma_u, xi, theta, n_values, bm) >= 0:
            break
    return {"beta": float(b), "delta_beta": float((b - bm.beta_nash) / (bm.beta_coll - bm.beta_nash))}
