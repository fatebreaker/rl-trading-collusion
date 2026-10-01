"""The benchmarks are the yardstick for every result, so check them hard."""

import math

import numpy as np
import pytest

from kylecollusion.theory import (
    collusion_index,
    kyle_benchmarks,
    learner_profit,
    market_lambda,
)


@pytest.mark.parametrize("I,P", [(1, 0), (2, 0), (3, 0), (5, 0), (2, 1), (3, 2)])
@pytest.mark.parametrize("su", [0.5, 1.0, 2.0])
def test_nash_is_mutual_best_response(I, P, su):
    """Given rational lambda and others at beta_nash, beta_nash is a best response."""
    sv = 1.3
    b = kyle_benchmarks(I, sv, su, P)
    n = I + P
    lam = market_lambda(n * b.beta_nash, sv, su)
    assert lam == pytest.approx(b.lam_nash, rel=1e-9)

    others = (n - 1) * b.beta_nash
    # Deviator's profit holding lambda fixed (market maker cannot see deviations).
    def dev_profit(beta):
        return beta * sv**2 * (1 - lam * (beta + others))

    grid = np.linspace(0, 3 * b.beta_nash, 20001)
    best = grid[np.argmax([dev_profit(g) for g in grid])]
    assert best == pytest.approx(b.beta_nash, rel=1e-3)


@pytest.mark.parametrize("I,P", [(2, 0), (3, 0), (2, 1), (4, 3)])
def test_collusive_maximises_joint_profit(I, P):
    sv, su = 1.0, 1.0
    b = kyle_benchmarks(I, sv, su, P)
    bp = P * b.beta_nash

    def joint(bl):
        tot = bl + bp
        return learner_profit(bl, tot, market_lambda(tot, sv, su), sv)

    grid = np.linspace(1e-3, 4, 40001)
    best = grid[np.argmax([joint(g) for g in grid])]
    assert best == pytest.approx(b.agg_coll, rel=1e-3)
    assert joint(b.agg_coll) / I == pytest.approx(b.profit_coll, rel=1e-9)


@pytest.mark.parametrize("I", [2, 3, 4])
def test_collusion_beats_nash_and_hurts_informativeness(I):
    b = kyle_benchmarks(I, 1.0, 1.0)
    assert b.profit_coll > b.profit_nash
    assert b.agg_coll < b.agg_nash
    assert b.info_coll < b.info_nash


def test_monopoly_closed_form():
    b = kyle_benchmarks(1, 2.0, 0.5)
    assert b.beta_nash == pytest.approx(0.5 / 2.0)
    assert b.lam_nash == pytest.approx(2.0 / (2 * 0.5))
    assert b.profit_nash == pytest.approx(b.profit_coll)
    assert math.isnan(collusion_index(1.0, b.profit_nash, b.profit_coll))


def test_collusion_index_endpoints():
    assert collusion_index(1.0, 1.0, 2.0) == 0.0
    assert collusion_index(2.0, 1.0, 2.0) == 1.0
    # Works when the collusive benchmark is below Nash (intensity, informativeness).
    assert collusion_index(1.0, 1.4, 1.0) == pytest.approx(1.0)


@pytest.mark.parametrize("P", [0, 1, 2, 4])
def test_passive_degenerate_when_learners_exceed_passive_by_one(P):
    """With I = P + 1 the collusive aggregate equals the Nash aggregate:
    B_L^coll = sqrt(B_P^2 + 1) = I / sqrt(I + P) = B_L^nash (sigma_v = sigma_u = 1)."""
    b = kyle_benchmarks(P + 1, 1.0, 1.0, P)
    assert b.agg_coll == pytest.approx(b.agg_nash)
    assert b.profit_coll == pytest.approx(b.profit_nash)
    assert math.isnan(collusion_index(0.5, b.agg_nash, b.agg_coll))
    assert np.isnan(collusion_index(np.array([0.5, 1.0]), b.agg_nash, b.agg_coll)).all()


def test_xi_benchmarks_match_dou_et_al():
    """Dou, Goldstein & Ji (2025) regime: sigma_u = 0.1, theta = 0.1, xi = 500, I = 2
    gives chi^N ~ 166.667, chi^M = 125 and lambda^N ~ 2e-3."""
    b = kyle_benchmarks(2, 1.0, 0.1, xi=500.0, theta=0.1)
    assert b.beta_nash == pytest.approx(166.667, rel=1e-4)
    assert b.beta_coll == pytest.approx(125.0, rel=1e-4)
    assert b.lam_nash == pytest.approx(2e-3, rel=1e-3)
    assert b.profit_coll > b.profit_nash


@pytest.mark.parametrize("I,P", [(2, 0), (3, 0), (3, 1)])
def test_numeric_solver_matches_closed_form_at_xi_zero(I, P):
    from kylecollusion.theory import _numeric_benchmarks

    a = kyle_benchmarks(I, 1.0, 0.7, P)
    c = _numeric_benchmarks(I, P, 1.0, 0.7, 1e-12, 0.1)
    for f in ("beta_nash", "agg_coll", "lam_nash", "profit_nash", "profit_coll"):
        assert getattr(c, f) == pytest.approx(getattr(a, f), rel=1e-6), f


def test_deviation_signal_to_noise_by_regime():
    """At xi = 0 the deviation's signal-to-noise ratio does not depend on sigma_u;
    at xi = 500 deviations are hundreds of noise standard deviations large."""
    from kylecollusion.theory import deviation_gap

    snr = [deviation_gap(kyle_benchmarks(2, 1.0, su)) / su for su in (0.1, 1.0, 5.0)]
    assert snr[0] == pytest.approx(0.25) and snr[1] == pytest.approx(0.25) and snr[2] == pytest.approx(0.25)
    b = kyle_benchmarks(2, 1.0, 0.1, xi=500.0, theta=0.1)
    assert deviation_gap(b, 500.0, 0.1) / 0.1 > 500


def test_informativeness_index_undefined_when_benchmarks_coincide():
    b = kyle_benchmarks(2, 1.0, 0.1, xi=500.0, theta=0.1)
    assert math.isnan(collusion_index(0.99, b.info_nash, b.info_coll))
    assert not math.isnan(collusion_index(150.0, b.agg_nash, b.agg_coll))
