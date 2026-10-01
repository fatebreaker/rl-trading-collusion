import math

from kylecollusion.sustain import ic_slack, max_sustainable, max_sustainable_any, necessary_slack
from kylecollusion.theory import kyle_benchmarks


def test_nash_always_sustainable():
    bm = kyle_benchmarks(2, 1.0, 1.0)
    assert ic_slack(bm.beta_nash, 0.5, 1, [2.0], bm=bm).max() >= -1e-12
    assert necessary_slack(bm.beta_nash, 0.5, bm=bm) >= -1e-12


def test_monotone_in_delta_and_detectability():
    lo = max_sustainable(0.5, math.inf, xi=500, sigma_u=0.1, n_values=10, n_b=100)["delta_beta"]
    hi = max_sustainable(0.9, math.inf, xi=500, sigma_u=0.1, n_values=10, n_b=100)["delta_beta"]
    assert 0 < lo < hi <= 1.0 + 1e-9
    # memory-one punishment is weaker than grim
    t1 = max_sustainable(0.9, 1, xi=500, sigma_u=0.1, n_values=10, n_b=100)["delta_beta"]
    assert t1 < hi


def test_kyle_nash_reversion_fails_but_necessary_condition_holds():
    assert max_sustainable(0.95, math.inf, n_b=100)["delta_beta"] < 0.02
    assert max_sustainable_any(0.95, n_b=100)["delta_beta"] > 0.99
