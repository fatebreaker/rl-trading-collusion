import numpy as np

from kylecollusion.bertrand import BertrandConfig, BertrandMarket, BertrandQ, benchmarks, deviation_response


def test_benchmarks_match_calvano():
    b = benchmarks(BertrandConfig())
    assert abs(b["p_nash"] - 1.473) < 2e-3
    assert abs(b["p_mono"] - 1.925) < 2e-3
    assert abs(b["pi_nash"] - 0.2229) < 1e-3
    assert abs(b["pi_mono"] - 0.3375) < 1e-3


def test_payoff_symmetry_and_states():
    env = BertrandMarket(BertrandConfig(), 4, seed=0)
    P = env.payoff
    assert np.allclose(P[:, :, 0], P[:, :, 1].T)
    env.reset()
    a = np.array([[1, 5], [5, 1], [0, 0], [14, 3]])
    _, s = env.step(a)
    m = env.n_actions
    # each firm sees (own, rival)
    assert s[0, 0] == 1 * m + 5 and s[0, 1] == 5 * m + 1
    assert s[0, 0] == s[1, 1]


def test_deviation_is_static_best_response_for_either_firm():
    env = BertrandMarket(BertrandConfig(), 8, seed=0)
    ag = BertrandQ(env, seed=0)
    # force both firms to play the top price everywhere: a deviation must undercut
    ag.Q[:] = 0.0
    ag.Q[..., -1] = 1.0
    s = env.reset()
    for dev in (0, 1):
        out = deviation_response(env, ag, s, horizon=3, deviator=dev)
        assert out["share_deviated"] == 1.0
        assert out["d_price_dev"][0] < 0
        assert out["d_price_rival"][0] == 0
        # the rival does not respond under a constant policy, so deviation pays
        assert out["cum_gain_dev"] > 0
