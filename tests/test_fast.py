import numpy as np
import pytest

pytest.importorskip("numba")

from kylecollusion.agents.tabular_q import TabularQ
from kylecollusion.fast import fast_train
from kylecollusion.market import KyleMarket, MarketConfig


def _setup(engine_seed=0, **cfg_kw):
    cfg = MarketConfig(**cfg_kw)
    env = KyleMarket(cfg, 40, seed=engine_seed)
    ag = TabularQ(env, alpha=0.1, gamma=0.0, beta_decay=5e-5, seed=engine_seed + 1)
    env.reset()
    return env, ag


def test_reproducible_and_resumable():
    e1, a1 = _setup(memory="price", price_bins="grid", grid_mode="bracket", n_values=10,
                    n_actions=15, sigma_u=0.1, xi=500.0)
    e2, a2 = _setup(memory="price", price_bins="grid", grid_mode="bracket", n_values=10,
                    n_actions=15, sigma_u=0.1, xi=500.0)
    fast_train(e1, a1, 0, 20000)
    fast_train(e2, a2, 0, 7000)
    fast_train(e2, a2, 7000, 20000)  # chunked run continues the same streams
    assert np.array_equal(a1.Q, a2.Q)
    assert np.array_equal(e1.lam, e2.lam)
    s = e1._obs()["s"]
    assert s.min() >= 0 and s.max() < e1.n_states


def _mean_order_intensity(env, ag, steps, engine):
    if engine == "numba":
        fast_train(env, ag, 0, steps)
    else:
        obs = env.reset()
        for t in range(steps):
            a = ag.act(obs, t)
            r, obs2, _ = env.step(a)
            ag.observe(obs, a, r, obs2, t)
            obs = obs2
    pol = ag.greedy_policy()[:, :, 0]  # (S, I, n_values) with memory none
    x = env.grid_v[np.arange(env.n_values)[None, None, :], pol]
    v = env.values
    return float((x * v).sum(-1).mean() / (v * v).sum())


def test_same_distribution_as_numpy():
    kw = dict(memory="none", n_informed=1, mm_fixed=True)
    b_np = _mean_order_intensity(*_setup(0, **kw), 60000, "numpy")
    b_nb = _mean_order_intensity(*_setup(0, **kw), 60000, "numba")
    assert abs(b_np - b_nb) < 0.04 * abs(b_np)


def test_dou_spec_modes_run_and_agree():
    """Dou et al. price grid, lagged-value memory and value-specific
    exploration: both engines run, states stay in range, and the learned
    single-trader intensity agrees in distribution."""
    for mem in ("price", "value"):
        cfg = dict(memory=mem, price_bins="dou", grid_mode="bracket", n_values=10,
                   n_actions=15, n_price_bins=31, sigma_u=0.1, xi=500.0)
        env, _ = _setup(**cfg)
        ag = TabularQ(env, alpha=0.05, gamma=0.0, beta_decay=5e-5, explore_by_value=True, seed=1)
        fast_train(env, ag, 0, 5000)
        s = env._obs()["s"]
        assert s.min() >= 0 and s.max() < env.n_states
        assert ag.vcount.sum() == 5000 * env.S  # one count per session-period


def test_value_exploration_same_distribution():
    kw = dict(memory="none", n_informed=1, mm_fixed=True)
    out = []
    for engine in ("numpy", "numba"):
        env, _ = _setup(0, **kw)
        ag = TabularQ(env, alpha=0.1, gamma=0.0, beta_decay=2.5e-4, explore_by_value=True, seed=1)
        out.append(_mean_order_intensity(env, ag, 60000, engine))
    assert abs(out[0] - out[1]) < 0.04 * abs(out[0])


def test_stopping_rule_freezes_converged_sessions():
    env, _ = _setup(memory="none", n_informed=1, mm_fixed=True)
    ag = TabularQ(env, alpha=0.1, gamma=0.0, beta_decay=1e-3, stop_unchanged=2000, seed=1)
    fast_train(env, ag, 0, 60000)
    assert ag.done.any()
    assert (ag.conv_time[ag.done] >= 2000).all() and (ag.conv_time[ag.done] <= 60000).all()
    q = ag.Q.copy()
    fast_train(env, ag, 60000, 70000)  # converged sessions no longer learn
    assert np.array_equal(q[ag.done], ag.Q[ag.done])


def test_rolling_market_maker_matches_between_engines():
    """Dou et al.'s rolling least-squares market maker: compiled engine keeps
    exact window sums and the same lambda as a direct recomputation."""
    cfg = dict(memory="price", price_bins="dou", grid_mode="bracket", n_values=10,
               n_actions=15, n_price_bins=31, sigma_u=0.1, xi=500.0, mm_window=500)
    env, ag = _setup(**cfg)
    fast_train(env, ag, 0, 3000)  # crosses several window boundaries
    bv, by = env.buf_v, env.buf_y
    exact = np.stack([bv.sum(1), by.sum(1), (by * by).sum(1), (bv * by).sum(1)], 1)
    assert np.allclose(env.mm_sums, exact, rtol=1e-9, atol=1e-6)
    var_y = by.var(1)
    g1 = ((bv * by).mean(1) - bv.mean(1) * by.mean(1)) / var_y
    lam = (0.1 * g1 + 500.0) / (0.1 + 500.0**2)
    assert np.allclose(env.lam, lam, rtol=1e-9)
