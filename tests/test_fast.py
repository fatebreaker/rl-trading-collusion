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
