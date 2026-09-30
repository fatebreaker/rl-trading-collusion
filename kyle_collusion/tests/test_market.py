"""Simulate known strategies and check the market reproduces the theory."""

import numpy as np
import pytest

from kylecollusion.market import KyleMarket, MarketConfig, value_grid
from kylecollusion.metrics import session_metrics


def test_value_grid_moments():
    g = value_grid(5, 1.7)
    assert g.mean() == pytest.approx(0.0, abs=1e-12)
    assert g.std() == pytest.approx(1.7)


def _simulate(env, beta_per_trader, steps):
    env.reset()
    log = {k: np.empty((steps, env.S)) for k in ("v", "p", "y", "lam")}
    log["x"] = np.empty((steps, env.S, env.I))
    log["profit"] = np.empty((steps, env.S, env.I))
    for t in range(steps):
        v = env.values[env.v_idx]
        x = np.repeat((beta_per_trader * v)[:, None], env.I, axis=1)
        r, _, info = env.step_orders(x)
        for k in ("v", "p", "y", "lam"):
            log[k][t] = info[k]
        log["x"][t] = info["x"]
        log["profit"][t] = r
    return log


@pytest.mark.parametrize("I,P", [(2, 0), (3, 0), (3, 1)])
def test_nash_play_recovers_nash_outcome(I, P):
    env = KyleMarket(MarketConfig(n_informed=I, n_passive=P, mm_halflife=500), 64, seed=1)
    b = env.bench
    log = _simulate(env, b.beta_nash, 6000)
    m = session_metrics({k: v[2000:] for k, v in log.items()}, b)
    assert m["lam"].mean() == pytest.approx(b.lam_nash, rel=0.03)
    assert m["profit"].mean() == pytest.approx(b.profit_nash, rel=0.05)
    assert m["agg_intensity"].mean() == pytest.approx(b.agg_nash, rel=0.01)
    assert abs(m["delta_intensity"].mean()) < 0.05


def test_collusive_play_recovers_collusive_outcome():
    env = KyleMarket(MarketConfig(n_informed=2, mm_halflife=500), 64, seed=2)
    b = env.bench
    log = _simulate(env, b.beta_coll, 6000)
    m = session_metrics({k: v[2000:] for k, v in log.items()}, b)
    assert m["lam"].mean() == pytest.approx(b.lam_coll, rel=0.03)
    assert m["profit"].mean() == pytest.approx(b.profit_coll, rel=0.05)
    assert m["delta_intensity"].mean() == pytest.approx(1.0, abs=0.05)


def test_observation_shapes_and_state_range():
    for mem in ("none", "flow", "residual", "orders"):
        env = KyleMarket(MarketConfig(memory=mem), 8, seed=0)
        obs = env.reset()
        for _ in range(50):
            a = env.rng.integers(env.n_actions, size=(8, 2))
            _, obs, _ = env.step(a)
            assert obs["s"].shape == (8, 2)
            assert obs["s"].min() >= 0 and obs["s"].max() < env.n_states
            assert obs["feat"].shape == (8, 2, env.feat_dim)


def test_order_cap_and_tick():
    env = KyleMarket(MarketConfig(order_cap=0.3, tick=0.25), 4, seed=0)
    assert np.abs(env.grid).max() <= 0.3 + 1e-12
    env.reset()
    _, _, info = env.step(np.zeros((4, 2), dtype=int))
    assert np.allclose(info["p"] / 0.25, np.round(info["p"] / 0.25))


def test_sessions_are_independent():
    """Changing one session's orders must not affect another session's prices."""
    cfg = MarketConfig()
    e1, e2 = KyleMarket(cfg, 4, seed=5), KyleMarket(cfg, 4, seed=5)
    e1.reset(), e2.reset()
    for _ in range(200):
        x = np.full((4, 2), 0.3)
        x2 = x.copy()
        x2[0] = -1.0  # perturb session 0 only
        _, _, i1 = e1.step_orders(x)
        _, _, i2 = e2.step_orders(x2)
    assert np.allclose(i1["lam"][1:], i2["lam"][1:])
    assert not np.isclose(i1["lam"][0], i2["lam"][0])


def test_orders_memory_sees_rival_order_exactly():
    env = KyleMarket(MarketConfig(memory="orders"), 16, seed=0)
    env.reset()
    for _ in range(20):
        a = env.rng.integers(env.n_actions, size=(16, 2))
        _, obs, _ = env.step(a)
        prev_v = env.prev_v_idx
        # state = v_prev * n_actions + rival's order index, for each trader
        assert (obs["s"][:, 0] == prev_v * env.n_actions + a[:, 1]).all()
        assert (obs["s"][:, 1] == prev_v * env.n_actions + a[:, 0]).all()


def test_bracket_grid_spans_collusive_to_nash():
    cfg = MarketConfig(grid_mode="bracket", n_values=10, n_actions=15, sigma_u=0.1)
    env = KyleMarket(cfg, 4, seed=0)
    b = env.bench
    for k, v in enumerate(env.values):
        xn, xm = b.beta_nash * v, b.beta_coll * v
        lo, hi = sorted([xm - 0.1 * (xn - xm), xn + 0.1 * (xn - xm)])
        assert env.grid_v[k, 0] == pytest.approx(lo)
        assert env.grid_v[k, -1] == pytest.approx(hi)


def test_price_memory_and_xi_market():
    cfg = MarketConfig(memory="price", grid_mode="bracket", n_values=10, n_actions=15,
                       sigma_u=0.1, xi=500.0)
    env = KyleMarket(cfg, 32, seed=1)
    obs = env.reset()
    for _ in range(3000):
        _, obs, info = env.step(np.full((32, 2), 7))  # middle of the bracket
        assert obs["s"].min() >= 0 and obs["s"].max() < env.n_states
    # With xi = 500 the market maker's impact collapses to about 1/xi.
    assert info["lam"].mean() == pytest.approx(2e-3, rel=0.05)


def test_old_checkpoints_still_load():
    import pickle

    env = KyleMarket(MarketConfig(memory="orders"), 4, seed=0)
    env.reset()
    state = dict(env.__dict__)
    for k in ("grid_v", "u_shock", "prev_rival_idx", "prev_price"):
        state.pop(k)
    blob = pickle.dumps(env)
    old = pickle.loads(blob)
    old.__dict__.clear()
    old.__setstate__(state)
    _, obs, _ = old.step(np.zeros((4, 2), dtype=int))
    assert obs["s"].shape == (4, 2)
