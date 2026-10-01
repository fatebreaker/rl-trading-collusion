"""The punishment test must find punishment exactly when it is there."""

import numpy as np
import pytest

from kylecollusion.diagnostics import convergence_stats, impulse_response
from kylecollusion.market import KyleMarket, MarketConfig


class Scripted:
    """Order = (base + slope * surprise) * v, snapped to the grid, where
    surprise = sign(v_prev) * (residual bin - middle bin).

    A rival over-trading pushes flow in the direction of last period's value,
    so the surprise is positive after a deviation whatever the value's sign.
    slope = 0 is memoryless: nobody can react. slope > 0 trades harder after a
    surprise, a crude punishment rule. A trader's own deviation cancels out of
    its own residual (residual = y - own x), so only rivals react.
    """

    def __init__(self, env: KyleMarket, base: float, slope: float):
        self.env, self.base, self.slope = env, base, slope

    def act(self, obs, t, greedy=False):
        env = self.env
        nb = env.cfg.n_flow_bins
        rbin = obs["s"] % nb
        v_prev = env.values[obs["s"] // nb]
        surprise = np.sign(v_prev) * (rbin - (nb - 1) / 2)
        v = env.values[obs["v_idx"]][:, None]
        x = (self.base + self.slope * surprise) * v
        return np.abs(env.grid[None, None, :] - x[..., None]).argmin(-1)


def _run(slope, sessions=200, reps=30):
    env = KyleMarket(MarketConfig(memory="residual", n_actions=61), sessions, seed=3)
    agent = Scripted(env, base=env.bench.beta_coll, slope=slope)
    obs = env.reset()
    for _ in range(3000):  # let the market maker learn the scripted strategy
        _, obs, _ = env.step(agent.act(obs, 0))
    return impulse_response(env, agent, obs, horizon=6, reps=reps)


def test_memoryless_rivals_do_not_react():
    imp = _run(slope=0.0)
    assert imp["n_events"] > 0
    assert np.allclose(imp["d_beta_rival"], 0.0)
    # The myopic best response is profitable in the deviation period...
    assert imp["gain_dev_period0"] > 0
    # ...and with no punishment the deviation pays overall.
    assert imp["cum_gain_dev"] > 0


def test_reactive_rivals_are_detected():
    imp = _run(slope=0.08)
    assert imp["d_beta_rival"][0] == 0.0  # rivals move after, not during
    lo = imp["d_beta_rival"][1] - imp["d_beta_rival_ci95"][1]
    assert lo > 0, imp["d_beta_rival"][:3]


def test_paired_copies_share_randomness():
    """With no deviation possible (grid of one order) every difference is zero."""
    env = KyleMarket(MarketConfig(n_actions=1), 16, seed=0)
    agent = Scripted(env, base=0.0, slope=0.0)
    obs = env.reset()
    imp = impulse_response(env, agent, obs, horizon=3, reps=3)
    assert imp["n_events"] == 0


def test_convergence_stats():
    snap = np.zeros((2, 1, 2, 1), dtype=int)
    now = snap.copy()
    now[0, 0, 0, 0] = 2  # session 0 moves one on-path entry by two steps
    now[1, 0, 1, 0] = 1  # session 1 moves one off-path entry
    onpath = np.zeros_like(snap, dtype=bool)
    onpath[:, 0, 0, 0] = True
    st = convergence_stats(snap, now, onpath)
    assert st["policy_change"].tolist() == [0.5, 0.5]
    assert st["policy_change_onpath"].tolist() == [1.0, 0.0]
    assert st["order_shift_onpath"].tolist() == pytest.approx([2.0, 0.0])


class PriceTrigger:
    """Trades at the collusive level unless last period's price surprise was
    large and in the direction of the value, then trades at Nash."""

    def __init__(self, env, threshold):
        self.env, self.th = env, threshold

    def act(self, obs, t, greedy=False):
        env = self.env
        surprise = obs["feat"][..., 2]  # price surprise in noise-price sd units
        prev_sign = np.sign(obs["feat"][..., 1])
        punish = surprise * prev_sign > self.th
        v = env.values[obs["v_idx"]][:, None]
        beta = np.where(punish, env.bench.beta_nash, env.bench.beta_coll)
        x = beta * v
        return np.abs(env.grid_v[obs["v_idx"]][:, None, :] - x[..., None]).argmin(-1)


def _shock_run(agent_cls, **kw):
    from kylecollusion.diagnostics import noise_shock_response

    env = KyleMarket(MarketConfig(memory="price", n_actions=61), 200, seed=2)
    agent = agent_cls(env, **kw)
    obs = env.reset()
    for _ in range(3000):
        _, obs, _ = env.step(agent.act(obs, 0))
    return noise_shock_response(env, agent, obs, shock_size=3.0 * env.cfg.sigma_u, horizon=3, reps=10)


class Memoryless:
    def __init__(self, env, beta):
        self.env, self.beta = env, beta

    def act(self, obs, t, greedy=False):
        env = self.env
        x = self.beta * env.values[obs["v_idx"]][:, None] * np.ones((1, env.I))
        return np.abs(env.grid_v[obs["v_idx"]][:, None, :] - x[..., None]).argmin(-1)


def test_noise_shock_ignored_by_memoryless_traders():
    sh = _shock_run(Memoryless, beta=0.5)
    assert np.allclose(sh["d_beta_all"][1:], 0.0)
    assert sh["d_price"][0] > 0  # the shock itself moves the price


def test_noise_shock_triggers_price_trigger_traders():
    sh = _shock_run(PriceTrigger, threshold=1.5)
    assert sh["d_beta_all"][0] == pytest.approx(0.0, abs=1e-12)
    assert sh["d_beta_all"][1] - sh["d_beta_all_ci95"][1] > 0
