"""Convergence and punishment diagnostics.

Low trading on its own does not prove collusion: it can come from learning
bias. Calvano et al. (2020) make the case with an impulse response: force one
agent to deviate once and show the rival punishes, so that deviating does not
pay. `impulse_response` does the same here, with paired simulations so the
comparison is exact: the deviating copy and the baseline copy of each market
share every value and noise draw, so all differences come from the deviation.
"""

from __future__ import annotations

import copy
import math

import numpy as np

from .market import KyleMarket


def convergence_stats(
    snap: np.ndarray, now: np.ndarray, onpath: np.ndarray | None, grid: np.ndarray
) -> dict[str, np.ndarray]:
    """Compare greedy tables (S, I, n_states, n_values) taken `window` steps apart.

    policy_change        share of all table entries whose greedy order changed
    policy_change_onpath same, restricted to entries visited in greedy play
    order_shift_onpath   mean |change in order| on-path, in grid steps
    """
    S = snap.shape[0]
    changed = now != snap
    out = {"policy_change": changed.reshape(S, -1).mean(1)}
    if onpath is not None:
        step = abs(grid[1] - grid[0]) if len(grid) > 1 else 1.0
        shift = np.abs(grid[now] - grid[snap]) / step
        m = onpath.reshape(S, -1)
        cnt = np.maximum(m.sum(1), 1)
        out["policy_change_onpath"] = (changed.reshape(S, -1) * m).sum(1) / cnt
        out["order_shift_onpath"] = (shift.reshape(S, -1) * m).sum(1) / cnt
    return out


def _best_response_idx(env: KyleMarket, others_x: np.ndarray) -> np.ndarray:
    """Myopic best order given the market maker's current rule and rivals' orders."""
    v = env.values[env.v_idx]
    lam = np.maximum(env.lam, 1e-6)
    expected_rest = others_x + env.passive_beta * v - env.m_y
    x_star = (v - env.m_v - lam * expected_rest) / (2.0 * lam)
    return np.abs(env.grid[None, :] - x_star[:, None]).argmin(1)


def impulse_response(
    env: KyleMarket,
    agent,
    obs: dict,
    horizon: int = 15,
    reps: int = 40,
    gap: int = 50,
    gamma: float = 0.95,
    deviator: int = 0,
) -> dict:
    """Paired deviation experiment on greedy play.

    For each repetition: run `gap` greedy periods, clone the market, make
    `deviator` play its myopic best response for one period in the clone, then
    let everyone follow their learned strategies in both copies for `horizon`
    more periods.

    Reported per lag k = 0..horizon, pooled over deviation events:
      d_beta_rival[k]  change in the rivals' trading intensity, E[v dx]/Var(v).
                       Positive = rivals trade harder = punishment.
      d_beta_dev[k]    same for the deviator.
      d_profit_dev[k], d_profit_rival[k]   per-period profit differences.
    and cum_gain_dev: discounted sum of the deviator's profit differences over
    the window. Negative means the deviation does not pay.
    """
    I = env.I
    rivals = [i for i in range(I) if i != deviator]
    K = horizon + 1
    var_v = float(np.mean(env.values**2))

    vdx_rival = np.zeros((reps, K, env.S))
    vdx_dev = np.zeros((reps, K, env.S))
    dpi_dev = np.zeros((reps, K, env.S))
    dpi_rival = np.zeros((reps, K, env.S))
    deviated = np.zeros((reps, env.S), dtype=bool)

    for r in range(reps):
        for _ in range(gap):
            _, obs, _ = env.step(agent.act(obs, 0, greedy=True))

        e_dev = copy.deepcopy(env)
        o_dev = {k: v.copy() for k, v in obs.items()}
        o_base = obs
        for k in range(K):
            a_b = agent.act(o_base, 0, greedy=True)
            a_d = agent.act(o_dev, 0, greedy=True)
            if k == 0:
                others = e_dev.grid[a_d[:, rivals]].sum(1)
                a_d = a_d.copy()
                a_d[:, deviator] = _best_response_idx(e_dev, others)
                deviated[r] = a_d[:, deviator] != a_b[:, deviator]
            r_b, o_base, i_b = env.step(a_b)
            r_d, o_dev, i_d = e_dev.step(a_d)
            v = i_b["v"]
            dx = i_d["x"] - i_b["x"]
            vdx_dev[r, k] = v * dx[:, deviator]
            vdx_rival[r, k] = v * dx[:, rivals].mean(1)
            dpi_dev[r, k] = r_d[:, deviator] - r_b[:, deviator]
            dpi_rival[r, k] = (r_d[:, rivals] - r_b[:, rivals]).mean(1)
        obs = o_base

    mask = deviated  # (reps, S)
    n = int(mask.sum())

    def profile(a):
        vals = a.transpose(1, 0, 2)[:, mask]  # (K, n)
        mean = vals.mean(1) if n else np.full(K, np.nan)
        se = vals.std(1, ddof=1) / math.sqrt(n) if n > 1 else np.full(K, np.nan)
        return mean, 1.96 * se

    disc = gamma ** np.arange(K)
    gains = (dpi_dev * disc[None, :, None]).sum(1)[mask]  # (n,)

    out = {"n_events": n, "share_deviated": float(mask.mean()), "horizon": horizon}
    for name, arr, scale in (
        ("d_beta_rival", vdx_rival, 1.0 / var_v),
        ("d_beta_dev", vdx_dev, 1.0 / var_v),
        ("d_profit_dev", dpi_dev, 1.0),
        ("d_profit_rival", dpi_rival, 1.0),
    ):
        m, ci = profile(arr)
        out[name] = (m * scale).tolist()
        out[name + "_ci95"] = (ci * scale).tolist()
    out["cum_gain_dev"] = float(gains.mean()) if n else float("nan")
    out["cum_gain_dev_ci95"] = (
        float(1.96 * gains.std(ddof=1) / math.sqrt(n)) if n > 1 else float("nan")
    )
    out["gain_dev_period0"] = float(dpi_dev[:, 0][mask].mean()) if n else float("nan")
    return out
