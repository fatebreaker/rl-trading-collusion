"""Compiled training loop for tabular Q-learning in the Kyle market.

`fast_train(env, agent, start, stop)` advances training from period `start`
to `stop` with the same model and learning rule as KyleMarket.step +
TabularQ.observe (standard "taken" updates, constant step size, separate or
shared tables), but each session runs in its own compiled loop and sessions
run in parallel. It reads and writes the env's and agent's arrays in place,
so evaluation, diagnostics and checkpoints use the ordinary numpy code.

Random numbers come from a per-session splitmix64 stream stored on the env
(`env.fast_rng`), so results do not depend on the number of threads and a
checkpointed run resumes exactly. The draws differ from the numpy engine, so
the two engines agree in distribution, not path by path
(tests/test_fast.py checks this).

Supported: memory none / flow / residual / orders / price (all binnings) /
value; calendar-time or value-specific exploration counters;
wide or bracket grids; xi, theta; passive traders; adaptive or fixed market
maker. Not supported (use the numpy engine): random memory, ticks, disclosure
noise, counterfactual updates, visit-based step sizes.
"""

from __future__ import annotations

import math

import numpy as np

try:
    import numba as nb
except ImportError:  # pragma: no cover
    nb = None

MEM_CODES = {"none": 0, "flow": 1, "residual": 2, "orders": 3, "price": 4, "value": 5}

_GOLD = np.uint64(0x9E3779B97F4A7C15)
_M1 = np.uint64(0xBF58476D1CE4E5B9)
_M2 = np.uint64(0x94D049BB133111EB)


def supported(env, agent) -> str:
    """Empty string if the fast engine can run this configuration, else why not."""
    cfg = env.cfg
    if nb is None:
        return "numba is not installed"
    if getattr(agent, "name", "") != "q":
        return "only tabular Q-learning"
    if agent.update != "taken" or agent.alpha_schedule != "const":
        return "only standard updates with a constant step size"
    if cfg.memory not in MEM_CODES:
        return f"memory '{cfg.memory}'"
    if cfg.tick > 0 or cfg.disclosure_noise > 0:
        return "ticks and disclosure noise"
    return ""


def seed_streams(n: int, seed: int) -> np.ndarray:
    ss = np.random.SeedSequence([seed, 104729])
    return ss.generate_state(n, dtype=np.uint64)


if nb is not None:

    @nb.njit(inline="always")
    def _next(st, k):
        st[k] += _GOLD
        z = st[k]
        z = (z ^ (z >> np.uint64(30))) * _M1
        z = (z ^ (z >> np.uint64(27))) * _M2
        return z ^ (z >> np.uint64(31))

    @nb.njit(inline="always")
    def _unif(st, k):
        return (_next(st, k) >> np.uint64(11)) * (1.0 / 9007199254740992.0)

    @nb.njit(inline="always")
    def _randint(st, k, n):
        r = int(_unif(st, k) * n)
        return r if r < n else n - 1

    @nb.njit(inline="always")
    def _normal(st, k):
        u1 = _unif(st, k)
        u2 = _unif(st, k)
        if u1 < 1e-300:
            u1 = 1e-300
        return math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)

    @nb.njit(inline="always")
    def _digitize(x, edges):
        # number of edges <= x  (np.digitize with increasing edges, right=False)
        lo, hi = 0, edges.shape[0]
        while lo < hi:
            mid = (lo + hi) // 2
            if edges[mid] <= x:
                lo = mid + 1
            else:
                hi = mid
        return lo

    @nb.njit(inline="always")
    def _state(mem, i, s, prev_v_idx, prev_flow, prev_resid, prev_rival_idx, prev_price, m_v,
               edges_flow, edges_resid, n_flow_bins, n_actions, price_grid, p_lo, p_span,
               edges_unit, edges_price, expected_coef, values, sd_price, n_price_bins,
               dou_lo, dou_step):
        if mem == 0:
            return 0
        if mem == 1:
            return _digitize(prev_flow[s], edges_flow)
        k = prev_v_idx[s]
        if mem == 2:
            return k * n_flow_bins + _digitize(prev_resid[s, i], edges_resid)
        if mem == 3:
            return k * n_actions + prev_rival_idx[s, i]
        if mem == 5:
            return k
        if price_grid == 2:
            j = int(np.rint((prev_price[s] - dou_lo) / dou_step))
            if j < 0:
                j = 0
            if j > n_price_bins - 1:
                j = n_price_bins - 1
            return k * n_price_bins + j
        if price_grid == 1:
            sur = (prev_price[s] - m_v[s] - p_lo[k]) / p_span[k]
            return k * n_price_bins + _digitize(sur, edges_unit)
        sur = (prev_price[s] - expected_coef * values[k]) / sd_price
        return k * n_price_bins + _digitize(sur, edges_price)

    @nb.njit(parallel=True, cache=True)
    def _train(Q, shared, alpha, gamma, beta_decay, start, stop,
               rng, v_idx, prev_v_idx, prev_flow, prev_resid, prev_rivals, prev_rival_idx,
               prev_price, m_v, m_y, m_yy, m_vy, lam,
               values, grid_v, passive_beta, sigma_u, mm_fixed, mm_decay, xi, theta,
               mem, edges_flow, edges_resid, n_flow_bins, price_grid, p_lo, p_span, edges_unit,
               edges_price, expected_coef, sd_price, n_price_bins, dou_lo, dou_step,
               by_value, vcount):
        S = Q.shape[0]
        I = prev_resid.shape[1]
        A = grid_v.shape[1]
        NV = values.shape[0]
        for s in nb.prange(S):
            st = np.empty(I, np.int64)
            st2 = np.empty(I, np.int64)
            a = np.empty(I, np.int64)
            x = np.empty(I)
            for i in range(I):
                st[i] = _state(mem, i, s, prev_v_idx, prev_flow, prev_resid, prev_rival_idx,
                               prev_price, m_v, edges_flow, edges_resid, n_flow_bins, A,
                               price_grid, p_lo, p_span, edges_unit, edges_price,
                               expected_coef, values, sd_price, n_price_bins, dou_lo, dou_step)
            for t in range(start, stop):
                k = v_idx[s]
                if by_value:
                    eps = math.exp(-beta_decay * vcount[s, k])
                    vcount[s, k] += 1
                else:
                    eps = math.exp(-beta_decay * t)
                for i in range(I):
                    tb = 0 if shared else i
                    best, bq = 0, Q[s, tb, st[i], k, 0]
                    for j in range(1, A):
                        q = Q[s, tb, st[i], k, j]
                        if q > bq:
                            best, bq = j, q
                    if _unif(rng, s) < eps:
                        best = _randint(rng, s, A)
                    a[i] = best
                    x[i] = grid_v[k, best]
                v = values[k]
                u = sigma_u * _normal(rng, s)
                y = passive_beta * v + u
                tot_a = 0
                for i in range(I):
                    y += x[i]
                    tot_a += a[i]
                p = m_v[s] + lam[s] * (y - m_y[s])
                if not mm_fixed:
                    m_v[s] += mm_decay * (v - m_v[s])
                    m_y[s] += mm_decay * (y - m_y[s])
                    m_yy[s] += mm_decay * (y * y - m_yy[s])
                    m_vy[s] += mm_decay * (v * y - m_vy[s])
                    var_y = m_yy[s] - m_y[s] * m_y[s]
                    if var_y < 1e-8:
                        var_y = 1e-8
                    lb = (m_vy[s] - m_v[s] * m_y[s]) / var_y
                    lam[s] = lb if xi == 0.0 else (theta * lb + xi) / (theta + xi * xi)
                prev_v_idx[s] = k
                prev_flow[s] = y
                sx = 0.0
                for i in range(I):
                    sx += x[i]
                for i in range(I):
                    prev_resid[s, i] = y - x[i]
                    prev_rivals[s, i] = sx - x[i]
                    if I > 1:
                        prev_rival_idx[s, i] = int(np.rint((tot_a - a[i]) / (I - 1)))
                prev_price[s] = p
                k2 = _randint(rng, s, NV)
                v_idx[s] = k2
                for i in range(I):
                    st2[i] = _state(mem, i, s, prev_v_idx, prev_flow, prev_resid, prev_rival_idx,
                                    prev_price, m_v, edges_flow, edges_resid, n_flow_bins, A,
                                    price_grid, p_lo, p_span, edges_unit, edges_price,
                                    expected_coef, values, sd_price, n_price_bins, dou_lo, dou_step)
                if shared:
                    # sequential: each trader's update sees the previous one's
                    for i in range(I):
                        nxt = Q[s, 0, st2[i], k2, 0]
                        for j in range(1, A):
                            if Q[s, 0, st2[i], k2, j] > nxt:
                                nxt = Q[s, 0, st2[i], k2, j]
                        r = (v - p) * x[i]
                        Q[s, 0, st[i], k, a[i]] += alpha * (r + gamma * nxt - Q[s, 0, st[i], k, a[i]])
                else:
                    # simultaneous targets (tables are separate, so order is irrelevant)
                    for i in range(I):
                        nxt = Q[s, i, st2[i], k2, 0]
                        for j in range(1, A):
                            if Q[s, i, st2[i], k2, j] > nxt:
                                nxt = Q[s, i, st2[i], k2, j]
                        r = (v - p) * x[i]
                        Q[s, i, st[i], k, a[i]] += alpha * (r + gamma * nxt - Q[s, i, st[i], k, a[i]])
                for i in range(I):
                    st[i] = st2[i]


def fast_train(env, agent, start: int, stop: int, seed: int = 0) -> None:
    why = supported(env, agent)
    if why:
        raise ValueError(f"fast engine does not support {why}")
    if getattr(env, "fast_rng", None) is None:
        env.fast_rng = seed_streams(env.S, seed)
    cfg, b = env.cfg, env.bench
    for name in ("v_idx", "prev_v_idx", "prev_rival_idx"):
        setattr(env, name, np.ascontiguousarray(getattr(env, name), dtype=np.int64))
    for name in ("prev_flow", "prev_resid", "prev_rivals", "prev_price", "m_v", "m_y", "m_yy", "m_vy", "lam"):
        setattr(env, name, np.ascontiguousarray(getattr(env, name), dtype=np.float64))
    price_grid = {"noise": 0, "grid": 1, "dou": 2}[getattr(cfg, "price_bins", "noise")]
    by_value = bool(getattr(agent, "explore_by_value", False))
    vcount = agent.vcount if by_value else np.zeros((1, 1), dtype=np.int64)
    expected_coef = b.lam_nash * (b.agg_nash + env.passive_beta)
    _train(
        agent.Q, agent.shared, float(agent.alpha), float(agent.gamma), float(agent.beta_decay),
        int(start), int(stop),
        env.fast_rng, env.v_idx, env.prev_v_idx, env.prev_flow, env.prev_resid, env.prev_rivals,
        env.prev_rival_idx, env.prev_price, env.m_v, env.m_y, env.m_yy, env.m_vy, env.lam,
        np.ascontiguousarray(env.values, dtype=np.float64),
        np.ascontiguousarray(env.grid_v, dtype=np.float64),
        float(env.passive_beta), float(cfg.sigma_u), bool(cfg.mm_fixed), float(env._mm_decay),
        float(cfg.xi), float(cfg.theta),
        MEM_CODES[cfg.memory],
        np.ascontiguousarray(env._edges_flow, dtype=np.float64),
        np.ascontiguousarray(env._edges_resid, dtype=np.float64), int(cfg.n_flow_bins),
        price_grid,
        np.ascontiguousarray(env._p_lo, dtype=np.float64),
        np.ascontiguousarray(env._p_span, dtype=np.float64),
        np.ascontiguousarray(env._edges_unit, dtype=np.float64),
        np.ascontiguousarray(env._edges_price, dtype=np.float64),
        float(expected_coef), float(env.sd_price), int(cfg.n_price_bins),
        float(getattr(env, "_dou_lo", 0.0)), float(getattr(env, "_dou_step", 1.0)),
        by_value, vcount,
    )
    env.u_shock = np.zeros(env.S)
