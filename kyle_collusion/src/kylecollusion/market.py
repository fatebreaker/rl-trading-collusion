"""Batched repeated Kyle market.

Every period a fresh asset value is drawn, informed traders submit orders, the
market maker prices the aggregate flow with a linear rule it keeps re-estimating,
and the value is revealed. The one-period game repeats forever, which is what
lets learning traders sustain collusion with trigger strategies.

S independent sessions run side by side as numpy arrays, so one `step` call
advances all of them. Sessions never interact; batching is purely for speed.

Market-design interventions are config fields so that the same code produces
the baseline and every treatment:
  sigma_u           noise trading volume
  n_passive         non-learning informed traders playing the Nash strategy
  tick              price grid (prices rounded to a multiple of tick)
  order_cap         position limit on each learner's order
  disclosure_noise  noise added to the order flow traders observe
                    (less transparent markets are harder to monitor)
"""

from __future__ import annotations

from dataclasses import dataclass, field

from statistics import NormalDist

import numpy as np

from .theory import Benchmarks, kyle_benchmarks

MEMORY_MODES = ("none", "flow", "residual", "orders")


@dataclass
class MarketConfig:
    n_informed: int = 2
    n_passive: int = 0
    sigma_v: float = 1.0
    sigma_u: float = 1.0
    n_values: int = 5
    n_actions: int = 31
    # Largest order on the grid, in units of sigma_v * beta_nash * max|v|.
    # 1.5 leaves room for over-trading beyond the Nash benchmark.
    grid_scale: float = 1.5
    order_cap: float | None = None
    tick: float = 0.0
    disclosure_noise: float = 0.0
    # What a trader remembers about the previous period.
    #   none      memoryless: no way to punish, collusion must be a learning bias
    #   flow      binned aggregate order flow y_{t-1}
    #   residual  (v_{t-1}, binned y_{t-1} - own x_{t-1}); lets a trader see
    #             what everyone else plus noise traders did
    #   orders    (v_{t-1}, rivals' exact total order x_{t-1}); perfect
    #             monitoring, the easiest case for punishment strategies.
    #             Exact for I = 2 (one bin per grid order); for I > 2 the rivals'
    #             total is binned on an (I-1)-scaled grid of n_actions points.
    memory: str = "residual"
    n_flow_bins: int = 7
    # Market maker re-estimates lambda with exponentially weighted moments.
    mm_halflife: float = 2000.0
    mm_fixed: bool = False  # freeze lambda at the Nash value (ablation)
    extra: dict = field(default_factory=dict)

    def __post_init__(self):
        if self.memory not in MEMORY_MODES:
            raise ValueError(f"memory must be one of {MEMORY_MODES}")


def value_grid(n: int, sigma_v: float) -> np.ndarray:
    """Equiprobable discrete approximation of N(0, sigma_v^2), rescaled so its
    variance is exactly sigma_v^2 (the benchmarks only use second moments)."""
    if n == 1:
        return np.zeros(1)
    q = (np.arange(n) + 0.5) / n
    g = np.array([NormalDist().inv_cdf(qi) for qi in q])
    g -= g.mean()
    return g * sigma_v / g.std()


class KyleMarket:
    def __init__(self, cfg: MarketConfig, n_sessions: int, seed: int = 0):
        self.cfg = cfg
        self.S = n_sessions
        self.I = cfg.n_informed
        self.rng = np.random.default_rng(seed)
        self.bench: Benchmarks = kyle_benchmarks(
            cfg.n_informed, cfg.sigma_v, cfg.sigma_u, cfg.n_passive
        )

        self.values = value_grid(cfg.n_values, cfg.sigma_v)
        vmax = np.abs(self.values).max()
        max_order = cfg.grid_scale * self.bench.beta_nash * vmax
        grid = np.linspace(-max_order, max_order, cfg.n_actions)
        if cfg.order_cap is not None:
            grid = np.clip(grid, -cfg.order_cap, cfg.order_cap)
        self.grid = grid
        self.passive_beta = cfg.n_passive * self.bench.beta_nash

        # Scales used to bin / normalise what traders observe.
        b = self.bench
        n_total = cfg.n_informed + cfg.n_passive
        self.sd_flow = np.sqrt(
            (n_total * b.beta_nash) ** 2 * cfg.sigma_v**2
            + cfg.sigma_u**2
            + cfg.disclosure_noise**2
        )
        self.sd_resid = np.sqrt(
            ((n_total - 1) * b.beta_nash) ** 2 * cfg.sigma_v**2
            + cfg.sigma_u**2
            + cfg.disclosure_noise**2
        )
        nb = cfg.n_flow_bins
        self._edges_flow = np.linspace(-2, 2, nb + 1)[1:-1] * self.sd_flow
        self._edges_resid = np.linspace(-2, 2, nb + 1)[1:-1] * self.sd_resid

        rival_grid = (cfg.n_informed - 1) * np.linspace(grid.min(), grid.max(), cfg.n_actions)
        self._edges_rivals = (rival_grid[1:] + rival_grid[:-1]) / 2
        self.sd_rivals = max((cfg.n_informed - 1) * b.beta_nash * cfg.sigma_v, 1e-12)

        if cfg.memory == "none":
            self.n_states = 1
        elif cfg.memory == "flow":
            self.n_states = nb
        elif cfg.memory == "residual":
            self.n_states = cfg.n_values * nb
        else:
            self.n_states = cfg.n_values * cfg.n_actions

        self.n_actions = cfg.n_actions
        self.n_values = cfg.n_values
        self.feat_dim = 3
        self._mm_decay = 1.0 - 0.5 ** (1.0 / cfg.mm_halflife)

    # ------------------------------------------------------------------ state
    def reset(self) -> dict:
        S, I = self.S, self.I
        b = self.bench
        # Start the market maker at the Nash lambda, with moments consistent
        # with Nash play so the first estimates are not wild.
        var_y = (
            (b.agg_nash + self.passive_beta) ** 2 * self.cfg.sigma_v**2
            + self.cfg.sigma_u**2
        )
        self.m_v = np.zeros(S)
        self.m_y = np.zeros(S)
        self.m_yy = np.full(S, var_y)
        self.m_vy = np.full(S, b.lam_nash * var_y)
        self.lam = np.full(S, b.lam_nash)

        self.v_idx = self.rng.integers(self.n_values, size=S)
        self.prev_v_idx = self.rng.integers(self.n_values, size=S)
        self.prev_resid = np.zeros((S, I))
        self.prev_flow = np.zeros(S)
        self.prev_rivals = np.zeros((S, I))
        return self._obs()

    def _obs(self) -> dict:
        S, I = self.S, self.I
        cfg = self.cfg
        if cfg.memory == "none":
            s = np.zeros((S, I), dtype=np.int64)
            f_prev_v = np.zeros((S, I))
            f_mem = np.zeros((S, I))
        elif cfg.memory == "flow":
            s = np.broadcast_to(
                np.digitize(self.prev_flow, self._edges_flow)[:, None], (S, I)
            ).copy()
            f_prev_v = np.zeros((S, I))
            f_mem = np.broadcast_to((self.prev_flow / self.sd_flow)[:, None], (S, I))
        elif cfg.memory == "orders":
            obin = np.digitize(self.prev_rivals, self._edges_rivals)
            s = self.prev_v_idx[:, None] * cfg.n_actions + obin
            f_prev_v = np.broadcast_to(
                (self.values[self.prev_v_idx] / cfg.sigma_v)[:, None], (S, I)
            )
            f_mem = self.prev_rivals / self.sd_rivals
        else:
            rbin = np.digitize(self.prev_resid, self._edges_resid)
            s = self.prev_v_idx[:, None] * cfg.n_flow_bins + rbin
            f_prev_v = np.broadcast_to(
                (self.values[self.prev_v_idx] / cfg.sigma_v)[:, None], (S, I)
            )
            f_mem = self.prev_resid / self.sd_resid

        v = self.values[self.v_idx] / cfg.sigma_v
        feat = np.stack(
            [np.broadcast_to(v[:, None], (S, I)), f_prev_v, f_mem], axis=-1
        ).astype(np.float32)
        return {"s": s, "v_idx": self.v_idx.copy(), "feat": feat}

    # ------------------------------------------------------------------- step
    def step(self, actions: np.ndarray) -> tuple[np.ndarray, dict, dict]:
        """actions: (S, I) integer indices into the order grid."""
        return self.step_orders(self.grid[actions])

    def step_orders(self, x: np.ndarray) -> tuple[np.ndarray, dict, dict]:
        """Advance one period with explicit order sizes (S, I)."""
        cfg = self.cfg
        v = self.values[self.v_idx]
        u = self.rng.normal(0.0, cfg.sigma_u, size=self.S)
        y = x.sum(axis=1) + self.passive_beta * v + u

        p = self.m_v + self.lam * (y - self.m_y)
        if cfg.tick > 0:
            p = np.round(p / cfg.tick) * cfg.tick
        profit = (v - p)[:, None] * x

        if not cfg.mm_fixed:
            a = self._mm_decay
            self.m_v += a * (v - self.m_v)
            self.m_y += a * (y - self.m_y)
            self.m_yy += a * (y * y - self.m_yy)
            self.m_vy += a * (v * y - self.m_vy)
            var_y = np.maximum(self.m_yy - self.m_y**2, 1e-8)
            self.lam = (self.m_vy - self.m_v * self.m_y) / var_y

        y_obs = y
        if cfg.disclosure_noise > 0:
            y_obs = y + self.rng.normal(0.0, cfg.disclosure_noise, size=self.S)

        info = {"v": v, "y": y, "p": p, "x": x, "lam": self.lam.copy()}

        self.prev_v_idx = self.v_idx
        self.prev_flow = y_obs
        self.prev_resid = y_obs[:, None] - x
        self.prev_rivals = x.sum(axis=1, keepdims=True) - x
        self.v_idx = self.rng.integers(self.n_values, size=self.S)
        return profit, self._obs(), info
