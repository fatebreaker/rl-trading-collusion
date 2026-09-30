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
  xi, theta         information-insensitive investors and the market maker's
                    weight on pricing error versus inventory (Dou et al. 2025)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from statistics import NormalDist

import numpy as np

from .theory import Benchmarks, kyle_benchmarks

MEMORY_MODES = ("none", "flow", "residual", "orders", "price")
GRID_MODES = ("wide", "bracket")


@dataclass
class MarketConfig:
    n_informed: int = 2
    n_passive: int = 0
    sigma_v: float = 1.0
    sigma_u: float = 1.0
    n_values: int = 5
    n_actions: int = 31
    # Order grid.
    #   wide     one grid shared by all values, symmetric around zero, reaching
    #            grid_scale * beta_nash * max|v|. Lets traders over-trade beyond
    #            Nash or trade the wrong way.
    #   bracket  Dou et al. (2025): for each value v_k, n_actions orders evenly
    #            spanning [x^M_k - iota (x^N_k - x^M_k), x^N_k + iota (x^N_k - x^M_k)],
    #            i.e. only the collusive-to-Nash range plus a margin.
    grid_mode: str = "wide"
    grid_scale: float = 1.5
    bracket_iota: float = 0.1
    order_cap: float | None = None
    tick: float = 0.0
    disclosure_noise: float = 0.0
    # Information-insensitive investors z = -xi (p - v_bar); the market maker
    # sets lambda = (theta * lambda_B + xi) / (theta + xi^2). xi = 0 is Kyle.
    xi: float = 0.0
    theta: float = 0.1
    # What a trader remembers about the previous period.
    #   none      memoryless: no way to punish, collusion must be a learning bias
    #   flow      binned aggregate order flow y_{t-1}
    #   residual  (v_{t-1}, binned y_{t-1} - own x_{t-1}); lets a trader see
    #             what everyone else plus noise traders did
    #   orders    (v_{t-1}, rivals' order index at t-1); perfect monitoring,
    #             the easiest case for punishment strategies. Exact for I = 2;
    #             for I > 2 the rivals' mean order index is used.
    #   price     (v_{t-1}, binned p_{t-1}); the state used by Dou et al. The
    #             price is binned as a surprise relative to the Nash-expected
    #             price given v_{t-1}, in units of the noise-driven price sd.
    memory: str = "residual"
    n_flow_bins: int = 7
    n_price_bins: int = 15
    # Market maker re-estimates lambda_B with exponentially weighted moments.
    mm_halflife: float = 2000.0
    mm_fixed: bool = False  # freeze lambda at the Nash value (ablation)
    extra: dict = field(default_factory=dict)

    def __post_init__(self):
        if self.memory not in MEMORY_MODES:
            raise ValueError(f"memory must be one of {MEMORY_MODES}")
        if self.grid_mode not in GRID_MODES:
            raise ValueError(f"grid_mode must be one of {GRID_MODES}")


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
        self._build_static()
        self.u_shock = np.zeros(self.S)

    # ----------------------------------------------------------- static setup
    def _build_static(self) -> None:
        """Everything derived from the config alone (grids, bins, scales)."""
        cfg = self.cfg
        self.bench: Benchmarks = kyle_benchmarks(
            cfg.n_informed, cfg.sigma_v, cfg.sigma_u, cfg.n_passive, cfg.xi, cfg.theta
        )
        b = self.bench
        self.values = value_grid(cfg.n_values, cfg.sigma_v)
        vmax = np.abs(self.values).max()

        if cfg.grid_mode == "wide":
            max_order = cfg.grid_scale * b.beta_nash * vmax
            grid = np.linspace(-max_order, max_order, cfg.n_actions)
            if cfg.order_cap is not None:
                grid = np.clip(grid, -cfg.order_cap, cfg.order_cap)
            self.grid = grid
            self.grid_v = np.broadcast_to(grid, (cfg.n_values, cfg.n_actions)).copy()
        else:
            xn, xm = b.beta_nash * self.values, b.beta_coll * self.values
            span = xn - xm
            ends = np.stack([xm - cfg.bracket_iota * span, xn + cfg.bracket_iota * span], 1)
            lo, hi = ends.min(1), ends.max(1)
            t = np.linspace(0.0, 1.0, cfg.n_actions)
            self.grid_v = lo[:, None] + (hi - lo)[:, None] * t[None, :]
            if cfg.order_cap is not None:
                self.grid_v = np.clip(self.grid_v, -cfg.order_cap, cfg.order_cap)
            self.grid = None  # no shared grid in bracket mode
        self.passive_beta = cfg.n_passive * b.beta_nash

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
        # Price surprise scale: noise-driven price sd under Nash pricing.
        self.sd_price = max(b.lam_nash * np.sqrt(cfg.sigma_u**2 + cfg.disclosure_noise**2), 1e-12)
        nb = cfg.n_flow_bins
        self._edges_flow = np.linspace(-2, 2, nb + 1)[1:-1] * self.sd_flow
        self._edges_resid = np.linspace(-2, 2, nb + 1)[1:-1] * self.sd_resid
        npb = cfg.n_price_bins
        self._edges_price = np.linspace(-2.5, 2.5, npb + 1)[1:-1]

        if cfg.memory == "none":
            self.n_states = 1
        elif cfg.memory == "flow":
            self.n_states = nb
        elif cfg.memory == "residual":
            self.n_states = cfg.n_values * nb
        elif cfg.memory == "orders":
            self.n_states = cfg.n_values * cfg.n_actions
        else:
            self.n_states = cfg.n_values * npb

        self.n_actions = cfg.n_actions
        self.n_values = cfg.n_values
        self.feat_dim = 3
        self._mm_decay = 1.0 - 0.5 ** (1.0 / cfg.mm_halflife)

    def __setstate__(self, state):
        # Checkpoints written before a field existed still load: rebuild the
        # static arrays and give new dynamic fields neutral values.
        self.__dict__.update(state)
        self._build_static()
        if "u_shock" not in state:
            self.u_shock = np.zeros(self.S)
        if "prev_rival_idx" not in state:
            self.prev_rival_idx = np.zeros((self.S, self.I), dtype=np.int64)
            self.prev_price = np.zeros(self.S)

    def orders(self, v_idx: np.ndarray, actions: np.ndarray) -> np.ndarray:
        """Order sizes (S, I) for action indices (S, I) given value indices (S,)."""
        return self.grid_v[v_idx[:, None], actions]

    def nearest_action(self, v_idx: np.ndarray, x: np.ndarray) -> np.ndarray:
        """Grid index closest to order size x (S,) at value v_idx (S,)."""
        return np.abs(self.grid_v[v_idx] - x[:, None]).argmin(1)

    # ------------------------------------------------------------------ state
    def reset(self) -> dict:
        S, I = self.S, self.I
        b = self.bench
        # Start the market maker at the Nash lambda_B, with moments consistent
        # with Nash play so the first estimates are not wild.
        total_n = b.agg_nash + self.passive_beta
        var_y = total_n**2 * self.cfg.sigma_v**2 + self.cfg.sigma_u**2
        # (at xi = 0 use the closed-form value so runs stay bit-identical)
        lam_b = b.lam_nash if self.cfg.xi == 0.0 else total_n * self.cfg.sigma_v**2 / var_y
        self.m_v = np.zeros(S)
        self.m_y = np.zeros(S)
        self.m_yy = np.full(S, var_y)
        self.m_vy = np.full(S, lam_b * var_y)
        self.lam = np.full(S, self._price_impact(lam_b))

        self.v_idx = self.rng.integers(self.n_values, size=S)
        self.prev_v_idx = self.rng.integers(self.n_values, size=S)
        self.prev_resid = np.zeros((S, I))
        self.prev_flow = np.zeros(S)
        self.prev_rivals = np.zeros((S, I))
        self.prev_rival_idx = np.full((S, I), self.n_actions // 2, dtype=np.int64)
        self.prev_price = np.zeros(S)
        self.u_shock = np.zeros(S)
        return self._obs()

    def _price_impact(self, lam_b):
        cfg = self.cfg
        if cfg.xi == 0.0:
            return lam_b
        return (cfg.theta * lam_b + cfg.xi) / (cfg.theta + cfg.xi**2)

    def _obs(self) -> dict:
        S, I = self.S, self.I
        cfg = self.cfg
        prev_v = self.values[self.prev_v_idx] / cfg.sigma_v
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
            s = self.prev_v_idx[:, None] * cfg.n_actions + self.prev_rival_idx
            f_prev_v = np.broadcast_to(prev_v[:, None], (S, I))
            f_mem = self.prev_rival_idx / max(cfg.n_actions - 1, 1) * 2.0 - 1.0
        elif cfg.memory == "price":
            b = self.bench
            expected = b.lam_nash * (b.agg_nash + self.passive_beta) * self.values[self.prev_v_idx]
            surprise = (self.prev_price - expected) / self.sd_price
            pbin = np.digitize(surprise, self._edges_price)
            s = np.broadcast_to(
                (self.prev_v_idx * cfg.n_price_bins + pbin)[:, None], (S, I)
            ).copy()
            f_prev_v = np.broadcast_to(prev_v[:, None], (S, I))
            f_mem = np.broadcast_to(surprise[:, None], (S, I))
        else:
            rbin = np.digitize(self.prev_resid, self._edges_resid)
            s = self.prev_v_idx[:, None] * cfg.n_flow_bins + rbin
            f_prev_v = np.broadcast_to(prev_v[:, None], (S, I))
            f_mem = self.prev_resid / self.sd_resid

        v = self.values[self.v_idx] / cfg.sigma_v
        feat = np.stack(
            [np.broadcast_to(v[:, None], (S, I)), f_prev_v, f_mem], axis=-1
        ).astype(np.float32)
        return {"s": s, "v_idx": self.v_idx.copy(), "feat": feat}

    # ------------------------------------------------------------------- step
    def step(self, actions: np.ndarray) -> tuple[np.ndarray, dict, dict]:
        """actions: (S, I) integer indices into the order grid for the current value."""
        return self._advance(self.orders(self.v_idx, actions), actions)

    def step_orders(self, x: np.ndarray) -> tuple[np.ndarray, dict, dict]:
        """Advance one period with explicit order sizes (S, I)."""
        idx = np.stack([self.nearest_action(self.v_idx, x[:, i]) for i in range(self.I)], 1)
        return self._advance(x, idx)

    def _advance(self, x: np.ndarray, actions: np.ndarray) -> tuple[np.ndarray, dict, dict]:
        cfg = self.cfg
        v = self.values[self.v_idx]
        u = self.rng.normal(0.0, cfg.sigma_u, size=self.S) + self.u_shock
        self.u_shock = np.zeros(self.S)
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
            self.lam = self._price_impact((self.m_vy - self.m_v * self.m_y) / var_y)

        y_obs = y
        if cfg.disclosure_noise > 0:
            y_obs = y + self.rng.normal(0.0, cfg.disclosure_noise, size=self.S)

        info = {"v": v, "y": y, "p": p, "x": x, "lam": self.lam.copy()}

        self.prev_v_idx = self.v_idx
        self.prev_flow = y_obs
        self.prev_resid = y_obs[:, None] - x
        self.prev_rivals = x.sum(axis=1, keepdims=True) - x
        if self.I > 1:
            tot = actions.sum(axis=1, keepdims=True)
            self.prev_rival_idx = np.rint((tot - actions) / (self.I - 1)).astype(np.int64)
        self.prev_price = p if cfg.disclosure_noise == 0 else p + self.lam * (y_obs - y)
        self.v_idx = self.rng.integers(self.n_values, size=self.S)
        return profit, self._obs(), info
