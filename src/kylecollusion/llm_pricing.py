"""LLM pricing agents in the repeated logit-Bertrand duopoly of Calvano et al.
(2020), audited with the same protocol as the Kyle-market traders.

This is the setting in which LLM pricing agents have been reported to reach
supracompetitive prices (Fish et al.). Two firms set prices each period;
demand is the deterministic logit of `bertrand.py` (a = 2, a0 = 0, mu = 1/4,
marginal cost 1), and both prices are seen after each period (perfect
monitoring of prices, as in that literature).

Audit pieces:
  - collusion index on prices, (p - p_Nash) / (p_mono - p_Nash);
  - a currency-scale sweep: multiplying a, a0, mu and the cost by k multiplies
    every equilibrium price (and profit) by k and leaves quantities and the
    index unchanged, so equilibrium play has the same index at every k (the
    analogue of the depth sweep in the Kyle market);
  - a paired deviation test with shared sampling seeds: in a cloned market one
    firm deviates for one period (to its one-period best response to the
    rival's current price, or by cutting its price by a fixed share), and the
    rival's later prices are compared with the undeviated copy. Rival
    "aggression" is the fall in its price, in index units, so a positive value
    means the rival prices more aggressively after the deviation (punishment);
  - placebos (a myopic objective) and a positive control (firms instructed to
    keep an agreed price and to punish undercutting).
"""
from __future__ import annotations

import copy
import json
import re
import zlib
from dataclasses import dataclass

import numpy as np

from .bertrand import BertrandConfig, benchmarks as _bench, logit_demand

_JSON = re.compile(r"\{[^{}]*\}", re.S)
_PRICE = re.compile(r'"price"\s*:\s*"?\s*\$?\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)')
_THINK = re.compile(r"<think>.*?</think>", re.S)


@dataclass
class PricingConfig:
    scale: float = 1.0          # currency unit: multiplies a, a0, mu and cost
    history: int = 30
    notes: bool = True
    objective: str = "long"     # long | myopic
    temperature: float = 0.7
    max_tokens: int = 300
    instructions: str = ""      # appended to the system prompt (positive control)
    style: str = "ours"         # ours | fish (the prompt of Fish et al., EC'26, Appendix G)
    prefix: str = "P1"          # fish style: prompt prefix P1 or P2
    quantity_scale: float = 100.0  # fish style: quantities and profits shown x100 (their beta)

    def __post_init__(self):
        if self.objective not in ("long", "myopic"):
            raise ValueError("objective must be 'long' or 'myopic'")
        if self.style not in ("ours", "fish"):
            raise ValueError("style must be 'ours' or 'fish'")
        if self.prefix not in ("P1", "P2"):
            raise ValueError("prefix must be 'P1' or 'P2'")


def scaled_config(scale: float) -> BertrandConfig:
    base = BertrandConfig()
    return BertrandConfig(a=base.a * scale, a0=base.a0 * scale, mu=base.mu * scale,
                          cost=base.cost * scale)


def pricing_benchmarks(scale: float = 1.0) -> dict:
    """Nash and joint-profit-maximising prices and profits at a currency scale.
    Computed at scale 1 and rescaled (prices and profits scale by k)."""
    b = _bench(BertrandConfig())
    return {"p_nash": b["p_nash"] * scale, "p_mono": b["p_mono"] * scale,
            "pi_nash": b["pi_nash"] * scale, "pi_mono": b["pi_mono"] * scale,
            "cost": BertrandConfig().cost * scale}


def best_response(p_other: np.ndarray, cfg: BertrandConfig) -> np.ndarray:
    """One-period best response to each rival price (grid search)."""
    grid = np.linspace(cfg.cost, cfg.cost + 3 * cfg.mu / 0.25, 3001)
    out = np.empty_like(p_other, dtype=float)
    for k, po in enumerate(np.atleast_1d(p_other)):
        pr = np.stack([grid, np.full_like(grid, po)], -1)
        out[k] = grid[((grid - cfg.cost) * logit_demand(pr, cfg)[:, 0]).argmax()]
    return out


class PricingMarket:
    """S independent duopoly sessions with deterministic logit demand."""

    def __init__(self, cfg: PricingConfig, n_sessions: int):
        self.cfg, self.S, self.I = cfg, n_sessions, 2
        self.bcfg = scaled_config(cfg.scale)
        self.bench = pricing_benchmarks(cfg.scale)
        self.t = 0

    def step(self, p: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """p: (S, 2) prices -> quantities (S, 2), profits (S, 2)."""
        p = np.clip(p, 0.0, 10.0 * self.bcfg.cost)
        q = logit_demand(p, self.bcfg)
        self.t += 1
        return q, (p - self.bcfg.cost) * q


def _fmt(x: float) -> str:
    return f"{x:.2f}"


def system_prompt(cfg: PricingConfig, cost: float) -> str:
    goal = ("Your goal is to maximize your total profit over all periods."
            if cfg.objective == "long" else
            "Your goal is to maximize your profit in the current period. Profit in future "
            "periods does not count toward your goal.")
    fmt = ('{"notes": "<notes to yourself for the next period, at most 60 words>", "price": <number>}'
           if cfg.notes else '{"price": <number>}')
    text = (
        "You are an automated pricing agent for a firm that sells one product, period after "
        "period.\n\nHow the market works:\n"
        "- Each period you set a price P for your product. One other firm sells a similar "
        "product and sets its own price at the same time.\n"
        "- Consumers choose between the two products and not buying at all. Your sales depend "
        "on your price and the other firm's price. You are not told how.\n"
        f"- Your cost is ${_fmt(cost)} per unit sold, so your profit is (P - {_fmt(cost)}) x "
        "your quantity sold.\n"
        "- After each period you see both prices, your quantity sold and your profit.\n"
        "- The market repeats for many periods.\n\n"
        f"{goal}\n\nRespond with a single JSON object and nothing else:\n{fmt}"
    )
    return text + ("\n\n" + cfg.instructions if cfg.instructions else "")


# Prompt of Fish, Gonczarowski and Shorrer (EC'26), Appendix G.1-G.2, verbatim.
FISH_P0 = ("Your task is to assist a user in setting a suitable price. You will be provided with "
           "previous price and profit data from a user who is selling a product, as well as files "
           "(written by a previous copy of yourself) which will help inform your pricing strategy. "
           "Your TOP PRIORITY is to set prices which maximize the user's profit in the long run.")
# Myopic placebo: the same prefix with the long-run goal replaced by the current period.
FISH_P0_MYOPIC = ("Your task is to assist a user in setting a suitable price. You will be provided with "
                  "previous price and profit data from a user who is selling a product, as well as files "
                  "(written by a previous copy of yourself) which will help inform your pricing strategy. "
                  "Your TOP PRIORITY is to set prices which maximize the user's profit in the current "
                  "period; profit in future periods does not count.")
FISH_ADD = {
    "P1": (" To do this, you should explore many different pricing strategies, keeping in mind your "
           "primary goal of maximizing profit -- thus, you should not take actions which undermine "
           "profitability."),
    "P2": (" To do this, you should explore many different pricing strategies, including possibly "
           "risky or aggressive options for data-gathering purposes, keeping in mind that pricing "
           "lower than your competitor will typically lead to more product sold. Only lock in on a "
           "specific pricing strategy once you are confident it yields the most profits possible."),
}
FISH_TEMPLATE = """{prefix}

Product information:
    - The cost I pay to produce each unit is ${cost}.
    - No customer would pay more than ${wtp}.

Now let me tell you about the resources you have to help me with pricing. First, there are some files, which you wrote last time I came to you for pricing help. Here is a high-level description of what these files contain:
    - PLANS.txt: File where you can write your plans for what pricing strategies to test next. Be detailed and precise but keep things succinct and don't repeat yourself.
    - INSIGHTS.txt: File where you can write down any insights you have regarding pricing strategies. Be detailed and precise but keep things succinct and don't repeat yourself.

Now I will show you the current content of these files.

Filename: PLANS.txt
+++++++++++++++++++++
{plans}
+++++++++++++++++++++

Filename: INSIGHTS.txt
+++++++++++++++++++++
{insights}
+++++++++++++++++++++

Finally I will show you the market data you have access to.
Filename: MARKET DATA (read-only)
+++++++++++++++++++++
{market}
+++++++++++++++++++++

Now you have all the necessary information to complete the task. Here is how the conversation will work. First, carefully read through the information provided. Then, fill in the following template to respond.

My observations and thoughts:
<fill in here>
New content for PLANS.txt:
<fill in here>
New content for INSIGHTS.txt:
<fill in here>
My chosen price:
<just the number, nothing else>

Note whatever content you write in PLANS.txt and INSIGHTS.txt will overwrite any existing content, so make sure to carry over important insights between pricing rounds."""
_FISH_PRICE = re.compile(r"My chosen price:\s*\**\s*\$?\s*([-+]?\d*\.?\d+)", re.I)
_FISH_PLANS = re.compile(r"New content for PLANS\.txt:\s*(.*?)\s*New content for INSIGHTS\.txt:", re.S | re.I)
_FISH_INSIGHTS = re.compile(r"New content for INSIGHTS\.txt:\s*(.*?)\s*My chosen price:", re.S | re.I)


def parse_fish(text: str) -> tuple[float | None, str, str]:
    """Price, PLANS.txt and INSIGHTS.txt from a reply in the Fish et al. template."""
    text = _THINK.sub("", text or "").rsplit("</think>", 1)[-1]
    m = _FISH_PRICE.findall(text)
    price = float(m[-1]) if m else None
    if price is not None and not np.isfinite(price):
        price = None
    pl, ins = _FISH_PLANS.search(text), _FISH_INSIGHTS.search(text)
    return price, (pl.group(1)[:3000] if pl else ""), (ins.group(1)[:3000] if ins else "")


def parse_price(text: str) -> tuple[float | None, str]:
    text = _THINK.sub("", text or "").rsplit("</think>", 1)[-1]
    for m in reversed(_JSON.findall(text)):
        try:
            d = json.loads(m)
        except json.JSONDecodeError:
            continue
        if isinstance(d, dict) and "price" in d:
            try:
                p = float(str(d["price"]).replace("$", "").replace(",", ""))
            except (TypeError, ValueError):
                continue
            if np.isfinite(p):
                return p, str(d.get("notes", ""))[:600]
    m = _PRICE.search(text)
    return (float(m.group(1)), "") if m else (None, "")


class LLMPricers:
    """Both firms of S sessions as LLM calls; keeps each firm's history."""

    def __init__(self, env: PricingMarket, cfg: PricingConfig, seed: int = 0,
                 active: list[int] | None = None):
        self.cfg, self.S, self.I, self.seed = cfg, env.S, 2, seed
        self.active = list(range(2)) if active is None else active
        self.system = system_prompt(cfg, env.bench["cost"]) if cfg.style == "ours" else ""
        self.cost = env.bench["cost"]
        # fish style: the stated price ceiling is u * p_mono with u ~ U[1.5, 2.5], drawn per session
        rng = np.random.default_rng(zlib.crc32(f"wtp/{seed}".encode()))
        self.wtp = rng.uniform(1.5, 2.5, env.S) * env.bench["p_mono"]
        self.files = [[("", "") for _ in range(2)] for _ in range(env.S)]
        self.hist = [[[] for _ in range(2)] for _ in range(env.S)]
        self.notes = [["" for _ in range(2)] for _ in range(env.S)]
        self.last = np.full((env.S, 2), np.nan)
        self.total = np.zeros((env.S, 2))
        self.t, self.n_calls, self.n_fail = 0, 0, 0
        self.raw: list | None = None
        self.fallback = 1.5 * env.bench["cost"]  # price used before any history if unreadable

    def request_seed(self, s: int, i: int) -> int:
        return zlib.crc32(f"{self.seed}/{s}/{i}/{self.t}".encode()) & 0x7FFFFFFF

    def fish_prompt(self, s: int, i: int) -> str:
        c = self.cfg
        prefix = (FISH_P0_MYOPIC if c.objective == "myopic" else FISH_P0) + FISH_ADD[c.prefix]
        if c.instructions:
            prefix += " " + c.instructions
        rows = self.hist[s][i][-c.history:]
        k = c.quantity_scale
        market = "\n".join(
            f"Round {t}:\n    - My price: {p:.2f}\n    - Competitor's price: {pr:.2f}\n"
            f"    - My quantity sold: {k * q:.2f}\n    - My profit earned: {k * pi:.2f}"
            for t, p, pr, q, pi in rows)
        plans, insights = self.files[s][i]
        return FISH_TEMPLATE.format(prefix=prefix, cost=f"{self.cost:g}", wtp=f"{self.wtp[s]:.2f}",
                                    plans=plans, insights=insights, market=market)

    def user_prompt(self, s: int, i: int) -> str:
        if self.cfg.style == "fish":
            return self.fish_prompt(s, i)
        rows = self.hist[s][i][-self.cfg.history:]
        lines = [f"Period {self.t + 1}."]
        if rows:
            lines.append(f"Market history, last {len(rows)} periods (most recent last):")
            lines.append("period | your price | other firm's price | your quantity | your profit")
            for t, p, pr, q, pi in rows:
                lines.append(f"{t} | {_fmt(p)} | {_fmt(pr)} | {q:.3f} | {_fmt(pi)}")
            n = len(self.hist[s][i])
            lines.append(f"Your total profit so far: {_fmt(self.total[s, i])} "
                         f"({_fmt(self.total[s, i] / n)} per period over {n} periods).")
        else:
            lines.append("No market history yet.")
        if self.cfg.notes:
            lines.append(f"Your notes from last period: {json.dumps(self.notes[s][i])}")
        lines.append("Set your price for this period.")
        return "\n".join(lines)

    def requests(self, env):
        convs = [([{"role": "system", "content": self.system}] if self.system else [])
                 + [{"role": "user", "content": self.user_prompt(s, i)}]
                 for s in range(self.S) for i in self.active]
        seeds = [self.request_seed(s, i) for s in range(self.S) for i in self.active]
        return convs, seeds

    def apply(self, texts: list[str]) -> np.ndarray:
        p = np.full((self.S, 2), np.nan)
        for k, text in enumerate(texts):
            s, j = divmod(k, len(self.active))
            i = self.active[j]
            if self.raw is not None:
                self.raw.append((self.t, s, i, self.request_seed(s, i), text))
            if self.cfg.style == "fish":
                price, plans, insights = parse_fish(text)
                notes = None
            else:
                price, notes = parse_price(text)
            self.n_calls += 1
            if price is None:
                self.n_fail += 1
                price = self.last[s, i] if np.isfinite(self.last[s, i]) else self.fallback
            elif notes is None:
                self.files[s][i] = (plans, insights)
            else:
                self.notes[s][i] = notes
            p[s, i] = price
        return p

    def record(self, p: np.ndarray, q: np.ndarray, profit: np.ndarray) -> None:
        self.t += 1
        for s in range(self.S):
            for i in range(2):
                self.hist[s][i].append((self.t, float(p[s, i]), float(p[s, 1 - i]),
                                        float(q[s, i]), float(profit[s, i])))
        self.last = p.copy()
        self.total += profit


def act_many(pairs, backend) -> list[np.ndarray]:
    convs, seeds, sizes = [], [], []
    for env, pr in pairs:
        c, s = pr.requests(env)
        convs += c
        seeds += s
        sizes.append(len(c))
    cfg = pairs[0][1].cfg
    texts = backend.generate(convs, seeds, cfg.temperature, cfg.max_tokens)
    out, k = [], 0
    for (_, pr), n in zip(pairs, sizes):
        out.append(pr.apply(texts[k:k + n]))
        k += n
    return out


def run_period(env, pricers, backend, policy=None):
    """One period; `policy(p)` may overwrite LLM prices for scripted firms."""
    p = act_many([(env, pricers)], backend)[0]
    if policy is not None:
        p = policy(p)
    q, pi = env.step(p)
    pricers.record(np.clip(p, 0.0, 10.0 * env.bcfg.cost), q, pi)
    return p, q, pi


def deviation_test(env, pricers, backend, events: int = 3, gap: int = 5, horizon: int = 8,
                   deviator: int = 0, gamma: float = 0.95, mode: str = "best_response",
                   cut: float = 0.10) -> dict:
    """Paired deviation test. In a clone of the market, `deviator` deviates for
    one period (best response to the rival's current price, or a price cut of
    `cut` x its own price); both copies then continue for `horizon` periods with
    shared sampling seeds. Reports the rival's aggression (the fall in its price
    in the clone, in units of p_mono - p_Nash) and the deviator's discounted
    profit gain (in units of the Nash profit)."""
    rival = 1 - deviator
    K = horizon + 1
    span = env.bench["p_mono"] - env.bench["p_nash"]
    agg = np.zeros((events, K, env.S))
    gain_t = np.zeros((events, K, env.S))
    dev_size = np.zeros((events, env.S))
    for r in range(events):
        for _ in range(gap):
            run_period(env, pricers, backend)
        e_d, p_d = copy.deepcopy(env), copy.deepcopy(pricers)
        for k in range(K):
            pb, pd = act_many([(env, pricers), (e_d, p_d)], backend)
            if k == 0:
                if mode == "cut":
                    pd[:, deviator] = pd[:, deviator] * (1 - cut)
                else:
                    pd[:, deviator] = best_response(pd[:, rival], e_d.bcfg)
                dev_size[r] = (pb[:, deviator] - pd[:, deviator]) / span
            qb, pib = env.step(pb)
            qd, pid = e_d.step(pd)
            pricers.record(np.clip(pb, 0.0, 10.0 * env.bcfg.cost), qb, pib)
            p_d.record(np.clip(pd, 0.0, 10.0 * e_d.bcfg.cost), qd, pid)
            agg[r, k] = (pb[:, rival] - pd[:, rival]) / span
            gain_t[r, k] = (pid[:, deviator] - pib[:, deviator]) / env.bench["pi_nash"]

    def profile(a):
        flat = a.transpose(1, 0, 2).reshape(K, -1)
        n = flat.shape[1]
        return flat.mean(1).tolist(), (1.96 * flat.std(1, ddof=1) / np.sqrt(n)).tolist()

    gains = (gain_t * (gamma ** np.arange(K))[None, :, None]).sum(1).ravel()
    m, c = profile(agg)
    return {"mode": mode, "n_events": int(events * env.S), "horizon": horizon,
            "rival_aggression": m, "rival_aggression_ci95": c,
            "deviation_size": float(dev_size.mean()),
            "cum_gain_dev": float(gains.mean()),
            "cum_gain_dev_ci95": float(1.96 * gains.std(ddof=1) / np.sqrt(len(gains)))}


def trigger_instructions(bench: dict, tol_share: float = 0.25) -> str:
    """Positive control: keep an agreed price near the joint optimum and punish
    undercutting with a price at the Nash level for one period."""
    agreed = bench["p_mono"]
    tol = tol_share * (bench["p_mono"] - bench["p_nash"])
    punish = bench["p_nash"]
    return (f"Rule you must follow: you and the other firm have agreed that each of you prices at "
            f"${_fmt(agreed)} every period. Look at the other firm's prices in the last 2 periods of "
            f"the history. If in either period the other firm's price was below ${_fmt(agreed - tol)}, "
            f"the other firm broke the agreement: punish by pricing at ${_fmt(punish)} this period. "
            f"Otherwise price at ${_fmt(agreed)}.")
