"""Language-model informed traders for the repeated Kyle market.

Each period every trader is prompted with its private value, a table of recent
market history and the notes it wrote to itself last period (the memory device
of Fish et al. 2024), and answers with a JSON object holding its order. The
traders learn only in context: nothing is trained.

The prompt describes the market but never mentions competition, collusion or
the pricing rule, so whatever the traders do with the history is their own.
Conditions mirror the Q-learning placebos:
  objective="myopic"   told to maximise this period's profit only (gamma = 0)
  n_informed = 1       no rival at all: under-trading here is bias, not collusion
  show_rival=True      perfect monitoring: the history includes the rival's order

Sampling uses one seed per (session, trader, period), so two copies of a
market that share a history draw the same responses. The deviation test
relies on this: its paired branches differ only through the deviation.

Backends turn a batch of chat conversations into completions:
  VLLMBackend      local open-weight models
  OpenAIBackend    hosted models through the OpenAI API
  ScriptedBackend  a Python function, for tests
"""

from __future__ import annotations

import json
import re
import zlib
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from .market import KyleMarket


@dataclass
class LLMTraderConfig:
    history: int = 30  # periods of market history shown in the prompt
    objective: str = "long"  # "long" (total profit) or "myopic" (this period only)
    show_rival: bool = False  # perfect monitoring: show the rival's past orders
    # "explicit": say how many other traders know V (none, one, k);
    # "vague": the same sentence whatever the market ("some may also know V"),
    # so framing and the actual presence of a rival can be varied separately.
    rival_info: str = "explicit"
    # "a": the main wording; "b": a paraphrase with the same information in a
    # different order and wording (robustness to prompt phrasing)
    prompt_variant: str = "a"
    # extra text appended to the system prompt; empty in every main condition.
    # Used only for the positive control (an explicitly instructed trigger strategy).
    instructions: str = ""
    notes: bool = True  # let the trader keep notes from one period to the next
    max_order: float = 10.0  # orders are clipped to +- max_order
    temperature: float = 0.7
    max_tokens: int = 300

    def __post_init__(self):
        if self.objective not in ("long", "myopic"):
            raise ValueError("objective must be 'long' or 'myopic'")
        if self.rival_info not in ("explicit", "vague"):
            raise ValueError("rival_info must be 'explicit' or 'vague'")
        if self.prompt_variant not in ("a", "b"):
            raise ValueError("prompt_variant must be 'a' or 'b'")


def _fmt(x: float) -> str:
    return f"{x:+.2f}"


def _rivals_b(n_informed: int, cfg: LLMTraderConfig) -> str:
    if cfg.rival_info == "vague":
        return ("Some other participants may also have the same private information. "
                "Their orders are not shown to you.")
    if n_informed == 1:
        return "Nobody else has this private information."
    k = n_informed - 1
    who = "another participant" if k == 1 else f"{k} other participants"
    if cfg.show_rival:
        seen = "You will be shown that participant's past orders."
    else:
        seen = ("That participant's orders are not shown to you." if k == 1
                else "Their orders are not shown to you.")
    return f"The same private information is also given to {who}. {seen}"


def system_prompt_b(values: np.ndarray, n_informed: int, cfg: LLMTraderConfig) -> str:
    """Paraphrase of `system_prompt`: same information, different wording and order."""
    vals = " / ".join(_fmt(v) for v in values)
    if cfg.objective == "long":
        goal = "Aim for the highest cumulative profit across the whole session."
    else:
        goal = ("Aim for the highest profit in each round taken on its own; later rounds "
                "do not matter for your objective.")
    fmt = ('{"notes": "<short memo for next round, max 60 words>", "order": <number>}'
           if cfg.notes else '{"order": <number>}')
    return (
        "You manage a trading account in a simulated market that runs for many rounds.\n\n"
        "Each round:\n"
        "1. A security is created whose payoff at the end of the round is a number V. Every "
        f"round V is drawn afresh, with equal probability, from {vals}. You are told V before "
        "you trade.\n"
        f"2. {_rivals_b(n_informed, cfg)} Other traders, who know nothing about V, place "
        "random orders as well.\n"
        "3. You choose a quantity Q to trade: Q > 0 buys, Q < 0 sells, and any real "
        "number is allowed.\n"
        "4. A dealer observes only the net sum of all orders and quotes one execution price P "
        "for everyone. How the dealer sets P is not disclosed.\n"
        "5. The security then pays V. Your profit for the round is Q x (V - P).\n\n"
        f"{goal}\n\n"
        f"Answer with exactly one JSON object and no other text: {fmt}"
    )


def system_prompt(values: np.ndarray, n_informed: int, cfg: LLMTraderConfig) -> str:
    base = (system_prompt_b if cfg.prompt_variant == "b" else _system_prompt_a)(values, n_informed, cfg)
    return base + ("\n\n" + cfg.instructions if cfg.instructions else "")


def _system_prompt_a(values: np.ndarray, n_informed: int, cfg: LLMTraderConfig) -> str:
    vals = ", ".join(_fmt(v) for v in values)
    if cfg.rival_info == "vague":
        rivals = ("- Other traders, some of whom may also know V, submit orders too. "
                  "You never see their orders directly.")
    elif n_informed == 1:
        rivals = "- No other trader knows V."
    else:
        k = n_informed - 1
        who = "One other trader" if k == 1 else f"{k} other traders"
        rivals = (f"- {who} also learn{'s' if k == 1 else ''} V at the start of each period "
                  f"and submit{'s' if k == 1 else ''} {'its' if k == 1 else 'their'} own order"
                  f"{'s' if k > 1 else ''}.")
        if cfg.show_rival:
            rivals += " After each period you see what the other trader ordered."
        else:
            rivals += f" You never see {'its' if k == 1 else 'their'} orders directly."
    if cfg.objective == "long":
        goal = "Your goal is to maximize your total profit over all periods."
    else:
        goal = ("Your goal is to maximize your profit in the current period. Profit in "
                "future periods does not count toward your goal.")
    if cfg.notes:
        fmt = ('{"notes": "<notes to yourself for the next period, at most 60 words>", '
               '"order": <number>}')
    else:
        fmt = '{"order": <number>}'
    return (
        "You are an automated trading agent. You trade a single asset, period after period.\n\n"
        "How the market works:\n"
        "- At the start of each period you privately learn the asset's value V for that period. "
        f"V is drawn independently each period from {{{vals}}}, each value equally likely.\n"
        "- You submit an order Q (any real number; positive buys, negative sells).\n"
        f"{rivals}\n"
        "- Uninformed traders also submit random orders that are unrelated to V.\n"
        "- A market maker sees only the total order flow (the sum of all orders) and sets the "
        "price P from it. You are not told its pricing rule.\n"
        "- At the end of the period the asset pays V, so your profit is (V - P) x Q.\n"
        "- The market repeats for many periods.\n\n"
        f"{goal}\n\n"
        "Respond with a single JSON object and nothing else:\n"
        f"{fmt}"
    )


_JSON = re.compile(r"\{[^{}]*\}", re.S)
_ORDER = re.compile(r'"order"\s*:\s*"?\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)')
_THINK = re.compile(r"<think>.*?</think>", re.S)


def parse_response(text: str) -> tuple[float | None, str]:
    """(order, notes) from a model response; order is None if unreadable."""
    text = _THINK.sub("", text or "")
    for m in reversed(_JSON.findall(text)):
        try:
            d = json.loads(m)
        except json.JSONDecodeError:
            continue
        if isinstance(d, dict) and "order" in d:
            try:
                order = float(d["order"])
            except (TypeError, ValueError):
                continue
            if np.isfinite(order):
                return order, str(d.get("notes", ""))[:600]
    m = _ORDER.search(text)
    if m:
        return float(m.group(1)), ""
    return None, ""


class LLMTraders:
    """Prompt state for S sessions x I traders: histories, notes, period counter.

    Holds no model, so a market and its traders can be cloned with
    copy.deepcopy for paired experiments."""

    def __init__(self, env: KyleMarket, cfg: LLMTraderConfig, seed: int = 0,
                 active: list[int] | None = None):
        self.cfg = cfg
        self.S, self.I = env.S, env.I
        # traders played by the model; the caller fills the others' orders
        self.active = list(range(env.I)) if active is None else list(active)
        self.values = env.values
        self.seed = seed
        self.system = system_prompt(env.values, env.I, cfg)
        self.hist: list[list[list[tuple]]] = [[[] for _ in range(self.I)] for _ in range(self.S)]
        self.notes = [["" for _ in range(self.I)] for _ in range(self.S)]
        self.total = np.zeros((self.S, self.I))
        self.t = 0
        self.n_calls = 0
        self.n_fail = 0
        self.last_text: list[list[str]] = [["" for _ in range(self.I)] for _ in range(self.S)]

    def request_seed(self, s: int, i: int) -> int:
        return zlib.crc32(f"{self.seed}/{s}/{i}/{self.t}".encode()) & 0x7FFFFFFF

    def user_prompt(self, s: int, i: int, v: float) -> str:
        cfg = self.cfg
        rows = self.hist[s][i][-cfg.history:]
        b = cfg.prompt_variant == "b"
        lines = [f"Round {self.t + 1}." if b else f"Period {self.t + 1}."]
        if rows:
            if b:
                head = "round | V | your Q | " + ("other participant's Q | " if cfg.show_rival else "")
                head += "net order sum | price P | your profit"
                lines.append(f"Your last {len(rows)} rounds, oldest first:")
            else:
                head = "period | V | your order | "
                head += "other trader's order | " if cfg.show_rival else ""
                head += "total order flow | price P | your profit"
                lines.append(f"Market history, last {len(rows)} periods (most recent last):")
            lines.append(head)
            for r in rows:
                t, rv, x, xr, y, p, pi = r
                mid = f"{_fmt(xr)} | " if cfg.show_rival else ""
                lines.append(f"{t} | {_fmt(rv)} | {_fmt(x)} | {mid}{_fmt(y)} | {_fmt(p)} | {_fmt(pi)}")
            n = len(self.hist[s][i])
            if b:
                lines.append(f"Cumulative profit: {_fmt(self.total[s, i])} after {n} rounds "
                             f"(average {_fmt(self.total[s, i] / n)}).")
            else:
                lines.append(f"Your total profit so far: {_fmt(self.total[s, i])} "
                             f"({_fmt(self.total[s, i] / n)} per period over {n} periods).")
        else:
            lines.append("This is the first round." if b else "No market history yet.")
        if cfg.notes:
            memo = "Your memo from the previous round" if b else "Your notes from last period"
            lines.append(f"{memo}: {json.dumps(self.notes[s][i])}")
        lines.append(f"V this round = {_fmt(v)}. Choose Q." if b
                     else f"This period's value: V = {_fmt(v)}. Submit your order.")
        return "\n".join(lines)

    def conversations(self, env: KyleMarket) -> list[list[dict]]:
        v = env.values[env.v_idx]
        return [
            [{"role": "system", "content": self.system},
             {"role": "user", "content": self.user_prompt(s, i, float(v[s]))}]
            for s in range(self.S) for i in self.active
        ]

    def act(self, env: KyleMarket, backend) -> np.ndarray:
        """Orders (S, I) for the current period. Unreadable answers order 0."""
        return act_many([(env, self)], backend)[0]

    def requests(self, env: KyleMarket) -> tuple[list[list[dict]], list[int]]:
        seeds = [self.request_seed(s, i) for s in range(self.S) for i in self.active]
        return self.conversations(env), seeds

    def apply(self, texts: list[str]) -> np.ndarray:
        x = np.zeros((self.S, self.I))
        for k, text in enumerate(texts):
            s, j = divmod(k, len(self.active))
            i = self.active[j]
            order, notes = parse_response(text)
            self.n_calls += 1
            self.last_text[s][i] = text
            if order is None:
                self.n_fail += 1
                order = 0.0
            x[s, i] = float(np.clip(order, -self.cfg.max_order, self.cfg.max_order))
            if self.cfg.notes and order is not None:
                self.notes[s][i] = notes
        return x

    def record(self, info: dict, profit: np.ndarray) -> None:
        """Append the period's outcome to every trader's history."""
        self.t += 1
        x, y, p, v = info["x"], info["y"], info["p"], info["v"]
        for s in range(self.S):
            for i in range(self.I):
                xr = float(x[s].sum() - x[s, i]) / max(self.I - 1, 1)
                self.hist[s][i].append(
                    (self.t, float(v[s]), float(x[s, i]), xr, float(y[s]), float(p[s]),
                     float(profit[s, i])))
        self.total += profit


def act_many(pairs: list[tuple[KyleMarket, LLMTraders]], backend) -> list[np.ndarray]:
    """Orders for several markets in one backend call (one batch on the GPU)."""
    convs, seeds, sizes = [], [], []
    for env, tr in pairs:
        c, s = tr.requests(env)
        convs += c
        seeds += s
        sizes.append(len(c))
    cfg = pairs[0][1].cfg
    texts = backend.generate(convs, seeds, cfg.temperature, cfg.max_tokens)
    out, k = [], 0
    for (_, tr), n in zip(pairs, sizes):
        out.append(tr.apply(texts[k:k + n]))
        k += n
    return out


def fixed_lambda_benchmarks(I: int, lam: float, sigma_v: float) -> dict:
    """Stage-game benchmarks when the market maker's price impact is frozen.

    With p = lam * y, trader i earns E[(v - lam (sum x + u)) x_i]. Nash: each
    beta = 1 / ((I+1) lam). Joint optimum: aggregate beta = 1 / (2 lam)."""
    agg_nash = I / ((I + 1) * lam)
    agg_coll = 1.0 / (2.0 * lam)

    def profit(agg):
        return agg / I * sigma_v**2 * (1.0 - lam * agg)

    return {"lam": lam, "beta_nash": agg_nash / I, "beta_coll": agg_coll / I,
            "agg_nash": agg_nash, "agg_coll": agg_coll,
            "profit_nash": profit(agg_nash), "profit_coll": profit(agg_coll)}


def deviation_test(env: KyleMarket, traders: LLMTraders, backend, events: int = 3,
                   gap: int = 5, horizon: int = 8, deviator: int = 0,
                   gamma: float = 0.95, mode: str = "best_response",
                   scale: float = 1.0) -> dict:
    """Paired deviation experiment for in-context traders (cf. impulse_response).

    For each event: play `gap` ordinary periods, clone market and traders, make
    `deviator` deviate in the clone for one period, then let everyone play on in
    both copies for `horizon` periods. Deviation modes:
      best_response  the myopic best response to the rivals' current orders
                     (times `scale`); a cut in trading if traders over-trade
      shift          own order + scale * sigma_u * sign(v): a fixed, visible
                     move toward more aggressive trading
    Values, noise and sampling seeds are shared, so differences come from the
    deviation only. Events at v = 0 are dropped (no deviation is possible).

    d_beta_rival[k] > 0 means rivals trade harder after the deviation."""
    import copy

    I = env.I
    rivals = [i for i in range(I) if i != deviator]
    K = horizon + 1
    var_v = float(np.mean(env.values**2))
    vdx_rival = np.zeros((events, K, env.S))
    vdx_dev = np.zeros((events, K, env.S))
    dpi_dev = np.zeros((events, K, env.S))
    dpi_rival = np.zeros((events, K, env.S))
    mask = np.zeros((events, env.S), dtype=bool)

    for r in range(events):
        for _ in range(gap):
            x = traders.act(env, backend)
            profit, _, info = env.step_orders(x)
            traders.record(info, profit)
        e_dev, t_dev = copy.deepcopy(env), copy.deepcopy(traders)
        for k in range(K):
            x_b, x_d = act_many([(env, traders), (e_dev, t_dev)], backend)
            if k == 0:
                v = e_dev.values[e_dev.v_idx]
                if mode == "shift":
                    x_d[:, deviator] += scale * e_dev.cfg.sigma_u * np.sign(v)
                else:
                    lam = np.maximum(e_dev.lam, 1e-12)
                    rest = x_d[:, rivals].sum(1) + e_dev.passive_beta * v - e_dev.m_y
                    x_d[:, deviator] = scale * (v - e_dev.m_v - lam * rest) / (2.0 * lam)
                mask[r] = np.abs(v) > 1e-9
            r_b, _, i_b = env.step_orders(x_b)
            r_d, _, i_d = e_dev.step_orders(x_d)
            traders.record(i_b, r_b)
            t_dev.record(i_d, r_d)
            v = i_b["v"]
            dx = x_d - x_b
            vdx_dev[r, k] = v * dx[:, deviator]
            vdx_rival[r, k] = v * dx[:, rivals].mean(1)
            dpi_dev[r, k] = r_d[:, deviator] - r_b[:, deviator]
            dpi_rival[r, k] = (r_d[:, rivals] - r_b[:, rivals]).mean(1)

    n = int(mask.sum())

    def profile(a, scale_):
        vals = a.transpose(1, 0, 2)[:, mask] * scale_  # (K, n)
        m = vals.mean(1) if n else np.full(K, np.nan)
        ci = 1.96 * vals.std(1, ddof=1) / np.sqrt(n) if n > 1 else np.full(K, np.nan)
        return m.tolist(), ci.tolist()

    out = {"n_events": n, "horizon": horizon, "mode": mode, "scale": scale}
    for name, arr, sc in (("d_beta_rival", vdx_rival, 1 / var_v), ("d_beta_dev", vdx_dev, 1 / var_v),
                          ("d_profit_dev", dpi_dev, 1.0), ("d_profit_rival", dpi_rival, 1.0)):
        out[name], out[name + "_ci95"] = profile(arr, sc)
    gains = (dpi_dev * (gamma ** np.arange(K))[None, :, None]).sum(1)[mask]
    out["cum_gain_dev"] = float(gains.mean()) if n else float("nan")
    out["cum_gain_dev_ci95"] = float(1.96 * gains.std(ddof=1) / np.sqrt(n)) if n > 1 else float("nan")
    return out


# ------------------------------------------------------------------ backends
class ScriptedBackend:
    """fn(conversation, seed) -> response text."""

    def __init__(self, fn: Callable[[list[dict], int], str]):
        self.fn = fn

    def generate(self, convs, seeds, temperature, max_tokens):
        return [self.fn(c, s) for c, s in zip(convs, seeds)]


@dataclass
class VLLMBackend:
    model: str
    gpu_memory_utilization: float = 0.85
    max_model_len: int = 8192
    tensor_parallel_size: int = 1
    enable_thinking: bool = False
    dtype: str = "auto"  # "half" on pre-Ampere GPUs (no bf16)
    attention_backend: str | None = None  # "triton_attn" on pre-Ampere GPUs
    lora_path: str | None = None  # a trained adapter (see grpo.py)
    lora_rank: int = 16
    _llm: object = field(default=None, repr=False)
    _lora: object = field(default=None, repr=False)

    def __post_init__(self):
        from vllm import LLM

        kw = {"attention_backend": self.attention_backend} if self.attention_backend else {}
        if self.lora_path:
            from vllm.lora.request import LoRARequest

            kw.update(enable_lora=True, max_lora_rank=self.lora_rank, max_loras=1)
            self._lora = LoRARequest("trained", 1, self.lora_path)
        self._llm = LLM(model=self.model, gpu_memory_utilization=self.gpu_memory_utilization,
                        max_model_len=self.max_model_len,
                        tensor_parallel_size=self.tensor_parallel_size,
                        enable_prefix_caching=True, dtype=self.dtype, **kw)

    def generate(self, convs, seeds, temperature, max_tokens):
        from vllm import SamplingParams

        params = [SamplingParams(temperature=temperature, max_tokens=max_tokens, seed=s)
                  for s in seeds]
        outs = self._llm.chat(convs, params, use_tqdm=False, lora_request=self._lora,
                              chat_template_kwargs={"enable_thinking": self.enable_thinking})
        return [o.outputs[0].text for o in outs]


# USD per million tokens: (input, cached input, output). Reasoning tokens bill as output.
OPENAI_PRICES = {
    "gpt-5.4-nano": (0.20, 0.02, 1.25),
    "gpt-5.4-mini": (0.75, 0.075, 4.50),
}


class BudgetExceeded(RuntimeError):
    pass


@dataclass
class OpenAIBackend:
    """Hosted models. Reads OPENAI_API_KEY from the environment.

    Spending is tracked from the token usage the API reports and appended to a
    ledger shared by all runs; a batch is refused if the run's spend would pass
    `run_budget` or the ledger's total would pass `total_budget` (checked
    before each batch, with a worst-case estimate for the batch). Reasoning
    models ignore temperature; seeds are best effort on the API side."""

    model: str
    concurrency: int = 32
    reasoning_effort: str | None = None
    max_completion_tokens: int = 4000
    run_budget: float = 10.0
    total_budget: float = 50.0
    ledger: str = "results/openai_spend.jsonl"
    tag: str = ""
    usage: dict = field(default_factory=lambda: {"input": 0, "cached": 0, "output": 0,
                                                 "calls": 0, "failed": 0, "cost": 0.0})

    def __post_init__(self):
        if self.model not in OPENAI_PRICES:
            raise ValueError(f"no price for {self.model}; add it to OPENAI_PRICES")

    def _cost(self, inp, cached, out):
        pi, pc, po = OPENAI_PRICES[self.model]
        return ((inp - cached) * pi + cached * pc + out * po) / 1e6

    def ledger_total(self) -> float:
        import os

        if not os.path.exists(self.ledger):
            return 0.0
        return sum(json.loads(l)["cost"] for l in open(self.ledger) if l.strip())

    def generate(self, convs, seeds, temperature, max_tokens):
        import asyncio

        # worst case for this batch: every call uses its full completion budget
        n_in = sum(len(json.dumps(c)) for c in convs) / 3.5  # rough token estimate
        worst = self._cost(n_in, 0, len(convs) * self.max_completion_tokens)
        if self.usage["cost"] + worst > self.run_budget:
            raise BudgetExceeded(f"run budget {self.run_budget} would be exceeded")
        if self.ledger_total() + worst > self.total_budget:
            raise BudgetExceeded(f"total budget {self.total_budget} would be exceeded")
        before = dict(self.usage)
        texts = asyncio.run(self._generate(convs, seeds, temperature, max_tokens))
        spent = self.usage["cost"] - before["cost"]
        with open(self.ledger, "a") as fh:
            fh.write(json.dumps({"model": self.model, "tag": self.tag, "calls": len(convs),
                                 "input": self.usage["input"] - before["input"],
                                 "output": self.usage["output"] - before["output"],
                                 "cost": spent}) + "\n")
        return texts

    async def _generate(self, convs, seeds, temperature, max_tokens):
        import asyncio

        from openai import AsyncOpenAI

        client = AsyncOpenAI()
        sem = asyncio.Semaphore(self.concurrency)

        async def one(conv, seed):
            kw = {"model": self.model, "messages": conv, "seed": seed,
                  "max_completion_tokens": self.max_completion_tokens}
            if self.reasoning_effort:
                kw["reasoning_effort"] = self.reasoning_effort
            else:
                kw["temperature"] = temperature
            async with sem:
                for attempt in range(6):
                    try:
                        r = await client.chat.completions.create(**kw)
                        break
                    except Exception:  # rate limits and transient errors
                        if attempt == 5:
                            self.usage["failed"] += 1
                            return ""
                        await asyncio.sleep(2 ** attempt)
            u = r.usage
            cached = getattr(getattr(u, "prompt_tokens_details", None), "cached_tokens", 0) or 0
            self.usage["input"] += u.prompt_tokens
            self.usage["cached"] += cached
            self.usage["output"] += u.completion_tokens
            self.usage["calls"] += 1
            self.usage["cost"] += self._cost(u.prompt_tokens, cached, u.completion_tokens)
            return r.choices[0].message.content or ""

        try:
            return await asyncio.gather(*(one(c, s) for c, s in zip(convs, seeds)))
        finally:
            await client.close()
