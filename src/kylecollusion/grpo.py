"""GRPO training of language-model traders in the repeated Kyle market, or of
language-model pricing agents in the repeated logit-Bertrand game (game="pricing").

Self-play: every informed trader is the same LoRA-adapted policy, prompted
exactly as the in-context traders in `llm_traders` (its own history, the
order flow and prices, its private value) and paid its trading profit.

Each iteration runs `groups` x `group_size` sessions of `periods` periods.
Sessions in a group share every value and noise draw (common random numbers),
so differences in their returns come only from the orders the policy sampled.
A response's advantage is its discounted return-to-go minus the mean over its
group at the same period and trader slot, scaled by the batch's standard
deviation (no per-group scaling, which inflates near-ties). gamma = 0 makes
each order answer for its own period only, the RL analogue of the myopic
placebo; gamma > 0 lets an order be credited with what rivals do afterwards,
which is what collusion sustained by reactions needs.

The loss is the on-policy policy gradient with these advantages (one update
per batch, so GRPO's ratio clipping is inactive), averaged over response
tokens, minus an entropy bonus: without it the policy collapses within a few
updates onto copying the value V from the prompt, after which every sample in
a group is identical and the gradient vanishes. Log-probabilities use the
rollout temperature, so the loss matches the sampling distribution. Rollouts run in vLLM with the latest adapter hot-swapped in; training
uses the token ids vLLM generated, so nothing is re-tokenised.
"""

from __future__ import annotations

import json
import math
import os
import re
import shutil
import time
from dataclasses import asdict, dataclass

import numpy as np

from .llm_traders import CHAT_DATE, LLMTraderConfig, LLMTraders, fixed_lambda_benchmarks
from .market import KyleMarket, MarketConfig


@dataclass
class GRPOConfig:
    model: str = "Qwen/Qwen3-0.6B"
    n_informed: int = 2
    sigma_u: float = 1.0
    groups: int = 8  # independent market draws per iteration
    group_size: int = 8  # sessions per group sharing values and noise
    periods: int = 16  # periods per training episode
    gamma: float = 0.9  # return-to-go discount; 0 = myopic credit
    history: int = 8
    notes: bool = False
    temperature: float = 1.0
    max_tokens: int = 64
    lr: float = 1e-5
    entropy_coef: float = 0.01  # bonus on response-token entropy (keeps exploring)
    lora_rank: int = 16
    lora_alpha: int = 32
    micro_batch: int = 32
    max_grad_norm: float = 1.0
    iterations: int = 200
    save_every: int = 25
    vllm_mem: float = 0.35
    train_dtype: str = "bfloat16"  # "float32" on GPUs without bf16
    vllm_dtype: str = "auto"
    attention_backend: str | None = None
    seed: int = 0
    # rival: "self" (self-play, both traders are the policy) or "trigger": the
    # policy plays trader 0 against a scripted rival who orders coop x V, and
    # punish x V for punish_len periods after the policy's order exceeds
    # coop x |V| + tol (perfect monitoring). coop/punish default to the
    # frozen-lambda joint optimum per trader and twice the Nash intensity.
    game: str = "kyle"  # "kyle" (traders) or "pricing" (logit-Bertrand duopoly, our pricing prompt)
    # pricing: rival "self" (self-play) or "tft", a scripted rival that posts the policy's price of the
    # previous period (tit-for-tat, starting at the monopoly price), so every cut is matched a period later,
    # or "cheater", a scripted rival that cheats on the monopoly price unless a price war deters it
    # (CheaterRival; cheat_prob, cheat_cut_min/max and deter_len set it)
    price_scale: float = 1.0  # pricing: currency unit
    rival: str = "self"
    monitor: bool = False  # show the rival's past orders in the prompt (self-play)
    coop: float = 0.0
    punish: float = 0.0
    punish_len: int = 3
    tol: float = 0.3
    cheat_prob: float = 0.5
    cheat_cut_min: float = 0.10
    cheat_cut_max: float = 0.25
    deter_len: int = 10
    war_margin: float = 0.22  # a deterring price war lies this share of (p^M - p^N) below Nash and the cheat
    explore_war: float = 0.0  # pricing vs the cheater: share of answers to a cheat replaced by a war (exploration check)
    explore_credit: bool = True  # credit the forced wars as if sampled (False: leave them out of the update)
    explore_placebo: float = 0.0  # control: share of answers to no cheat replaced by the same war price
    format_penalty: bool = False  # pricing: an answer that does not parse earns nothing that period (training reward only)
    init_adapter: str = ""  # start from another run's adapter (e.g. continue a trained policy in self-play)


class CheaterRival:
    """Scripted pricing rival that cheats on the monopoly price unless the policy deters it.

    It posts the monopoly price p^M. From the third period on, unless deterred, it cheats with
    probability `prob`, cutting its price by a share drawn from `cut` (draws shared by the sessions
    of a group). The policy deters it for `deter_len` periods by answering a cheat, in the next
    period, with a price war: a price at least `margin` x (p^M - p^N) below both the Nash price and
    the cheat. That is below the policy's best response to the cheat, so the war costs profit in the
    period it is fought and pays only through the cheats it deters: a forward-looking policy gains by
    learning it, a myopic one does not. An unprovoked undercut by the policy (a price below
    p^M - tol x (p^M - p^N) that does not answer a cheat) is answered by one period at the Nash price,
    so the policy's best standing price is the monopoly price."""

    def __init__(self, bench: dict, n_sessions: int, group_size: int, periods: int,
                 rng: np.random.Generator, prob: float = 0.5, cut: tuple = (0.10, 0.25),
                 deter_len: int = 10, margin: float = 0.22, tol: float = 0.1):
        self.pM, self.pN = bench["p_mono"], bench["p_nash"]
        span = self.pM - self.pN
        self.margin, self.tol = margin * span, tol * span
        G = n_sessions // group_size
        self.u = np.repeat(rng.random((G, periods)), group_size, axis=0)
        self.d = np.repeat(rng.uniform(cut[0], cut[1], (G, periods)), group_size, axis=0)
        self.prob, self.deter_len = prob, deter_len
        self.deterred_until = np.full(n_sessions, -1)
        self.cheat_prev = np.zeros(n_sessions, bool)  # the rival cheated last period
        self.cheat_prev2 = np.zeros(n_sessions, bool)  # ... two periods ago
        self.own_prev = None  # the policy's last price
        self.r_prev = np.full(n_sessions, self.pM)  # the rival's last price
        self._cheat_now = np.zeros(n_sessions, bool)
        self.n_cheats = self.n_answers = self.n_wars = self.n_unprovoked = 0
        self.answer_prices: list[np.ndarray] = []  # the policy's prices in periods that answer a cheat

    def price(self, t: int) -> np.ndarray:
        S = self.u.shape[0]
        unprov = (np.zeros(S, bool) if self.own_prev is None
                  else (self.own_prev < self.pM - self.tol) & ~self.cheat_prev2)
        r = np.where(unprov, self.pN, self.pM)
        cheat = ~unprov & (t >= 2) & (t > self.deterred_until) & (self.u[:, t] < self.prob)
        self._cheat_now = cheat
        self.n_unprovoked += int(unprov.sum())
        return np.where(cheat, (1.0 - self.d[:, t]) * self.pM, r)

    def observe(self, t: int, p_policy: np.ndarray, r: np.ndarray) -> None:
        """Record period t: the policy's price (which may answer last period's cheat) and the rival's."""
        war = self.cheat_prev & (p_policy <= np.minimum(self.pN, self.r_prev) - self.margin)
        self.deterred_until = np.where(war, t + self.deter_len, self.deterred_until)
        if self.cheat_prev.any():
            self.answer_prices.append(p_policy[self.cheat_prev])
        self.n_answers += int(self.cheat_prev.sum())
        self.n_wars += int(war.sum())
        self.n_cheats += int(self._cheat_now.sum())
        self.cheat_prev2, self.cheat_prev = self.cheat_prev, self._cheat_now
        self.own_prev, self.r_prev = np.array(p_policy, float), np.array(r, float)


class GroupRNG:
    """numpy Generator stand-in that draws once per group and repeats the draw
    for each of the group's consecutive sessions (common random numbers)."""

    def __init__(self, rng: np.random.Generator, group_size: int):
        self.rng, self.G = rng, group_size

    def _n(self, size):
        if not isinstance(size, (int, np.integer)) or size % self.G:
            raise ValueError("GroupRNG supports 1-d draws over all sessions only")
        return size // self.G

    def normal(self, loc=0.0, scale=1.0, size=None):
        return np.repeat(self.rng.normal(loc, scale, self._n(size)), self.G)

    def integers(self, high, size=None):
        return np.repeat(self.rng.integers(high, size=self._n(size)), self.G)


def group_advantages(profit: np.ndarray, gamma: float, group_size: int) -> np.ndarray:
    """profit (T, S, I) -> advantages (T, S, I): discounted return-to-go minus
    its group mean, scaled by the batch sd of the centred returns."""
    T, S, I = profit.shape
    ret = np.zeros_like(profit)
    acc = np.zeros((S, I))
    for t in range(T - 1, -1, -1):
        acc = profit[t] + gamma * acc
        ret[t] = acc
    g = ret.reshape(T, S // group_size, group_size, I)
    centred = (g - g.mean(2, keepdims=True)).reshape(T, S, I)
    sd = centred.std()
    return centred / (sd + 1e-8)


class PolicyBackend:
    """vLLM generation with the current LoRA adapter; records token ids."""

    def __init__(self, llm):
        self.llm = llm
        self.lora = None
        self.records: list | None = None

    def generate(self, convs, seeds, temperature, max_tokens):
        from vllm import SamplingParams

        params = [SamplingParams(temperature=temperature, max_tokens=max_tokens, seed=s)
                  for s in seeds]
        outs = self.llm.chat(convs, params, use_tqdm=False, lora_request=self.lora,
                             chat_template_kwargs={"enable_thinking": False, "date_string": CHAT_DATE})
        if self.records is not None:
            self.records.append([(list(o.prompt_token_ids), list(o.outputs[0].token_ids))
                                 for o in outs])
        return [o.outputs[0].text for o in outs]


def rollout(backend: PolicyBackend, cfg: GRPOConfig, seed: int):
    """Play one batch of episodes. Returns training samples and statistics."""
    S = cfg.groups * cfg.group_size
    mcfg = MarketConfig(n_informed=cfg.n_informed, mm_fixed=True, memory="flow",
                        sigma_u=cfg.sigma_u)
    env = KyleMarket(mcfg, S, seed=seed)
    env.rng = GroupRNG(env.rng, cfg.group_size)
    env.reset()
    trigger = cfg.rival == "trigger"
    if trigger and env.I != 2:
        raise ValueError("the trigger rival needs n_informed = 2")
    fb = fixed_lambda_benchmarks(env.I, env.bench.lam_nash, mcfg.sigma_v)
    tcfg = LLMTraderConfig(history=cfg.history, notes=cfg.notes,
                           show_rival=trigger or cfg.monitor,
                           temperature=cfg.temperature, max_tokens=cfg.max_tokens)
    active = [0] if trigger else None
    traders = LLMTraders(env, tcfg, seed=seed, active=active)
    coop = cfg.coop or fb["beta_coll"]
    punish = cfg.punish or 2.0 * fb["beta_nash"]
    T, I = cfg.periods, env.I
    profit = np.zeros((T, S, I))
    x_all = np.zeros((T, S, I))
    v_all = np.zeros((T, S))
    punishing = np.zeros((T, S), dtype=bool)
    left = np.zeros(S, dtype=np.int64)
    backend.records = []
    for t in range(T):
        x = traders.act(env, backend)
        if trigger:
            v = env.values[env.v_idx]
            punishing[t] = left > 0
            x[:, 1] = np.where(punishing[t], punish, coop) * v
        pi, _, info = env.step_orders(x)
        traders.record(info, pi)
        profit[t], x_all[t], v_all[t] = pi, x, info["v"]
        if trigger:
            v = info["v"]
            broke = (np.abs(x[:, 0]) > coop * np.abs(v) + cfg.tol) & (np.abs(v) > 0)
            left = np.where(broke, cfg.punish_len, np.maximum(left - 1, 0))
    records = backend.records
    backend.records = None

    adv = group_advantages(profit, cfg.gamma, cfg.group_size)
    act = traders.active
    samples = []
    for t in range(T):
        for k, (p_ids, o_ids) in enumerate(records[t]):
            s, j = divmod(k, len(act))
            if o_ids:
                samples.append((p_ids, o_ids, float(adv[t, s, act[j]])))

    agg = x_all.sum(-1)
    beta = float((agg * v_all).sum() / (v_all**2).sum())
    stats = {
        "agg_intensity": beta,
        "intensity_over_nash": beta / fb["agg_nash"],
        "delta_intensity": (beta - fb["agg_nash"]) / (fb["agg_coll"] - fb["agg_nash"])
        if I > 1 else float("nan"),
        "profit": float(profit.mean()),
        "profit_over_nash": float(profit.mean() / fb["profit_nash"]),
        "zero_adv_share": float(np.mean([a == 0.0 for _, _, a in samples])) if samples else 1.0,
        "parse_fail": traders.n_fail / max(traders.n_calls, 1),
        "resp_tokens": float(np.mean([len(o) for _, o, _ in samples])) if samples else 0.0,
    }
    if trigger:  # the policy's own intensity, profit, and how often it is punished
        b0 = float((x_all[:, :, 0] * v_all).sum() / (v_all**2).sum())
        stats.update(policy_intensity=b0, coop=coop, policy_br_to_coop=(1 - fb["lam"] * coop)
                     / (2 * fb["lam"]), punished_share=float(punishing.mean()),
                     policy_profit=float(profit[:, :, 0].mean()))
    return samples, stats


def rollout_pricing(backend: PolicyBackend, cfg: GRPOConfig, seed: int):
    """The logit-Bertrand duopoly: in self-play both firms are the policy, prompted as the
    in-context pricing agents of `llm_pricing` and paid their profit; with rival "tft" the
    policy is firm 0 and firm 1 posts the policy's previous price. Demand is deterministic,
    so the sessions of a group differ only through the prices the policy sampled."""
    from .llm_pricing import LLMPricers, PricingConfig, PricingMarket, act_many

    if cfg.rival not in ("self", "tft", "cheater"):
        raise ValueError(f"pricing rival must be 'self', 'tft' or 'cheater', not {cfg.rival!r}")
    tft, cheater = cfg.rival == "tft", cfg.rival == "cheater"
    scripted = tft or cheater
    explore = cfg.explore_war > 0 or cfg.explore_placebo > 0
    if explore and not cheater:
        raise ValueError("war exploration is for training against the cheating rival")
    S = cfg.groups * cfg.group_size
    pcfg = PricingConfig(scale=cfg.price_scale, history=cfg.history, notes=cfg.notes,
                         temperature=cfg.temperature, max_tokens=cfg.max_tokens)
    env = PricingMarket(pcfg, S)
    pricers = LLMPricers(env, pcfg, seed=seed, active=[0] if scripted else None)
    T = cfg.periods
    profit = np.zeros((T, S, 2))
    prices = np.zeros((T, S, 2))
    failed = np.zeros((T, S, 2), bool)
    prev = np.full(S, env.bench["p_mono"])  # the tit-for-tat rival opens at the monopoly price
    if cheater:
        rival = CheaterRival(env.bench, S, cfg.group_size, T, np.random.default_rng(seed + 104729),
                             prob=cfg.cheat_prob, cut=(cfg.cheat_cut_min, cfg.cheat_cut_max),
                             deter_len=cfg.deter_len, margin=cfg.war_margin)
    backend.records = []
    xrng = np.random.default_rng(seed + 7919)  # which answers to a cheat are replaced by a war
    own_ans, own_other, n_inj = [], [], 0
    forced = np.zeros((T, S), bool)  # the policy's answer was replaced by a war
    for t in range(T):
        p = act_many([(env, pricers)], backend)[0]
        failed[t] = pricers.failed
        if tft:
            p[:, 1] = prev
            prev = np.clip(p[:, 0], 0.0, 10.0 * env.bcfg.cost)
        elif cheater:
            if explore:  # exploration check: force a war in a share of the answers to a cheat
                thr = np.minimum(rival.pN, rival.r_prev) - rival.margin
                u = xrng.random(S)
                inj = rival.cheat_prev & (u < cfg.explore_war)
                inj |= ~rival.cheat_prev & (u < cfg.explore_placebo)  # control: the war price, answering no cheat
                own = rival.cheat_prev & ~inj
                own_ans.append(np.stack([np.clip(p[own, 0], 0.0, 10.0 * env.bcfg.cost), thr[own]], 1))
                other = ~rival.cheat_prev & ~inj
                own_other.append(np.stack([np.clip(p[other, 0], 0.0, 10.0 * env.bcfg.cost), thr[other]], 1))
                for s in np.flatnonzero(inj):
                    p[s, 0] = math.floor(thr[s] * 100) / 100
                    _force_answer(backend, s, p[s, 0])
                    failed[t, s, 0] = False
                forced[t] = inj
                n_inj += int(inj.sum())
            p[:, 1] = rival.price(t)
        q, pi = env.step(p)
        if cheater:
            rival.observe(t, np.clip(p[:, 0], 0.0, 10.0 * env.bcfg.cost), p[:, 1])
        pricers.record(np.clip(p, 0.0, 10.0 * env.bcfg.cost), q, pi)
        profit[t], prices[t] = pi, p
    records = backend.records
    backend.records = None
    act = pricers.active
    reward = profit[:, :, act]
    if cfg.format_penalty:  # the market keeps the last price; the answer itself earns nothing
        reward = np.where(failed[:, :, act], 0.0, reward)
    adv = group_advantages(reward, cfg.gamma, cfg.group_size)  # policy firms only
    samples = []
    for t in range(T):
        for k, (p_ids, o_ids) in enumerate(records[t]):
            s, j = divmod(k, len(act))
            if o_ids and (cfg.explore_credit or not forced[t, s]):  # uncredited forced wars stay out
                samples.append((p_ids, o_ids, float(adv[t, s, j])))
    b = env.bench
    idx = (np.clip(prices, 0.0, 10.0 * env.bcfg.cost).mean(-1) - b["p_nash"]) / (b["p_mono"] - b["p_nash"])
    stats = {
        "price_index": float(idx.mean()),
        "price_index_second_half": float(idx[T // 2:].mean()),
        "price_sd_across_sessions": float(prices[T // 2:].mean((0, 2)).std()),
        "profit_over_nash": float(profit.mean() / b["pi_nash"]),
        "zero_adv_share": float(np.mean([a == 0.0 for _, _, a in samples])) if samples else 1.0,
        "parse_fail": pricers.n_fail / max(pricers.n_calls, 1),
        "resp_tokens": float(np.mean([len(o) for _, o, _ in samples])) if samples else 0.0,
    }
    if scripted:  # the policy's own price index and profit
        stats.update(policy_price_index_second_half=float(
            ((np.clip(prices[T // 2:, :, 0], 0.0, 10.0 * env.bcfg.cost) - b["p_nash"])
             / (b["p_mono"] - b["p_nash"])).mean()),
            policy_profit_over_nash=float(profit[:, :, 0].mean() / b["pi_nash"]))
    if cheater:  # how often the rival cheats and how the policy answers
        ans = np.concatenate(rival.answer_prices) if rival.answer_prices else np.array([])
        stats.update(cheats_per_session=rival.n_cheats / S, unprovoked_per_session=rival.n_unprovoked / S,
                     war_share=rival.n_wars / max(rival.n_answers, 1),
                     answer_price_index=float(((ans - b["p_nash"]) / (b["p_mono"] - b["p_nash"])).mean())
                     if ans.size else float("nan"))
        if explore:  # the policy's own answers, without the forced wars
            own = np.concatenate(own_ans) if own_ans else np.zeros((0, 2))
            other = np.concatenate(own_other) if own_other else np.zeros((0, 2))
            stats.update(forced_wars_per_session=n_inj / S,
                         own_war_share=float((own[:, 0] <= own[:, 1]).mean()) if len(own) else float("nan"),
                         own_answer_price_index=float(((own[:, 0] - b["p_nash"]) / (b["p_mono"] - b["p_nash"])).mean())
                         if len(own) else float("nan"),
                         own_other_war_share=float((other[:, 0] <= other[:, 1]).mean()) if len(other) else float("nan"))
    return samples, stats


def _force_answer(backend, s: int, price: float) -> None:
    """Replace session s's answer in the latest recorded batch by the same answer with `price`, so the
    update credits the forced war as if the policy had sampled it."""
    tok = backend.llm.get_tokenizer()
    p_ids, o_ids = backend.records[-1][s]
    eos = [o_ids[-1]] if o_ids and o_ids[-1] in tok.all_special_ids else []
    text = tok.decode(o_ids, skip_special_tokens=True)
    new, n = re.subn(r'("price"\s*:\s*)-?[0-9]*\.?[0-9]+', lambda m: m.group(1) + f"{price:.2f}", text, count=1)
    if not n:
        new = '{"price": ' + f"{price:.2f}" + "}"
    backend.records[-1][s] = (p_ids, tok.encode(new, add_special_tokens=False) + eos)


def policy_gradient_step(model, opt, samples, cfg: GRPOConfig, device) -> tuple[float, float]:
    """One on-policy update: maximise [sum(adv * log pi) + c * entropy] / n_tokens.
    Returns (loss, mean response-token entropy)."""
    import torch

    model.train()
    n_tok = sum(len(o) for _, o, _ in samples)
    order = sorted(range(len(samples)), key=lambda j: len(samples[j][0]) + len(samples[j][1]))
    total, ent_sum = 0.0, 0.0
    for b in range(0, len(order), cfg.micro_batch):
        batch = [samples[j] for j in order[b:b + cfg.micro_batch]]
        L = max(len(p) + len(o) for p, o, _ in batch)
        R = max(len(o) for _, o, _ in batch)
        ids = torch.zeros((len(batch), L), dtype=torch.long)
        mask = torch.zeros((len(batch), L), dtype=torch.long)
        rmask = torch.zeros((len(batch), R), dtype=torch.float32)
        targets = torch.zeros((len(batch), R), dtype=torch.long)
        adv = torch.tensor([a for _, _, a in batch], dtype=torch.float32)
        for r, (p, o, _) in enumerate(batch):
            seq = p + o
            ids[r, L - len(seq):] = torch.tensor(seq)  # left padding: responses end aligned
            mask[r, L - len(seq):] = 1
            targets[r, R - len(o):] = torch.tensor(o)
            rmask[r, R - len(o):] = 1.0
        pos = (mask.cumsum(1) - 1).clamp(min=0)
        ids, mask, pos = ids.to(device), mask.to(device), pos.to(device)
        # logits at positions L-R-1 .. L-2 predict the last R tokens
        out = model(input_ids=ids, attention_mask=mask, position_ids=pos, logits_to_keep=R + 1)
        logits = out.logits[:, :-1].float() / max(cfg.temperature, 1e-6)
        logp_all = torch.log_softmax(logits, -1)
        logp = logp_all.gather(-1, targets.to(device)[..., None])[..., 0]
        rm = rmask.to(device)
        ent = -(logp_all.exp() * logp_all).sum(-1)
        loss = -((adv.to(device)[:, None] * logp + cfg.entropy_coef * ent) * rm).sum() / n_tok
        loss.backward()
        total += float(loss)
        ent_sum += float((ent * rm).sum())
    torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],
                                   cfg.max_grad_norm)
    opt.step()
    opt.zero_grad(set_to_none=True)
    return total, ent_sum / max(n_tok, 1)


def train(cfg: GRPOConfig, out_dir: str, train_device: str = "cuda:0") -> None:
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM
    from vllm import LLM
    from vllm.lora.request import LoRARequest

    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "config.json"), "w") as fh:
        json.dump(asdict(cfg), fh, indent=1)
    kw = {"attention_backend": cfg.attention_backend} if cfg.attention_backend else {}
    llm = LLM(model=cfg.model, gpu_memory_utilization=cfg.vllm_mem, max_model_len=4096,
              enable_lora=True, max_lora_rank=cfg.lora_rank, max_loras=1,
              enable_prefix_caching=True, dtype=cfg.vllm_dtype, seed=cfg.seed, **kw)
    backend = PolicyBackend(llm)

    torch.manual_seed(cfg.seed)
    base = AutoModelForCausalLM.from_pretrained(cfg.model, dtype=getattr(torch, cfg.train_dtype))
    base.gradient_checkpointing_enable()
    base.enable_input_require_grads()
    lcfg = LoraConfig(r=cfg.lora_rank, lora_alpha=cfg.lora_alpha, lora_dropout=0.0,
                      target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                      "gate_proj", "up_proj", "down_proj"])
    model = get_peft_model(base, lcfg).to(train_device)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=cfg.lr,
                            weight_decay=0.0)

    log_path = os.path.join(out_dir, "log.jsonl")
    start = 0
    if os.path.exists(log_path):  # resume from the last saved adapter
        done = [json.loads(l) for l in open(log_path)]
        saved = [d["iteration"] for d in done if d.get("saved")]
        if saved:
            from peft import set_peft_model_state_dict
            from safetensors.torch import load_file

            last = saved[-1]
            path = os.path.join(out_dir, f"adapter_{last:04d}")
            set_peft_model_state_dict(model, load_file(os.path.join(path, "adapter_model.safetensors")))
            backend.lora = LoRARequest(f"it{last}", last + 1, path)
            start = last + 1
            with open(log_path, "w") as fh:  # drop iterations after the checkpoint
                for d in done:
                    if d["iteration"] <= last:
                        fh.write(json.dumps(d) + "\n")

    if start == 0 and cfg.init_adapter:
        from peft import set_peft_model_state_dict
        from safetensors.torch import load_file

        set_peft_model_state_dict(model, load_file(os.path.join(cfg.init_adapter, "adapter_model.safetensors")))
        backend.lora = LoRARequest("init", 1, cfg.init_adapter)

    tmp = os.path.join(out_dir, "adapter_latest")
    for it in range(start, cfg.iterations):
        t0 = time.time()
        play = rollout_pricing if cfg.game == "pricing" else rollout
        samples, stats = play(backend, cfg, seed=cfg.seed * 100003 + it)
        t1 = time.time()
        loss, entropy = policy_gradient_step(model, opt, samples, cfg, train_device)
        t2 = time.time()
        saved = (it + 1) % cfg.save_every == 0 or it + 1 == cfg.iterations
        path = os.path.join(out_dir, f"adapter_{it:04d}") if saved else tmp
        if os.path.exists(path):
            shutil.rmtree(path)
        model.save_pretrained(path)
        # a fresh id makes vLLM load the new weights
        backend.lora = LoRARequest(f"it{it}", it + 2, path)
        rec = {"iteration": it, **stats, "loss": loss, "entropy": entropy,
               "n_samples": len(samples),
               "rollout_s": round(t1 - t0, 1), "train_s": round(t2 - t1, 1), "saved": saved}
        with open(log_path, "a") as fh:
            fh.write(json.dumps(rec) + "\n")
        print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in rec.items()}),
              flush=True)
