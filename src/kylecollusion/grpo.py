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
import shutil
import time
from dataclasses import asdict, dataclass

import numpy as np

from .llm_traders import LLMTraderConfig, LLMTraders, fixed_lambda_benchmarks
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
    price_scale: float = 1.0  # pricing: currency unit
    rival: str = "self"
    monitor: bool = False  # show the rival's past orders in the prompt (self-play)
    coop: float = 0.0
    punish: float = 0.0
    punish_len: int = 3
    tol: float = 0.3


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
                             chat_template_kwargs={"enable_thinking": False})
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
    """Self-play in the logit-Bertrand duopoly: both firms are the policy, prompted as the
    in-context pricing agents of `llm_pricing` and paid their profit. Demand is deterministic,
    so the sessions of a group differ only through the prices the policy sampled."""
    from .llm_pricing import LLMPricers, PricingConfig, PricingMarket, act_many

    S = cfg.groups * cfg.group_size
    pcfg = PricingConfig(scale=cfg.price_scale, history=cfg.history, notes=cfg.notes,
                         temperature=cfg.temperature, max_tokens=cfg.max_tokens)
    env = PricingMarket(pcfg, S)
    pricers = LLMPricers(env, pcfg, seed=seed)
    T = cfg.periods
    profit = np.zeros((T, S, 2))
    prices = np.zeros((T, S, 2))
    backend.records = []
    for t in range(T):
        p = act_many([(env, pricers)], backend)[0]
        q, pi = env.step(p)
        pricers.record(np.clip(p, 0.0, 10.0 * env.bcfg.cost), q, pi)
        profit[t], prices[t] = pi, p
    records = backend.records
    backend.records = None
    adv = group_advantages(profit, cfg.gamma, cfg.group_size)
    samples = []
    for t in range(T):
        for k, (p_ids, o_ids) in enumerate(records[t]):
            s, i = divmod(k, 2)
            if o_ids:
                samples.append((p_ids, o_ids, float(adv[t, s, i])))
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
    return samples, stats


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
