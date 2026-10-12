"""How often untrained LLMs try the price war against the cheating rival.

Each model plays the first design of the cheating rival exactly as in GRPO training (the same prompt, 16-period
episodes, groups of eight sessions, temperature 1), without any training, for a number of rollouts. We record every
answer to a cheat with the cheat's price and the war threshold, and report the share of answers that are wars, that
match the cheat (within a cent), that undercut it without reaching a war, and that price above it.

    python experiments/war_try_rates.py --model Qwen/Qwen3-8B --rollouts 40 --out results/war_try/qwen3p8b.json
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np

from kylecollusion import grpo
from kylecollusion.grpo import GRPOConfig, PolicyBackend, rollout_pricing


class RecordingCheater(grpo.CheaterRival):
    """The cheating rival of training, recording each answer to a cheat with the cheat's price and the war
    threshold (a war is a price at or below it)."""

    log: list = []

    def observe(self, t, p_policy, r):
        if self.cheat_prev.any():
            m = self.cheat_prev
            thr = np.minimum(self.pN, self.r_prev) - self.margin
            RecordingCheater.log.append(np.stack([np.asarray(p_policy, float)[m], self.r_prev[m], thr[m]], 1))
        super().observe(t, p_policy, r)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--rollouts", type=int, default=40, help="rollouts of 32 sessions x 16 periods")
    ap.add_argument("--gpu-mem", type=float, default=0.85)
    ap.add_argument("--dtype", default="auto")
    ap.add_argument("--quantization", default=None)
    ap.add_argument("--attention-backend", default=None)
    ap.add_argument("--max-num-seqs", type=int, default=64,
                    help="a rollout runs 32 sessions at once; fewer sequences need less memory at start-up")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)

    from vllm import LLM

    kw = {"attention_backend": a.attention_backend} if a.attention_backend else {}
    if a.max_num_seqs:
        kw["max_num_seqs"] = a.max_num_seqs
    if a.quantization:
        kw["quantization"] = a.quantization
    llm = LLM(model=a.model, gpu_memory_utilization=a.gpu_mem, max_model_len=4096, enable_prefix_caching=True,
              dtype=a.dtype, seed=0, **kw)
    grpo.CheaterRival = RecordingCheater  # rollout_pricing builds the rival by this name
    cfg = GRPOConfig(model=a.model, game="pricing", rival="cheater", groups=4, group_size=8, periods=16)
    backend = PolicyBackend(llm)
    t0, stats = time.time(), []
    for k in range(a.rollouts):
        _, st = rollout_pricing(backend, cfg, seed=1000 + k)
        backend.records = None
        stats.append({key: st[key] for key in ("cheats_per_session", "war_share", "answer_price_index", "parse_fail",
                                               "unprovoked_per_session") if key in st})
    A = np.concatenate(RecordingCheater.log) if RecordingCheater.log else np.zeros((0, 3))
    ans, cheat, thr = A[:, 0], A[:, 1], A[:, 2]
    war = ans <= thr + 1e-9
    match = np.abs(ans - cheat) <= 0.01 + 1e-9
    below = (ans < cheat - 0.01 - 1e-9) & ~war
    above = ans > cheat + 0.01 + 1e-9
    n = len(ans)
    out = {"model": a.model, "rollouts": a.rollouts, "answers": int(n), "seconds": round(time.time() - t0, 1),
           "war_share": float(war.mean()) if n else None, "wars": int(war.sum()),
           "match_share": float(match.mean()) if n else None, "below_share": float(below.mean()) if n else None,
           "above_share": float(above.mean()) if n else None,
           "parse_fail": float(np.mean([s_["parse_fail"] for s_ in stats])),
           "cheats_per_session": float(np.mean([s_["cheats_per_session"] for s_ in stats])),
           "unprovoked_per_session": float(np.mean([s_.get("unprovoked_per_session", np.nan) for s_ in stats])),
           "per_rollout": stats, "args": vars(a)}
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(out, open(a.out, "w"), indent=1)
    print(json.dumps({k: out[k] for k in ("model", "answers", "war_share", "wars", "match_share", "below_share",
                                         "above_share", "parse_fail", "cheats_per_session", "seconds")}))


if __name__ == "__main__":
    main()
