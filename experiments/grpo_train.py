"""Train LLM traders with GRPO self-play in the Kyle market (see kylecollusion.grpo).

Example (vLLM and training on one A100):
  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=2 PYTHONPATH=src \\
    python experiments/grpo_train.py --model Qwen/Qwen3-1.7B --gamma 0.9 \\
    --out results/grpo/qwen3_1p7b_g09
Resumes from the last saved adapter in --out.
"""

from __future__ import annotations

import argparse
from dataclasses import fields

from kylecollusion.grpo import GRPOConfig, train


def main(argv=None):
    ap = argparse.ArgumentParser()
    for f in fields(GRPOConfig):
        flag = "--" + f.name.replace("_", "-")
        if f.type in ("bool", bool):
            ap.add_argument(flag, type=lambda s: s.lower() in ("1", "true", "yes"), default=f.default)
        elif f.name == "attention_backend":
            ap.add_argument(flag, default=f.default)
        else:
            typ = {"int": int, "float": float, "str": str}.get(str(f.type), type(f.default))
            ap.add_argument(flag, type=typ, default=f.default)
    ap.add_argument("--out", required=True)
    ap.add_argument("--train-device", default="cuda:0")
    a = ap.parse_args(argv)
    cfg = GRPOConfig(**{f.name: getattr(a, f.name) for f in fields(GRPOConfig)})
    train(cfg, a.out, a.train_device)


if __name__ == "__main__":
    main()
