"""Run a grid of experiments in parallel worker processes.

    PYTHONPATH=src python experiments/sweep.py experiments/grids/interventions.json --workers 4

A grid file is JSON: {"name": ..., "base": [cli args...], "grid": {"--flag": [values...]}}.
Every combination of grid values is run once; results land in results/<name>/.
Runs whose output file already exists are skipped, so an interrupted sweep
resumes where it stopped.
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import subprocess
import sys


def expand(spec: dict) -> list[tuple[str, list[str]]]:
    keys = list(spec["grid"])
    runs = []
    for combo in itertools.product(*(spec["grid"][k] for k in keys)):
        tag = "_".join(f"{k.lstrip('-').replace('-', '')}{v}" for k, v in zip(keys, combo))
        args = list(spec.get("base", []))
        for k, v in zip(keys, combo):
            args += [k, str(v)] if v is not True else [k]
        runs.append((tag, args))
    return runs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("grid")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    spec = json.load(open(a.grid))
    outdir = os.path.join("results", spec["name"])
    os.makedirs(outdir, exist_ok=True)

    todo = []
    for tag, args in expand(spec):
        out = os.path.join(outdir, f"{tag}.json")
        if os.path.exists(out):
            continue
        cmd = [sys.executable, "-m", "kylecollusion.run", *args, "--out", out]
        todo.append((cmd, os.path.join(outdir, f"{tag}.log")))

    print(f"{len(todo)} runs to do in {outdir}")
    if a.dry_run:
        for cmd, _ in todo:
            print(" ".join(cmd))
        return

    env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    running: list[subprocess.Popen] = []
    for cmd, log in todo:
        while len(running) >= a.workers:
            running[0].wait()
            running = [p for p in running if p.poll() is None]
        running.append(subprocess.Popen(cmd, stdout=open(log, "w"), stderr=subprocess.STDOUT, env=env))
        print("started", os.path.basename(log), flush=True)
    for p in running:
        p.wait()
    print("done")


if __name__ == "__main__":
    main()
