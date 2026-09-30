"""Run a grid of experiments in parallel worker processes.

    PYTHONPATH=src python experiments/sweep.py experiments/grids/interventions.json --workers 4

A grid file is JSON: {"name": ..., "base": [cli args...], "grid": {"--flag": [values...]}}.
Every combination of grid values is run once; results land in results/<name>/.
Runs whose output file already exists are skipped, and every run checkpoints
to checkpoints/<name>/<tag>.pkl, so rerunning an interrupted sweep resumes each
unfinished run where it stopped.

Grid values are passed as single CLI arguments, so JSON for --agent-kwargs can
be given as a plain string value.
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
        labels = spec.get("labels", {})
        tag = "_".join(
            f"{k.lstrip('-').replace('-', '')}{labels.get(k, {}).get(str(v), v)}"
            for k, v in zip(keys, combo)
        )
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
    ckdir = os.path.join("checkpoints", spec["name"])
    os.makedirs(outdir, exist_ok=True)
    os.makedirs(ckdir, exist_ok=True)

    todo = []
    for tag, args in expand(spec):
        out = os.path.join(outdir, f"{tag}.json")
        if os.path.exists(out):
            continue
        ck = os.path.join(ckdir, f"{tag}.pkl")
        cmd = [sys.executable, "-m", "kylecollusion.run", *args, "--out", out, "--checkpoint", ck]
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
        running.append(
            subprocess.Popen(cmd, stdout=open(log, "a"), stderr=subprocess.STDOUT, env=env,
                             start_new_session=True)
        )
        print("started", os.path.basename(log), flush=True)
    for p in running:
        p.wait()
    # Checkpoints are only needed to resume; drop them once every run is done.
    for tag, _ in expand(spec):
        if os.path.exists(os.path.join(outdir, f"{tag}.json")):
            ck = os.path.join(ckdir, f"{tag}.pkl")
            if os.path.exists(ck):
                os.remove(ck)
    print("done")


if __name__ == "__main__":
    main()
