"""Episode runner: Task + Agent + Grader -> TaskResult, with trajectory capture.

Sequential and dependency-free by design; parallelism/retries belong in a thin
wrapper so this stays trivially testable.
"""

from __future__ import annotations

import json
import os
import time
import traceback

from .agent import Agent, Grader
from .task import Task, TaskResult


def run_task(
    task: Task,
    agent: Agent,
    grader: Grader,
    model_label: str,
    trajectory_dir: str | None = None,
) -> TaskResult:
    submission = None
    detail: dict = {}
    started = time.time()
    try:
        submission = agent.solve(task)
        solved, score, detail = grader.grade(task, submission)
    except NotImplementedError:
        raise
    except Exception as exc:  # a crashed episode is a data point, not a stop
        solved, score = False, 0.0
        detail = {"error": f"{type(exc).__name__}: {exc}", "trace": traceback.format_exc()}

    if task.disclosure_date is not None:
        detail.setdefault("disclosure_date", task.disclosure_date.isoformat())

    traj_path = None
    if trajectory_dir:
        os.makedirs(trajectory_dir, exist_ok=True)
        traj_path = os.path.join(trajectory_dir, f"{model_label}__{task.task_id}.json")
        with open(traj_path, "w") as fh:
            json.dump(
                {
                    "task_id": task.task_id,
                    "benchmark": task.benchmark.value,
                    "model": model_label,
                    "submission": submission.__dict__ if submission else None,
                    "detail": detail,
                    "elapsed_s": round(time.time() - started, 2),
                },
                fh,
                indent=2,
                default=str,
            )

    return TaskResult(
        task_id=task.task_id,
        benchmark=task.benchmark,
        model=model_label,
        solved=solved,
        score=score,
        steps_used=detail.get("steps_used", -1),
        submission=submission,
        trajectory_path=traj_path,
        grader_detail=detail,
    )
