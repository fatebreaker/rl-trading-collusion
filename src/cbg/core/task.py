"""Core task abstractions shared across all benchmarks.

The whole point of this project is that CyberGym, CTF suites, and CVE-Bench
all reduce to the same shape: a task hands the agent an environment and a goal,
the agent produces a submission, and a grader turns that submission into a
binary (or scored) outcome. Keeping this interface tiny is what lets us run one
fixed scaffold everywhere.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from enum import Enum
from typing import Any


class Benchmark(str, Enum):
    CYBERGYM = "cybergym"
    CYBENCH = "cybench"
    CVEBENCH = "cvebench"


@dataclass(frozen=True)
class Task:
    """A single benchmark task, normalized to a common shape."""

    task_id: str
    benchmark: Benchmark
    prompt: str
    # Root of the working directory / codebase mounted for the agent.
    workdir: str
    # Free-form, adapter-specific data the grader needs (fix commit, flag,
    # exploit oracle config, ...). Never shown to the agent verbatim.
    grader_spec: dict[str, Any] = field(default_factory=dict)
    # Date the underlying vuln/challenge became public, for contamination
    # analysis. None when unknown.
    disclosure_date: date | None = None
    # Coarse difficulty / category tags used for stratified sampling.
    tags: tuple[str, ...] = ()


@dataclass
class Submission:
    """What the agent hands back. Interpretation is benchmark-specific."""

    # e.g. a PoC file path (CyberGym), a flag string (CTF), an exploit dir.
    payload: str
    kind: str  # "poc" | "flag" | "exploit" | "patch"
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class TaskResult:
    task_id: str
    benchmark: Benchmark
    model: str
    solved: bool
    score: float  # 0.0/1.0 for binary graders; continuous where supported
    steps_used: int
    submission: Submission | None
    failure_mode: str | None = None  # filled by taxonomy labeling pass
    trajectory_path: str | None = None  # where the full transcript is stored
    grader_detail: dict[str, Any] = field(default_factory=dict)
