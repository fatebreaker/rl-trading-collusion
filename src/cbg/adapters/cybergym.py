"""CyberGym adapter (stub).

CyberGym tasks derive from OSS-Fuzz vulnerabilities: the agent is given a
codebase and must produce a PoC input that triggers the vuln before the fix
commit and is neutralized after it. This adapter is responsible for:

  1. Loading the task set (task_id, codebase snapshot, fix-commit metadata).
  2. Building a `Task` whose `grader_spec` carries the pre/post-patch build
     recipe and expected crash signature.
  3. A Grader that runs the submitted PoC against both builds in a sandbox and
     checks: crashes pre-patch AND clean post-patch AND matching sanitizer
     signature (the last guards against "crashed for the wrong reason").

Implementation notes for wiring:
  - Vendor CyberGym task data under data/cybergym/ (not committed).
  - Run builds in a container with no network egress.
  - `disclosure_date` should be set from the OSS-Fuzz report date to feed the
    contamination analysis.
"""

from __future__ import annotations

from typing import Any, Iterable

from ..core.task import Benchmark, Submission, Task


def load_tasks(data_root: str) -> Iterable[Task]:
    raise NotImplementedError("wire in CyberGym task loading from data_root")


class CyberGymGrader:
    def grade(self, task: Task, submission: Submission | None) -> tuple[bool, float, dict[str, Any]]:
        # differential check: pre-patch crash ∧ post-patch clean ∧ signature match
        raise NotImplementedError("run PoC against pre/post-patch builds in sandbox")
