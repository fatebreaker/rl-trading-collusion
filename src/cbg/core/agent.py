"""Agent and tool protocols.

An Agent consumes a Task and produces a Submission. A Grader (defined per
adapter) turns the Submission into a TaskResult. Because the scaffold is held
fixed across the study, there is intentionally exactly one production Agent
implementation (agents.react_agent.ReactAgent); this protocol exists so the
runner and the analysis code do not depend on it directly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from .task import Submission, Task


@dataclass
class ToolCall:
    name: str
    args: dict[str, Any]


@dataclass
class ToolResult:
    output: str
    is_error: bool = False
    # True when the agent invoked `submit`; ends the episode.
    terminal: bool = False


class Tool(Protocol):
    name: str

    def spec(self) -> dict[str, Any]:
        """JSON schema for the tool, passed to the model."""
        ...

    def run(self, args: dict[str, Any], task: Task) -> ToolResult:
        ...


class Agent(Protocol):
    def solve(self, task: Task) -> Submission | None:
        """Run one episode against the task, returning a Submission or None."""
        ...


class Grader(Protocol):
    def grade(self, task: Task, submission: Submission | None) -> tuple[bool, float, dict[str, Any]]:
        """Return (solved, score, detail)."""
        ...
