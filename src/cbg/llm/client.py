"""Thin, provider-agnostic LLM client.

The scaffold must be byte-identical across providers, so all model differences
are confined here. Only the chat + tool-call surface the ReAct agent needs is
exposed. Real provider calls are stubbed — wire in the SDKs (anthropic, openai,
etc.) behind `complete`. Kept dependency-free so the harness imports cleanly in
CI before any keys are configured.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Message:
    role: str  # "system" | "user" | "assistant" | "tool"
    content: str
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    tool_call_id: str | None = None


@dataclass
class Completion:
    text: str
    tool_calls: list[dict[str, Any]]
    stop_reason: str
    usage: dict[str, int] = field(default_factory=dict)


# Model registry: label -> (provider, api_model_id, training_cutoff).
# training_cutoff feeds the contamination analysis (RQ3).
MODELS: dict[str, dict[str, str]] = {
    "claude-opus": {"provider": "anthropic", "id": "claude-opus-4-8", "cutoff": "2026-01"},
    "gpt": {"provider": "openai", "id": "gpt-5", "cutoff": "2025-06"},
    # add an open-weights model served via an OpenAI-compatible endpoint
}


class LLMClient:
    def __init__(self, model_label: str):
        if model_label not in MODELS:
            raise KeyError(f"unknown model label: {model_label}")
        self.label = model_label
        self.cfg = MODELS[model_label]
        self.provider = self.cfg["provider"]

    def complete(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]],
        max_tokens: int = 4096,
        temperature: float = 0.0,
    ) -> Completion:
        """Dispatch to the configured provider.

        Providers are wired in behind this single method so the agent code
        never branches on provider. Implementations should translate `messages`
        + `tools` into the provider's native format and normalize the response
        back into a Completion.
        """
        raise NotImplementedError(
            f"provider '{self.provider}' not wired in yet; "
            "implement the SDK call here (see MODELS registry)."
        )

    @staticmethod
    def available() -> bool:
        return bool(os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("OPENAI_API_KEY"))
