"""Prompt construction utilities for benchmark environments.

This module separates benchmark-format constraints from model-specific
reasoning guidance so environments can switch prompting behavior without
rewriting task templates.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum


class PromptMode(str, Enum):
    """Supported reasoning/prompting modes for environment prompts."""

    NO_THINK = "no_think"
    THINK_TAG = "think_tag"
    PROVIDER_REASONING = "provider_reasoning"


@dataclass(frozen=True)
class BenchmarkPromptSpec:
    """Benchmark-level prompt requirements that should stay model-agnostic."""

    task_instructions: str
    output_format_requirements: str


@dataclass(frozen=True)
class PromptingConfig:
    """Runtime configuration for prompt style selection."""

    mode: PromptMode = PromptMode.NO_THINK
    provider: str | None = None


@dataclass(frozen=True)
class PromptBundle:
    """Prompt payload returned to model adapters."""

    system: str
    user: str


_PROVIDER_REASONING_TEMPLATES: Mapping[str, str] = {
    "openai": ("Reason internally and return only the final response using the required format."),
    "anthropic": ("Use concise internal reasoning and provide only the final formatted answer."),
    "generic": "Think step by step internally, then output only the final formatted answer.",
}


class PromptBuilder:
    """Compose prompts from benchmark spec and configurable reasoning guidance."""

    def __init__(self, spec: BenchmarkPromptSpec, config: PromptingConfig):
        self._spec = spec
        self._config = config

    def build(self, question: str) -> PromptBundle:
        """Build the prompt bundle for a benchmark example."""
        reasoning_guidance = self._reasoning_guidance()
        system = "\n\n".join(
            [
                self._spec.task_instructions.strip(),
                self._spec.output_format_requirements.strip(),
                reasoning_guidance,
            ]
        )
        user = f"Question:\n{question.strip()}"
        return PromptBundle(system=system, user=user)

    def _reasoning_guidance(self) -> str:
        if self._config.mode is PromptMode.NO_THINK:
            return "Do not include chain-of-thought; return only the final answer format."

        if self._config.mode is PromptMode.THINK_TAG:
            return (
                "If reasoning is needed, keep it inside <think>...</think> and place "
                "the final answer outside the tag in the required format."
            )

        provider_key = (self._config.provider or "generic").lower()
        return _PROVIDER_REASONING_TEMPLATES.get(
            provider_key,
            _PROVIDER_REASONING_TEMPLATES["generic"],
        )
