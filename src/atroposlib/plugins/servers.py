"""Decoupled server backend definitions for Atropos plugins."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class HttpServerBackend:
    """Simple HTTP backend abstraction used by server plugins."""

    base_url: str
    timeout_seconds: int = 30

    def invoke(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Placeholder invoke implementation for integration seams/tests."""

        return {
            "ok": True,
            "base_url": self.base_url,
            "timeout_seconds": self.timeout_seconds,
            "request": payload,
        }


class OpenAIBackend(HttpServerBackend):
    """OpenAI-compatible backend plugin."""


class VLLMBackend(HttpServerBackend):
    """vLLM-compatible backend plugin."""


class SGLangBackend(HttpServerBackend):
    """SGLang-compatible backend plugin."""


def register_builtin_servers(registry: Any) -> None:
    """Register core decoupled server backends in a plugin registry."""

    registry.register_server("openai", OpenAIBackend)
    registry.register_server("vllm", VLLMBackend)
    registry.register_server("sglang", SGLangBackend)
