"""Example pip-installable Atropos plugin package."""

from __future__ import annotations

from atroposlib.envs.base import BaseEnv
from atroposlib.plugins import PluginRegistry


class ExampleThirdPartyEnvironment(BaseEnv):
    """Simple example environment provided by a third-party package."""


def create_environment() -> BaseEnv:
    """Factory for the example third-party environment."""

    return ExampleThirdPartyEnvironment()


def create_openai_like_server(base_url: str = "https://api.openai.com") -> dict[str, str]:
    """Factory for a server backend instance owned by this plugin package."""

    return {"provider": "openai-compatible", "base_url": base_url}


def register(registry: PluginRegistry) -> None:
    """Entry point invoked by Atropos plugin discovery."""

    registry.register_environment("third_party/gsm8k-example", create_environment)
    registry.register_server("third_party/openai-compatible", create_openai_like_server)
