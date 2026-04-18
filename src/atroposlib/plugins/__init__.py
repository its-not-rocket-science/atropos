"""Plugin extension APIs for Atropos environments and servers.

Stability tier: Tier 3 (experimental/community extensions).
"""

from .registry import (
    ENVIRONMENT_ENTRYPOINT_GROUP,
    PLUGIN_ENTRYPOINT_GROUP,
    SERVER_ENTRYPOINT_GROUP,
    PluginRegistry,
    RegisteredPlugin,
)
from .servers import OpenAIBackend, SGLangBackend, VLLMBackend, register_builtin_servers

__all__ = [
    "PluginRegistry",
    "RegisteredPlugin",
    "PLUGIN_ENTRYPOINT_GROUP",
    "ENVIRONMENT_ENTRYPOINT_GROUP",
    "SERVER_ENTRYPOINT_GROUP",
    "OpenAIBackend",
    "VLLMBackend",
    "SGLangBackend",
    "register_builtin_servers",
]
