"""Compatibility package for modular environment refactors.

Stability tier: Tier 1 (platform core).
"""

from .plugins import (
    ENVIRONMENT_ENTRYPOINT_GROUP,
    PLUGIN_ENTRYPOINT_GROUP,
    SERVER_ENTRYPOINT_GROUP,
    PluginRegistry,
    RegisteredPlugin,
    register_builtin_servers,
)

__all__ = [
    "PluginRegistry",
    "RegisteredPlugin",
    "PLUGIN_ENTRYPOINT_GROUP",
    "ENVIRONMENT_ENTRYPOINT_GROUP",
    "SERVER_ENTRYPOINT_GROUP",
    "register_builtin_servers",
]
