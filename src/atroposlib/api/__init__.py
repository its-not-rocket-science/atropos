"""FastAPI helpers for atroposlib runtime services.

Stability tier: Tier 1 (platform core).
"""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .server import HardeningTier, RuntimeStatus, build_runtime_app, get_runtime_state
    from .storage import AtroposStore, InMemoryStore, PostgresStore, RedisStore, RuntimeStore

__all__ = [
    "HardeningTier",
    "RuntimeStatus",
    "AtroposStore",
    "RuntimeStore",
    "InMemoryStore",
    "RedisStore",
    "PostgresStore",
    "build_runtime_app",
    "get_runtime_state",
]


def __getattr__(name: str) -> object:
    if name in {
        "HardeningTier",
        "RuntimeStatus",
        "build_runtime_app",
        "get_runtime_state",
    }:
        server = import_module("atroposlib.api.server")
        return getattr(server, name)
    if name in {
        "AtroposStore",
        "RuntimeStore",
        "InMemoryStore",
        "RedisStore",
        "PostgresStore",
    }:
        storage = import_module("atroposlib.api.storage")
        return getattr(storage, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
