from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest

from atroposlib.envs.base import BaseEnv
from atroposlib.plugins import PluginRegistry, register_builtin_servers
from atroposlib.plugins.servers import SGLangBackend


def _base_env_factory() -> BaseEnv:
    return BaseEnv()


def test_plugin_registry_register_and_create_environment() -> None:
    registry = PluginRegistry()
    registry.register_environment("toy-env", _base_env_factory)

    created = registry.create_environment("toy-env")

    assert isinstance(created, BaseEnv)


def test_plugin_registry_register_builtin_servers() -> None:
    registry = PluginRegistry()

    register_builtin_servers(registry)

    backend = registry.create_server("sglang", base_url="http://localhost:30000")
    assert isinstance(backend, SGLangBackend)


def test_plugin_registry_rejects_duplicate_names() -> None:
    registry = PluginRegistry()
    registry.register_environment("duplicate", _base_env_factory)

    with pytest.raises(ValueError, match="already registered"):
        registry.register_environment("duplicate", _base_env_factory)


@dataclass
class _FakeEntryPoint:
    group: str
    name: str
    value: str
    loaded: Any

    def load(self) -> Any:
        return self.loaded


class _FakeEntryPoints:
    def __init__(self, entries: list[_FakeEntryPoint]) -> None:
        self._entries = entries

    def select(self, *, group: str) -> list[_FakeEntryPoint]:
        return [entry for entry in self._entries if entry.group == group]


def test_plugin_registry_loads_entry_points(monkeypatch: pytest.MonkeyPatch) -> None:
    registry = PluginRegistry()

    def register_bundle(plugin_registry: PluginRegistry) -> None:
        plugin_registry.register_environment("bundle-env", _base_env_factory)

    entries = _FakeEntryPoints(
        [
            _FakeEntryPoint("atropos.plugins", "bundle", "x:bundle", register_bundle),
            _FakeEntryPoint("atropos.environments", "ep-env", "x:ep_env", _base_env_factory),
            _FakeEntryPoint(
                "atropos.servers",
                "ep-server",
                "x:ep_server",
                lambda **kwargs: {"ok": True, **kwargs},
            ),
        ]
    )
    monkeypatch.setattr(PluginRegistry, "_entry_points", staticmethod(lambda: entries))

    loaded = registry.load_entry_points()

    assert len(loaded) == 3
    assert "bundle-env" in registry.environments
    assert "ep-env" in registry.environments
    assert "ep-server" in registry.servers
