"""Plugin registry for third-party environments and server backends.

This module provides a light-weight extension mechanism inspired by Gym's
registry patterns and Transformers' auto-discovery architecture.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from importlib import metadata
from typing import Any, Protocol, runtime_checkable

from atroposlib.envs.base import BaseEnv

EnvironmentFactory = Callable[..., BaseEnv]
ServerFactory = Callable[..., Any]

PLUGIN_ENTRYPOINT_GROUP = "atropos.plugins"
ENVIRONMENT_ENTRYPOINT_GROUP = "atropos.environments"
SERVER_ENTRYPOINT_GROUP = "atropos.servers"


@runtime_checkable
class RegistryPlugin(Protocol):
    """Protocol for plugin objects that can register components."""

    def register(self, registry: PluginRegistry) -> None:
        """Register one or more components in a registry."""


@dataclass(frozen=True)
class RegisteredPlugin:
    """Metadata for a loaded plugin entry point."""

    group: str
    name: str
    value: str


class PluginRegistry:
    """Registry for environments and server backends.

    The registry supports direct in-process registration as well as
    auto-discovery from Python entry points.
    """

    def __init__(self) -> None:
        self._environment_factories: dict[str, EnvironmentFactory] = {}
        self._server_factories: dict[str, ServerFactory] = {}
        self._loaded_plugins: list[RegisteredPlugin] = []

    def register_environment(self, name: str, factory: EnvironmentFactory) -> None:
        """Register an environment constructor by name."""

        self._validate_name(name)
        if name in self._environment_factories:
            raise ValueError(f"Environment {name!r} is already registered.")
        self._environment_factories[name] = factory

    def register_server(self, name: str, factory: ServerFactory) -> None:
        """Register a server backend constructor by name."""

        self._validate_name(name)
        if name in self._server_factories:
            raise ValueError(f"Server backend {name!r} is already registered.")
        self._server_factories[name] = factory

    def create_environment(self, name: str, **kwargs: Any) -> BaseEnv:
        """Create an environment instance from a registered factory."""

        factory = self._environment_factories.get(name)
        if factory is None:
            known = ", ".join(sorted(self._environment_factories)) or "<none>"
            raise KeyError(f"Unknown environment {name!r}. Registered environments: {known}.")
        return factory(**kwargs)

    def create_server(self, name: str, **kwargs: Any) -> Any:
        """Create a server backend instance from a registered factory."""

        factory = self._server_factories.get(name)
        if factory is None:
            known = ", ".join(sorted(self._server_factories)) or "<none>"
            raise KeyError(f"Unknown server backend {name!r}. Registered servers: {known}.")
        return factory(**kwargs)

    @property
    def environments(self) -> Mapping[str, EnvironmentFactory]:
        """View all currently registered environment factories."""

        return self._environment_factories

    @property
    def servers(self) -> Mapping[str, ServerFactory]:
        """View all currently registered server factories."""

        return self._server_factories

    @property
    def loaded_plugins(self) -> tuple[RegisteredPlugin, ...]:
        """Metadata for all entry points loaded through this registry."""

        return tuple(self._loaded_plugins)

    def load_entry_points(self) -> list[RegisteredPlugin]:
        """Load plugins from supported Atropos entry point groups."""

        discovered = self._entry_points()
        loaded: list[RegisteredPlugin] = []

        for entry_point in discovered.select(group=PLUGIN_ENTRYPOINT_GROUP):
            self._register_from_bundle(entry_point.load())
            loaded_plugin = RegisteredPlugin(
                group=PLUGIN_ENTRYPOINT_GROUP,
                name=entry_point.name,
                value=entry_point.value,
            )
            self._loaded_plugins.append(loaded_plugin)
            loaded.append(loaded_plugin)

        for entry_point in discovered.select(group=ENVIRONMENT_ENTRYPOINT_GROUP):
            self.register_environment(entry_point.name, entry_point.load())
            loaded_plugin = RegisteredPlugin(
                group=ENVIRONMENT_ENTRYPOINT_GROUP,
                name=entry_point.name,
                value=entry_point.value,
            )
            self._loaded_plugins.append(loaded_plugin)
            loaded.append(loaded_plugin)

        for entry_point in discovered.select(group=SERVER_ENTRYPOINT_GROUP):
            self.register_server(entry_point.name, entry_point.load())
            loaded_plugin = RegisteredPlugin(
                group=SERVER_ENTRYPOINT_GROUP,
                name=entry_point.name,
                value=entry_point.value,
            )
            self._loaded_plugins.append(loaded_plugin)
            loaded.append(loaded_plugin)

        return loaded

    @staticmethod
    def _validate_name(name: str) -> None:
        if not name or not name.strip():
            raise ValueError("Plugin names must be non-empty strings.")

    @staticmethod
    def _entry_points() -> metadata.EntryPoints:
        entry_points = metadata.entry_points()
        if hasattr(entry_points, "select"):
            return entry_points
        return metadata.EntryPoints(entry_points)  # pragma: no cover

    def _register_from_bundle(self, bundle: Any) -> None:
        if callable(bundle) and not isinstance(bundle, RegistryPlugin):
            bundle(self)
            return

        if isinstance(bundle, RegistryPlugin):
            bundle.register(self)
            return

        raise TypeError("Plugin bundle must be a callable(registry) or expose register(registry).")
