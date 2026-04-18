from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from atroposlib.api.storage import AtroposStore, InMemoryStore, RedisStore


class FakeRedis:
    """Lightweight Redis double for backend contract tests."""

    def __init__(self) -> None:
        self._kv: dict[str, str] = {}
        self._hashes: dict[str, dict[str, str]] = {}
        self._lists: dict[str, list[str]] = {}

    def set(self, key: str, value: str, *, nx: bool = False, ex: int | None = None) -> bool:
        _ = ex
        if nx and key in self._kv:
            return False
        self._kv[key] = value
        return True

    def get(self, key: str) -> str | None:
        return self._kv.get(key)

    def hset(self, key: str, mapping: dict[str, str]) -> int:
        bucket = self._hashes.setdefault(key, {})
        bucket.update(mapping)
        return len(mapping)

    def hgetall(self, key: str) -> dict[str, str]:
        return dict(self._hashes.get(key, {}))

    def rpush(self, key: str, *values: str) -> int:
        bucket = self._lists.setdefault(key, [])
        bucket.extend(values)
        return len(bucket)

    def llen(self, key: str) -> int:
        return len(self._lists.get(key, []))

    def lrange(self, key: str, start: int, end: int) -> list[str]:
        values = self._lists.get(key, [])
        if end < 0:
            return values[start:]
        return values[start : end + 1]

    def scan(
        self,
        cursor: int = 0,
        match: str | None = None,
        count: int | None = None,
    ) -> tuple[int, list[str]]:
        _ = count
        if not match:
            keys = [*self._kv, *self._hashes, *self._lists]
            return (0, keys if cursor == 0 else [])

        prefix = match.rstrip("*")
        keys = [key for key in [*self._kv, *self._hashes, *self._lists] if key.startswith(prefix)]
        return (0, keys if cursor == 0 else [])

    def delete(self, *keys: str) -> int:
        deleted = 0
        for key in keys:
            deleted += int(self._kv.pop(key, None) is not None)
            deleted += int(self._hashes.pop(key, None) is not None)
            deleted += int(self._lists.pop(key, None) is not None)
        return deleted

    def ping(self) -> bool:
        return True


@dataclass(frozen=True, slots=True)
class BackendAdapter:
    """Factory adapter that allows the same contract tests to run per backend."""

    name: str
    durable: bool
    build: Callable[[], AtroposStore]
    restart: Callable[[AtroposStore], AtroposStore]


@dataclass(frozen=True, slots=True)
class BackendSession:
    """Backend instance plus adapter metadata for a contract-test run."""

    adapter: BackendAdapter
    store: AtroposStore


CONTRACT_BACKEND_NAMES = ("inmemory", "redis")


def create_backend_adapter(name: str) -> BackendAdapter:
    """Build a backend adapter by name for store contract tests."""

    if name == "inmemory":
        return BackendAdapter(
            name="inmemory",
            durable=False,
            build=InMemoryStore,
            restart=lambda _store: InMemoryStore(),
        )

    if name == "redis":
        shared = FakeRedis()
        return BackendAdapter(
            name="redis",
            durable=True,
            build=lambda: RedisStore(redis_client=shared),
            restart=lambda _store: RedisStore(redis_client=shared),
        )

    raise ValueError(f"Unknown contract backend adapter: {name}")
