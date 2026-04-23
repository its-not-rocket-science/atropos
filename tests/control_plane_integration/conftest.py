from __future__ import annotations

from dataclasses import dataclass
from importlib.util import find_spec
from typing import Any

import pytest

if find_spec("fastapi") is None:
    pytestmark = pytest.mark.skip(reason="fastapi is not installed")


class FakeRedis:
    """CI-safe Redis stand-in for durable-store integration paths."""

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
        keys = [*self._kv, *self._hashes, *self._lists]
        if match is not None:
            prefix = match.rstrip("*")
            keys = [key for key in keys if key.startswith(prefix)]
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

    def close(self) -> None:
        return None


@dataclass(frozen=True)
class RuntimeBackend:
    mode: str
    store: Any
    durable: bool
    redis: FakeRedis | None


@pytest.fixture(params=["local", "durable"], ids=["local-inmemory", "durable-redis"])
def runtime_backend(request: pytest.FixtureRequest) -> RuntimeBackend:
    from atroposlib.api.storage import InMemoryStore, RedisStore

    mode = str(request.param)
    if mode == "local":
        return RuntimeBackend(mode="local", store=InMemoryStore(), durable=False, redis=None)

    redis = FakeRedis()
    store = RedisStore(redis_client=redis)
    return RuntimeBackend(mode="durable", store=store, durable=True, redis=redis)


@pytest.fixture
def runtime_client(runtime_backend: RuntimeBackend) -> Any:
    test_client_cls = pytest.importorskip("fastapi.testclient").TestClient

    from atroposlib.api.server import build_runtime_app

    app = build_runtime_app(store=runtime_backend.store)
    with test_client_cls(app) as client:
        yield client


@pytest.fixture
def durable_backend() -> RuntimeBackend:
    from atroposlib.api.storage import RedisStore

    redis = FakeRedis()
    store = RedisStore(redis_client=redis)
    return RuntimeBackend(mode="durable", store=store, durable=True, redis=redis)
