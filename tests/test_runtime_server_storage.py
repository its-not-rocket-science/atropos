from __future__ import annotations

from datetime import datetime, timezone

from fastapi.testclient import TestClient

from atroposlib.api.server import HardeningTier, build_runtime_app
from atroposlib.api.storage import InMemoryStore, RedisStore


class FakeRedis:
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
        self._hashes[key] = dict(mapping)
        return len(mapping)

    def hgetall(self, key: str) -> dict[str, str]:
        return dict(self._hashes.get(key, {}))

    def rpush(self, key: str, value: str) -> int:
        values = self._lists.setdefault(key, [])
        values.append(value)
        return len(values)

    def llen(self, key: str) -> int:
        return len(self._lists.get(key, []))

    def scan(self, *, cursor: int, match: str, count: int) -> tuple[int, list[str]]:
        _ = count
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


def test_inmemory_store_idempotency_header_deduplicates_jobs() -> None:
    app = build_runtime_app(store=InMemoryStore())
    client = TestClient(app)

    first = client.post("/jobs", json={"task": "a"}, headers={"X-Idempotency-Key": "same"})
    second = client.post("/jobs", json={"task": "a"}, headers={"X-Idempotency-Key": "same"})

    assert first.status_code == 200
    assert second.status_code == 200
    assert first.json()["job_id"] == second.json()["job_id"]
    assert first.json()["deduplicated"] is False
    assert second.json()["deduplicated"] is True
    assert second.json()["queue_depth"] == 1


def test_production_tier_uses_injected_redis_store() -> None:
    store = RedisStore(redis_client=FakeRedis())
    app = build_runtime_app(
        tier=HardeningTier.PRODUCTION_SAFE,
        api_token="secret",
        allowed_origins=["https://internal.example"],
        store=store,
    )
    client = TestClient(app)

    no_token = client.post("/jobs", json={"task": "a"})
    with_token = client.post("/jobs", json={"task": "a"}, headers={"X-API-Token": "secret"})
    health = client.get("/health")

    assert no_token.status_code == 401
    assert with_token.status_code == 200
    assert health.status_code == 200
    assert health.json()["store"] == "redis"


def test_redis_store_returns_status_record() -> None:
    now = datetime.now(tz=timezone.utc)
    store = RedisStore(redis_client=FakeRedis())

    enqueue_result = store.enqueue_job(job_id="job-1", now=now, idempotency_key="idem-1")
    status = store.get_job_status("job-1")

    assert enqueue_result.job_id == "job-1"
    assert enqueue_result.queue_depth == 1
    assert enqueue_result.deduplicated is False
    assert status is not None
    assert status.job_id == "job-1"
    assert status.state == "queued"
