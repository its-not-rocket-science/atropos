from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from importlib.util import find_spec
from typing import Any

import pytest

from atroposlib.envs.base import BaseEnv
from atroposlib.envs.transport_client import TransportClient

if find_spec("fastapi") is None:
    pytestmark = pytest.mark.skip(reason="fastapi is not installed")


class FakeRedis:
    """Small Redis double suitable for CI-safe integration testing."""

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

    def close(self) -> None:
        return None


@dataclass
class DeterministicScoringTransport(TransportClient):
    def send(self, payload: dict[str, Any]) -> dict[str, Any]:
        samples = payload["work_item"].get("samples", [])
        scored = [
            {
                "sample_id": sample["sample_id"],
                "score": float(len(sample["text"])),
                "text": sample["text"],
            }
            for sample in samples
        ]
        return {"ok": True, "scored_records": scored}


@pytest.fixture(params=["inmemory", "redis"], ids=["inmemory", "redis"])
def runtime_mode(request: pytest.FixtureRequest) -> str:
    return str(request.param)


@pytest.fixture
def runtime_resources(runtime_mode: str) -> dict[str, Any]:
    from atroposlib.api.storage import InMemoryStore, RedisStore

    if runtime_mode == "inmemory":
        return {"mode": runtime_mode, "store": InMemoryStore(), "redis": None}

    redis = FakeRedis()
    return {"mode": runtime_mode, "store": RedisStore(redis_client=redis), "redis": redis}


@pytest.fixture
def runtime_client(runtime_resources: dict[str, Any]) -> Any:
    from fastapi.testclient import TestClient

    from atroposlib.api.server import build_runtime_app

    app = build_runtime_app(store=runtime_resources["store"])
    with TestClient(app) as client:
        yield client


@pytest.fixture
def registered_environment() -> Any:
    registry: dict[str, BaseEnv] = {}

    def _register(environment_id: str) -> str:
        registry[environment_id] = BaseEnv(transport=DeterministicScoringTransport())
        return environment_id

    return _register, registry


def test_e2e_env_registration_ingestion_and_idempotent_duplicates(
    runtime_client: Any,
    registered_environment: Any,
) -> None:
    register, registry = registered_environment
    environment_id = register("env-e2e")
    env = registry[environment_id]

    rollout = env.step(
        {
            "env": environment_id,
            "samples": [
                {"sample_id": "s1", "text": "alpha"},
                {"sample_id": "s2", "text": "beta beta"},
            ],
        },
        worker_count=2,
    )

    first = runtime_client.post(
        "/scored_data",
        headers={"X-Request-ID": "req-env-register"},
        json={"environment_id": environment_id, "records": rollout["scored_records"]},
    )
    duplicate = runtime_client.post(
        "/scored_data",
        headers={"X-Request-ID": "req-env-register"},
        json={"environment_id": environment_id, "records": rollout["scored_records"]},
    )
    listed = runtime_client.get(
        "/scored_data_list",
        params={"environment_id": environment_id, "limit": 10},
    )

    assert first.status_code == 200
    assert first.json()["accepted_count"] == 2
    assert first.json()["deduplicated"] is False
    assert duplicate.status_code == 200
    assert duplicate.json()["accepted_count"] == 0
    assert duplicate.json()["deduplicated"] is True
    assert listed.status_code == 200
    assert listed.json()["count"] == 2


def test_e2e_mismatched_group_sizes_batch_creation_and_status_reporting(
    runtime_client: Any,
    runtime_resources: dict[str, Any],
) -> None:
    response = runtime_client.post(
        "/scored_data_list",
        headers={"X-Request-ID": "req-groups-1"},
        json={
            "groups": [
                {
                    "environment_id": "env-groups",
                    "group_id": "group-large",
                    "records": [
                        {"sample_id": "a", "score": 0.9},
                        {"sample_id": "b", "score": 0.8},
                        {"sample_id": "c", "score": 0.7},
                    ],
                },
                {
                    "environment_id": "env-groups",
                    "group_id": "group-small",
                    "records": [{"sample_id": "d", "score": 0.6}],
                },
            ]
        },
    )

    assert response.status_code == 200
    assert response.json()["accepted_groups"] == 2
    assert response.json()["accepted_count"] == 4

    store = runtime_resources["store"]
    large_group = store.get_scored_group_status(environment_id="env-groups", group_id="group-large")
    small_group = store.get_scored_group_status(environment_id="env-groups", group_id="group-small")

    assert large_group is not None
    assert small_group is not None
    assert large_group.state == "acknowledged"
    assert small_group.state == "acknowledged"
    assert large_group.buffered_at is not None
    assert small_group.buffered_at is not None
    assert large_group.batched_at is not None
    assert small_group.batched_at is not None

    trainer_batch = runtime_client.get(
        "/scored_data_list",
        params={"environment_id": "env-groups", "limit": 2},
    )
    assert trainer_batch.status_code == 200
    assert trainer_batch.json()["count"] == 2
    assert [item["sample_id"] for item in trainer_batch.json()["records"]] == ["a", "b"]

    enqueued = runtime_client.post(
        "/jobs",
        headers={"X-Idempotency-Key": "job-key-1"},
        json={"task": "train"},
    )
    job_id = enqueued.json()["job_id"]
    job_status = runtime_client.get(f"/jobs/{job_id}")
    ready = runtime_client.get("/health/ready")

    assert enqueued.status_code == 200
    assert job_status.status_code == 200
    assert job_status.json()["state"] == "queued"
    assert ready.status_code == 200
    assert ready.json()["status"] == "ready"


def test_e2e_restart_recovery_for_durable_backend() -> None:
    from fastapi.testclient import TestClient

    from atroposlib.api.server import build_runtime_app
    from atroposlib.api.storage import RedisStore

    redis = FakeRedis()

    first_store = RedisStore(redis_client=redis)
    first_app = build_runtime_app(store=first_store)
    with TestClient(first_app) as first_client:
        enqueue = first_client.post(
            "/jobs",
            headers={"X-Idempotency-Key": "restart-job"},
            json={"task": "train"},
        )
        ingest = first_client.post(
            "/scored_data",
            headers={"X-Request-ID": "req-restart"},
            json={
                "environment_id": "env-restart",
                "group_id": "group-restart",
                "records": [{"sample_id": "restart-s1", "score": 0.3}],
            },
        )

    assert enqueue.status_code == 200
    assert ingest.status_code == 200

    restarted_store = RedisStore(redis_client=redis)
    restarted_app = build_runtime_app(store=restarted_store)
    with TestClient(restarted_app) as restarted_client:
        ready = restarted_client.get("/health/ready")
        duplicate = restarted_client.post(
            "/scored_data",
            headers={"X-Request-ID": "req-restart"},
            json={
                "environment_id": "env-restart",
                "group_id": "group-restart",
                "records": [{"sample_id": "restart-s2", "score": 0.1}],
            },
        )

    status = restarted_store.get_scored_group_status(
        environment_id="env-restart",
        group_id="group-restart",
    )
    metrics = restarted_store.get_scored_queue_metrics(now=datetime.now(tz=timezone.utc))

    assert ready.status_code == 200
    assert ready.json()["store_durable"] is True
    assert ready.json()["recovered_items"] >= 1
    assert duplicate.status_code == 200
    assert duplicate.json()["deduplicated"] is True
    assert status is not None
    assert status.state == "acknowledged"
    assert metrics.depth == 0
