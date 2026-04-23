from __future__ import annotations

from datetime import datetime, timezone
from importlib.util import find_spec

import pytest

if find_spec("fastapi") is None:
    pytestmark = pytest.mark.skip(reason="fastapi is not installed")


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
        bucket = self._hashes.setdefault(key, {})
        bucket.update(mapping)
        return len(mapping)

    def hgetall(self, key: str) -> dict[str, str]:
        return dict(self._hashes.get(key, {}))

    def rpush(self, key: str, value: str) -> int:
        values = self._lists.setdefault(key, [])
        values.append(value)
        return len(values)

    def llen(self, key: str) -> int:
        return len(self._lists.get(key, []))

    def lrange(self, key: str, start: int, end: int) -> list[str]:
        values = self._lists.get(key, [])
        if end < 0:
            return values[start:]
        return values[start : end + 1]

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

    def ping(self) -> bool:
        return True

    def close(self) -> None:
        return None


def test_inmemory_store_idempotency_header_deduplicates_jobs() -> None:
    from fastapi.testclient import TestClient

    from atroposlib.api.server import build_runtime_app
    from atroposlib.api.storage import InMemoryStore

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
    from fastapi.testclient import TestClient

    from atroposlib.api.server import HardeningTier, build_runtime_app
    from atroposlib.api.storage import RedisStore

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


def test_production_tier_rejects_inmemory_store() -> None:
    from atroposlib.api.server import HardeningTier, build_runtime_app
    from atroposlib.api.storage import InMemoryStore

    with pytest.raises(ValueError, match="durable store backend"):
        build_runtime_app(
            tier=HardeningTier.PRODUCTION_SAFE,
            api_token="secret",
            allowed_origins=["https://internal.example"],
            store=InMemoryStore(),
        )


def test_production_restart_recovers_job_and_dedup_state() -> None:
    from fastapi.testclient import TestClient

    from atroposlib.api.server import HardeningTier, build_runtime_app
    from atroposlib.api.storage import RedisStore

    redis = FakeRedis()

    app_first = build_runtime_app(
        tier=HardeningTier.PRODUCTION_SAFE,
        api_token="secret",
        allowed_origins=["https://internal.example"],
        store=RedisStore(redis_client=redis),
    )
    with TestClient(app_first) as client:
        first = client.post(
            "/jobs",
            json={"task": "persist-me"},
            headers={"X-API-Token": "secret", "X-Idempotency-Key": "idem-restart-1"},
        )

    app_second = build_runtime_app(
        tier=HardeningTier.PRODUCTION_SAFE,
        api_token="secret",
        allowed_origins=["https://internal.example"],
        store=RedisStore(redis_client=redis),
    )
    with TestClient(app_second) as client:
        second = client.post(
            "/jobs",
            json={"task": "persist-me"},
            headers={"X-API-Token": "secret", "X-Idempotency-Key": "idem-restart-1"},
        )
        status = client.get(f"/jobs/{first.json()['job_id']}")

    assert first.status_code == 200
    assert second.status_code == 200
    assert first.json()["job_id"] == second.json()["job_id"]
    assert second.json()["deduplicated"] is True
    assert second.json()["queue_depth"] == 1
    assert status.status_code == 200
    assert status.json()["state"] == "queued"


def test_production_restart_recovers_scored_request_dedupe_state() -> None:
    from fastapi.testclient import TestClient

    from atroposlib.api.server import HardeningTier, build_runtime_app
    from atroposlib.api.storage import RedisStore

    redis = FakeRedis()

    app_first = build_runtime_app(
        tier=HardeningTier.PRODUCTION_SAFE,
        api_token="secret",
        allowed_origins=["https://internal.example"],
        store=RedisStore(redis_client=redis),
    )
    with TestClient(app_first) as client:
        first = client.post(
            "/scored_data",
            json={"environment_id": "env-restart", "records": [{"sample_id": "a", "score": 0.6}]},
            headers={"X-API-Token": "secret", "X-Request-ID": "req-restart-1"},
        )

    app_second = build_runtime_app(
        tier=HardeningTier.PRODUCTION_SAFE,
        api_token="secret",
        allowed_origins=["https://internal.example"],
        store=RedisStore(redis_client=redis),
    )
    with TestClient(app_second) as client:
        second = client.post(
            "/scored_data",
            json={"environment_id": "env-restart", "records": [{"sample_id": "a", "score": 0.6}]},
            headers={"X-API-Token": "secret", "X-Request-ID": "req-restart-1"},
        )
        listed = client.get(
            "/scored_data_list",
            params={"environment_id": "env-restart", "limit": 100},
            headers={"X-API-Token": "secret"},
        )

    assert first.status_code == 200
    assert first.json()["deduplicated"] is False
    assert second.status_code == 200
    assert second.json()["deduplicated"] is True
    assert listed.status_code == 200
    assert listed.json()["count"] == 1


def test_redis_store_returns_status_record() -> None:
    from atroposlib.api.storage import RedisStore

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


def test_metrics_endpoint_is_exposed_and_tracks_requests() -> None:
    from fastapi.testclient import TestClient

    from atroposlib.api.server import build_runtime_app
    from atroposlib.api.storage import InMemoryStore

    app = build_runtime_app(store=InMemoryStore())
    client = TestClient(app)

    _ = client.get("/health", headers={"X-Env": "staging"})
    _ = client.post(
        "/scored_data",
        json={"environment_id": "staging", "records": [{"sample_id": "s1", "score": 0.5}]},
        headers={"X-Request-ID": "req-1"},
    )
    _ = client.post(
        "/scored_data",
        json={"environment_id": "staging", "records": [{"sample_id": "s1", "score": 0.5}]},
        headers={"X-Request-ID": "req-1"},
    )
    metrics = client.get("/metrics")

    assert metrics.status_code == 200
    assert "atropos_api_requests_total" in metrics.text
    assert "atropos_api_request_latency_seconds" in metrics.text
    assert "atropos_api_requests_by_env_total" in metrics.text
    assert "atropos_runtime_queue_oldest_age_seconds" in metrics.text
    assert "atropos_buffered_groups" in metrics.text
    assert "atropos_runtime_groups_by_state" in metrics.text
    assert "atropos_ingestion_records_total" in metrics.text
    assert "atropos_duplicate_ingestion_rejections_total" in metrics.text
    assert "atropos_duplicate_ingestion_groups_total" in metrics.text
    assert "atropos_batch_formation_latency_seconds" in metrics.text
    assert "atropos_failed_sends_total" in metrics.text
    assert "atropos_eval_duration_seconds" in metrics.text


def test_environment_registration_endpoints_are_idempotent() -> None:
    from fastapi.testclient import TestClient

    from atroposlib.api.server import build_runtime_app
    from atroposlib.api.storage import InMemoryStore

    app = build_runtime_app(store=InMemoryStore())
    client = TestClient(app)

    first = client.post("/environments", json={"environment_id": "env-contract"})
    second = client.post("/environments", json={"environment_id": "env-contract"})
    listing = client.get("/environments")

    assert first.status_code == 200
    assert first.json() == {"environment_id": "env-contract", "created": True}
    assert second.status_code == 200
    assert second.json() == {"environment_id": "env-contract", "created": False}
    assert listing.status_code == 200
    assert listing.json() == {"count": 1, "environments": ["env-contract"]}


def test_runtime_startup_with_empty_store_reports_ready() -> None:
    from fastapi.testclient import TestClient

    from atroposlib.api.server import build_runtime_app
    from atroposlib.api.storage import InMemoryStore

    app = build_runtime_app(store=InMemoryStore())
    with TestClient(app) as client:
        live = client.get("/health/live")
        ready = client.get("/health/ready")
        dependencies = client.get("/health/dependencies")

    assert live.status_code == 200
    assert live.json()["status"] == "alive"
    assert live.json()["health_state"] == "healthy"
    assert ready.status_code == 200
    assert ready.json()["status"] == "ready"
    assert ready.json()["health_state"] == "healthy"
    assert ready.json()["control_plane_ready"] is True
    assert ready.json()["backing_store_healthy"] is True
    assert ready.json()["store_durable"] is False
    assert ready.json()["recovered_items"] == 0
    assert dependencies.status_code == 200
    assert dependencies.json()["status"] == "ok"
    assert dependencies.json()["health_state"] == "healthy"


def test_runtime_startup_with_warm_durable_store_recovers_state() -> None:
    from fastapi.testclient import TestClient

    from atroposlib.api.server import HardeningTier, build_runtime_app
    from atroposlib.api.storage import RedisStore

    redis = FakeRedis()
    store = RedisStore(redis_client=redis)
    store.enqueue_job(
        job_id="job-existing",
        now=datetime.now(tz=timezone.utc),
        idempotency_key="warm-start-key",
    )

    app = build_runtime_app(
        tier=HardeningTier.PRODUCTION_SAFE,
        api_token="secret",
        allowed_origins=["https://internal.example"],
        store=store,
    )
    with TestClient(app) as client:
        ready = client.get("/health/ready")
        dependencies = client.get("/health/dependencies")

    assert ready.status_code == 200
    assert ready.json()["status"] == "ready"
    assert ready.json()["health_state"] == "healthy"
    assert ready.json()["control_plane_ready"] is True
    assert ready.json()["backing_store_healthy"] is True
    assert ready.json()["store"] == "redis"
    assert ready.json()["store_durable"] is True
    assert ready.json()["recovered_items"] >= 1
    assert dependencies.status_code == 200
    assert dependencies.json()["dependency"] == "redis"
    assert dependencies.json()["health_state"] == "healthy"


def test_runtime_readiness_reports_degraded_when_dependency_fails() -> None:
    from fastapi.testclient import TestClient

    from atroposlib.api.server import build_runtime_app
    from atroposlib.api.storage import DependencyHealth, InMemoryStore

    class UnhealthyStore(InMemoryStore):
        def dependency_health(self) -> DependencyHealth:
            return DependencyHealth(name="in_memory", healthy=False, detail="simulated outage")

    app = build_runtime_app(store=UnhealthyStore())
    with TestClient(app) as client:
        ready = client.get("/health/ready")
        dependencies = client.get("/health/dependencies")

    assert ready.status_code == 503
    assert ready.json()["status"] == "not_ready"
    assert ready.json()["health_state"] == "degraded"
    assert ready.json()["control_plane_ready"] is True
    assert ready.json()["backing_store_healthy"] is False
    assert dependencies.status_code == 503
    assert dependencies.json()["status"] == "degraded"
    assert dependencies.json()["health_state"] == "degraded"


def test_scored_data_requires_request_id_header() -> None:
    from fastapi.testclient import TestClient

    from atroposlib.api.server import build_runtime_app
    from atroposlib.api.storage import InMemoryStore

    app = build_runtime_app(store=InMemoryStore())
    client = TestClient(app)

    response = client.post(
        "/scored_data",
        json={"environment_id": "env-a", "records": [{"sample_id": "1", "score": 0.9}]},
    )

    assert response.status_code == 400
    assert "X-Request-ID or X-Idempotency-Key header is required" in response.text


def test_scored_data_deduplicates_retry_storm_requests() -> None:
    from fastapi.testclient import TestClient

    from atroposlib.api.server import build_runtime_app
    from atroposlib.api.storage import InMemoryStore

    app = build_runtime_app(store=InMemoryStore())
    client = TestClient(app)

    for _ in range(50):
        response = client.post(
            "/scored_data",
            json={
                "environment_id": "env-a",
                "records": [
                    {"sample_id": "1", "score": 0.9},
                    {"sample_id": "2", "score": 0.8},
                ],
            },
            headers={"X-Request-ID": "retry-storm-1"},
        )
        assert response.status_code == 200

    listed = client.get("/scored_data_list", params={"environment_id": "env-a", "limit": 100})
    payload = listed.json()
    assert listed.status_code == 200
    assert payload["count"] == 2
    assert [item["sample_id"] for item in payload["records"]] == ["1", "2"]


def test_scored_data_accepts_distinct_request_ids() -> None:
    from fastapi.testclient import TestClient

    from atroposlib.api.server import build_runtime_app
    from atroposlib.api.storage import InMemoryStore

    app = build_runtime_app(store=InMemoryStore())
    client = TestClient(app)

    first = client.post(
        "/scored_data",
        json={"environment_id": "env-a", "records": [{"sample_id": "1", "score": 0.9}]},
        headers={"X-Request-ID": "req-1"},
    )
    second = client.post(
        "/scored_data",
        json={"environment_id": "env-a", "records": [{"sample_id": "2", "score": 0.7}]},
        headers={"X-Request-ID": "req-2"},
    )
    listed = client.get("/scored_data_list", params={"environment_id": "env-a", "limit": 100})

    assert first.json()["deduplicated"] is False
    assert second.json()["deduplicated"] is False
    assert listed.json()["count"] == 2


def test_scored_data_list_post_deduplicates_groups() -> None:
    from fastapi.testclient import TestClient

    from atroposlib.api.server import build_runtime_app
    from atroposlib.api.storage import InMemoryStore

    app = build_runtime_app(store=InMemoryStore())
    client = TestClient(app)

    payload = {
        "groups": [
            {
                "group_id": "g-1",
                "environment_id": "env-a",
                "records": [{"sample_id": "1", "score": 0.9}],
            },
            {
                "group_id": "g-2",
                "environment_id": "env-a",
                "records": [{"sample_id": "2", "score": 0.8}],
            },
        ]
    }
    first = client.post("/scored_data_list", json=payload, headers={"X-Request-ID": "req-batch-1"})
    second = client.post("/scored_data_list", json=payload, headers={"X-Request-ID": "req-batch-1"})

    listed = client.get("/scored_data_list", params={"environment_id": "env-a", "limit": 100})
    assert first.status_code == 200
    assert first.json()["accepted_groups"] == 2
    assert first.json()["accepted_count"] == 2
    assert second.status_code == 200
    assert second.json()["deduplicated"] is True
    assert second.json()["duplicate_groups"] == 0
    assert listed.json()["count"] == 2


def test_scored_data_list_deduplicates_duplicate_groups_within_single_request() -> None:
    from fastapi.testclient import TestClient

    from atroposlib.api.server import build_runtime_app
    from atroposlib.api.storage import InMemoryStore

    app = build_runtime_app(store=InMemoryStore())
    client = TestClient(app)

    response = client.post(
        "/scored_data_list",
        json={
            "groups": [
                {
                    "group_id": "g-repeat",
                    "environment_id": "env-a",
                    "records": [{"sample_id": "1", "score": 0.9}],
                },
                {
                    "group_id": "g-repeat",
                    "environment_id": "env-a",
                    "records": [{"sample_id": "1", "score": 0.9}],
                },
            ]
        },
        headers={"X-Request-ID": "req-dup-inside"},
    )
    listed = client.get("/scored_data_list", params={"environment_id": "env-a", "limit": 100})

    assert response.status_code == 200
    assert response.json()["accepted_groups"] == 1
    assert response.json()["accepted_count"] == 1
    assert response.json()["duplicate_groups"] == 1
    assert listed.status_code == 200
    assert listed.json()["count"] == 1


def test_scored_data_list_partial_failure_retry_only_accepts_new_groups() -> None:
    from fastapi.testclient import TestClient

    from atroposlib.api.server import build_runtime_app
    from atroposlib.api.storage import InMemoryStore

    app = build_runtime_app(store=InMemoryStore())
    client = TestClient(app)

    first = client.post(
        "/scored_data_list",
        json={
            "groups": [
                {
                    "group_id": "g-1",
                    "environment_id": "env-a",
                    "records": [{"sample_id": "1", "score": 0.9}],
                },
                {
                    "group_id": "g-2",
                    "environment_id": "env-a",
                    "records": ["invalid-record"],
                },
            ]
        },
        headers={"X-Request-ID": "req-partial-1"},
    )
    retry = client.post(
        "/scored_data_list",
        json={
            "groups": [
                {
                    "group_id": "g-1",
                    "environment_id": "env-a",
                    "records": [{"sample_id": "1", "score": 0.9}],
                },
                {
                    "group_id": "g-2",
                    "environment_id": "env-a",
                    "records": [{"sample_id": "2", "score": 0.8}],
                },
            ]
        },
        headers={"X-Request-ID": "req-partial-1"},
    )

    listed = client.get("/scored_data_list", params={"environment_id": "env-a", "limit": 100})
    assert first.status_code == 200
    assert first.json()["status"] == "partial_failed"
    assert first.json()["accepted_groups"] == 1
    assert first.json()["failed_groups"] == 1
    assert retry.status_code == 200
    assert retry.json()["status"] == "completed"
    assert retry.json()["accepted_groups"] == 1
    assert retry.json()["duplicate_groups"] == 1
    assert listed.json()["count"] == 2
