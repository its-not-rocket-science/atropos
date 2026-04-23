from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import pytest

from .conftest import FakeRedis, RuntimeBackend

pytest.importorskip("fastapi")


def test_environment_registration_and_listing(
    runtime_client: Any,
    runtime_backend: RuntimeBackend,
) -> None:
    created = runtime_client.post("/environments", json={"environment_id": "env-register"})
    duplicate = runtime_client.post("/environments", json={"environment_id": "env-register"})
    listed = runtime_client.get("/environments")

    assert created.status_code == 200
    assert created.json() == {"environment_id": "env-register", "created": True}
    assert duplicate.status_code == 200
    assert duplicate.json() == {"environment_id": "env-register", "created": False}
    assert listed.status_code == 200
    assert listed.json()["count"] == 1
    assert listed.json()["environments"] == ["env-register"]
    assert runtime_backend.mode in {"local", "durable"}


def test_queue_buffer_batch_and_status_reporting(
    runtime_client: Any,
    runtime_backend: RuntimeBackend,
) -> None:
    ingest = runtime_client.post(
        "/scored_data_list",
        headers={"X-Request-ID": "req-batch-1"},
        json={
            "groups": [
                {
                    "environment_id": "env-batch",
                    "group_id": "group-1",
                    "records": [
                        {"sample_id": "sample-1", "score": 0.9},
                        {"sample_id": "sample-2", "score": 0.8},
                    ],
                },
                {
                    "environment_id": "env-batch",
                    "group_id": "group-2",
                    "records": [{"sample_id": "sample-3", "score": 0.7}],
                },
            ]
        },
    )
    records = runtime_client.get(
        "/scored_data_list", params={"environment_id": "env-batch", "limit": 2}
    )

    status_group_1 = runtime_backend.store.get_scored_group_status(
        environment_id="env-batch",
        group_id="group-1",
    )
    queue_metrics = runtime_backend.store.get_scored_queue_metrics(
        now=datetime.now(tz=timezone.utc)
    )

    enqueue = runtime_client.post(
        "/jobs",
        headers={"X-Idempotency-Key": "job-key-queue"},
        json={"task": "train"},
    )
    queued_job = runtime_client.get(f"/jobs/{enqueue.json()['job_id']}")

    assert ingest.status_code == 200
    assert ingest.json()["accepted_count"] == 3
    assert ingest.json()["accepted_groups"] == 2
    assert records.status_code == 200
    assert records.json()["count"] == 2
    assert [row["sample_id"] for row in records.json()["records"]] == ["sample-1", "sample-2"]

    assert status_group_1 is not None
    assert status_group_1.state == "acknowledged"
    assert status_group_1.buffered_at is not None
    assert status_group_1.batched_at is not None
    assert status_group_1.delivered_at is not None
    assert status_group_1.acknowledged_at is not None

    assert queue_metrics.depth == 0
    assert enqueue.status_code == 200
    assert enqueue.json()["queue_depth"] == 1
    assert enqueue.json()["deduplicated"] is False
    assert queued_job.status_code == 200
    assert queued_job.json()["state"] == "queued"


def test_dedupe_and_idempotency_for_jobs_and_scored_data(runtime_client: Any) -> None:
    first_job = runtime_client.post(
        "/jobs",
        headers={"X-Idempotency-Key": "job-dedupe"},
        json={"task": "train"},
    )
    duplicate_job = runtime_client.post(
        "/jobs",
        headers={"X-Idempotency-Key": "job-dedupe"},
        json={"task": "train"},
    )

    first_scored = runtime_client.post(
        "/scored_data",
        headers={"X-Request-ID": "req-dedupe-scored"},
        json={
            "environment_id": "env-idempotency",
            "group_id": "group-idempotency",
            "records": [{"sample_id": "id-1", "score": 0.95}],
        },
    )
    duplicate_scored = runtime_client.post(
        "/scored_data",
        headers={"X-Request-ID": "req-dedupe-scored"},
        json={
            "environment_id": "env-idempotency",
            "group_id": "group-idempotency",
            "records": [{"sample_id": "id-1", "score": 0.95}],
        },
    )

    assert first_job.status_code == 200
    assert duplicate_job.status_code == 200
    assert first_job.json()["job_id"] == duplicate_job.json()["job_id"]
    assert first_job.json()["deduplicated"] is False
    assert duplicate_job.json()["deduplicated"] is True

    assert first_scored.status_code == 200
    assert first_scored.json()["accepted_count"] == 1
    assert first_scored.json()["deduplicated"] is False
    assert duplicate_scored.status_code == 200
    assert duplicate_scored.json()["accepted_count"] == 0
    assert duplicate_scored.json()["deduplicated"] is True


def test_recovery_after_restart_for_durable_store(durable_backend: RuntimeBackend) -> None:
    test_client_cls = pytest.importorskip("fastapi.testclient").TestClient

    from atroposlib.api.server import build_runtime_app
    from atroposlib.api.storage import RedisStore

    first_app = build_runtime_app(store=durable_backend.store)
    with test_client_cls(first_app) as client:
        enqueue = client.post(
            "/jobs",
            headers={"X-Idempotency-Key": "recover-job-1"},
            json={"task": "train"},
        )
        ingest = client.post(
            "/scored_data",
            headers={"X-Request-ID": "recover-scored-1"},
            json={
                "environment_id": "env-recovery",
                "group_id": "group-recovery",
                "records": [{"sample_id": "r1", "score": 0.3}],
            },
        )

    assert durable_backend.redis is not None
    restarted_store = RedisStore(redis_client=durable_backend.redis)
    restarted_app = build_runtime_app(store=restarted_store)
    with test_client_cls(restarted_app) as client:
        ready = client.get("/health/ready")
        duplicate = client.post(
            "/scored_data",
            headers={"X-Request-ID": "recover-scored-1"},
            json={
                "environment_id": "env-recovery",
                "group_id": "group-recovery",
                "records": [{"sample_id": "r1", "score": 0.3}],
            },
        )
        job_status = client.get(f"/jobs/{enqueue.json()['job_id']}")

    assert enqueue.status_code == 200
    assert ingest.status_code == 200
    assert ready.status_code == 200
    assert ready.json()["store_durable"] is True
    assert ready.json()["recovered_items"] >= 1
    assert duplicate.status_code == 200
    assert duplicate.json()["deduplicated"] is True
    assert job_status.status_code == 200
    assert job_status.json()["state"] == "queued"


def test_production_mode_configuration_validation() -> None:
    test_client_cls = pytest.importorskip("fastapi.testclient").TestClient

    from atroposlib.api.server import HardeningTier, build_runtime_app
    from atroposlib.api.storage import InMemoryStore, RedisStore

    with test_client_cls(
        build_runtime_app(
            tier=HardeningTier.PRODUCTION_SAFE,
            api_token="secret-token",
            allowed_origins=["https://internal.example"],
            store=RedisStore(redis_client=FakeRedis()),
        )
    ) as client:
        unauthorized = client.post("/jobs", json={"task": "train"})
        authorized = client.post(
            "/jobs",
            headers={"X-API-Token": "secret-token", "X-Idempotency-Key": "prod-job-1"},
            json={"task": "train"},
        )

    assert unauthorized.status_code == 401
    assert authorized.status_code == 200

    try:
        build_runtime_app(
            tier=HardeningTier.PRODUCTION_SAFE,
            api_token="secret-token",
            allowed_origins=["https://internal.example"],
            store=InMemoryStore(),
        )
    except ValueError as exc:
        assert "durable store backend" in str(exc)
    else:  # pragma: no cover - defensive guard
        raise AssertionError("production-safe tier should reject in-memory backend")
