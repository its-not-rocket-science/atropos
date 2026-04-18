from __future__ import annotations

from datetime import datetime, timezone

import pytest
from contracts_store_adapters import (
    CONTRACT_BACKEND_NAMES,
    BackendAdapter,
    BackendSession,
    create_backend_adapter,
)

from atroposlib.api.storage import ScoredDataGroup


@pytest.fixture(params=CONTRACT_BACKEND_NAMES)
def backend(request: pytest.FixtureRequest) -> BackendSession:
    adapter = create_backend_adapter(request.param)
    assert isinstance(adapter, BackendAdapter)
    return BackendSession(adapter=adapter, store=adapter.build())


def test_enqueue_contract(backend: BackendSession) -> None:
    now = datetime.now(tz=timezone.utc)

    first = backend.store.enqueue_job(job_id="job-1", now=now, idempotency_key="stable")
    second = backend.store.enqueue_job(job_id="job-2", now=now, idempotency_key="stable")
    third = backend.store.enqueue_job(job_id="job-3", now=now, idempotency_key="other")

    status = backend.store.get_job_status("job-1")

    assert first.job_id == "job-1"
    assert first.queue_depth == 1
    assert first.deduplicated is False
    assert second.job_id == "job-1"
    assert second.queue_depth == 1
    assert second.deduplicated is True
    assert third.job_id == "job-3"
    assert third.queue_depth == 2
    assert third.deduplicated is False
    assert status is not None
    assert status.state == "queued"
    assert status.created_at == now
    assert status.updated_at == now


def test_scored_data_buffer_and_step_status_contract(backend: BackendSession) -> None:
    result = backend.store.ingest_scored_data(
        request_id="req-lifecycle",
        groups=[
            ScoredDataGroup(
                environment_id="env-a",
                group_id="g-1",
                records=[{"sample_id": "1", "score": 0.9}],
            )
        ],
    )

    status = backend.store.get_scored_group_status(environment_id="env-a", group_id="g-1")
    metrics = backend.store.get_scored_queue_metrics(now=datetime.now(tz=timezone.utc))

    assert result.status == "completed"
    assert result.accepted_groups == 1
    assert result.accepted_count == 1
    assert status is not None
    assert status.state == "acknowledged"
    assert status.accepted_at <= status.buffered_at <= status.batched_at <= status.delivered_at
    assert status.delivered_at <= status.acknowledged_at <= status.updated_at
    assert metrics.depth == 0


def test_scored_data_dedupe_contract(backend: BackendSession) -> None:
    first = backend.store.ingest_scored_data(
        request_id="req-1",
        groups=[
            ScoredDataGroup(
                environment_id="env-a",
                group_id="g-1",
                records=[{"sample_id": "1", "score": 0.7}],
            )
        ],
    )
    duplicate = backend.store.ingest_scored_data(
        request_id="req-1",
        groups=[
            ScoredDataGroup(
                environment_id="env-a",
                group_id="g-1",
                records=[{"sample_id": "2", "score": 0.1}],
            )
        ],
    )

    listed = backend.store.list_scored_data(environment_id="env-a", limit=10)

    assert first.accepted_count == 1
    assert first.accepted_groups == 1
    assert first.deduplicated is False
    assert duplicate.accepted_count == 0
    assert duplicate.accepted_groups == 0
    assert duplicate.deduplicated is True
    assert listed == [{"sample_id": "1", "score": 0.7}]


def test_scored_data_batch_selection_and_env_registration_contract(backend: BackendSession) -> None:
    assert backend.store.list_scored_data(environment_id="missing-env", limit=10) == []

    result = backend.store.ingest_scored_data(
        request_id="req-batch",
        groups=[
            ScoredDataGroup(
                environment_id="env-a",
                group_id="g-1",
                records=[
                    {"sample_id": "a-1", "score": 0.1},
                    {"sample_id": "a-2", "score": 0.2},
                ],
            ),
            ScoredDataGroup(
                environment_id="env-b",
                group_id="g-2",
                records=[{"sample_id": "b-1", "score": 0.9}],
            ),
        ],
    )

    env_a_batch = backend.store.list_scored_data(environment_id="env-a", limit=1)
    env_a_all = backend.store.list_scored_data(environment_id="env-a", limit=10)
    env_b_all = backend.store.list_scored_data(environment_id="env-b", limit=10)

    assert result.accepted_groups == 2
    assert env_a_batch == [{"sample_id": "a-1", "score": 0.1}]
    assert env_a_all == [
        {"sample_id": "a-1", "score": 0.1},
        {"sample_id": "a-2", "score": 0.2},
    ]
    assert env_b_all == [{"sample_id": "b-1", "score": 0.9}]


def test_backend_startup_and_restart_contract(backend: BackendSession) -> None:
    startup = backend.store.startup()

    assert startup.backend_name == backend.store.backend_name
    assert startup.durable == backend.adapter.durable
    assert startup.dependency_healthy is True

    backend.store.ingest_scored_data(
        request_id="req-restart",
        groups=[
            ScoredDataGroup(
                environment_id="env-a",
                group_id="g-restart",
                records=[{"sample_id": "1", "score": 0.5}],
            )
        ],
    )
    restarted_store = backend.adapter.restart(backend.store)

    restarted_status = restarted_store.get_scored_group_status(
        environment_id="env-a",
        group_id="g-restart",
    )
    restarted_list = restarted_store.list_scored_data(environment_id="env-a", limit=10)

    if backend.adapter.durable:
        assert restarted_status is not None
        assert restarted_status.state == "acknowledged"
        assert restarted_list == [{"sample_id": "1", "score": 0.5}]
    else:
        assert restarted_status is None
        assert restarted_list == []
