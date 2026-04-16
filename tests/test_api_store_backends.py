from __future__ import annotations

from collections.abc import Callable
from datetime import datetime, timezone

import pytest

from atroposlib.api.storage import AtroposStore, InMemoryStore, RedisStore, ScoredDataGroup


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


StoreFactory = Callable[[], AtroposStore]


@pytest.fixture(
    params=[
        pytest.param(lambda: InMemoryStore(), id="inmemory"),
        pytest.param(lambda: RedisStore(redis_client=FakeRedis()), id="redis"),
    ]
)
def store(request: pytest.FixtureRequest) -> AtroposStore:
    factory: StoreFactory = request.param
    return factory()


def test_enqueue_idempotency_consistency(store: AtroposStore) -> None:
    now = datetime.now(tz=timezone.utc)

    first = store.enqueue_job(job_id="job-1", now=now, idempotency_key="stable")
    second = store.enqueue_job(job_id="job-2", now=now, idempotency_key="stable")

    assert first.job_id == "job-1"
    assert first.deduplicated is False
    assert first.queue_depth == 1
    assert second.job_id == "job-1"
    assert second.deduplicated is True
    assert second.queue_depth == 1

    status = store.get_job_status("job-1")
    assert status is not None
    assert status.state == "queued"


def test_scored_data_deduplication_consistency(store: AtroposStore) -> None:
    first = store.ingest_scored_data(
        request_id="req-1",
        groups=[
            ScoredDataGroup(
                environment_id="env-a",
                group_id="g-1",
                records=[{"sample_id": "1", "score": 0.7}],
            )
        ],
    )
    duplicate = store.ingest_scored_data(
        request_id="req-1",
        groups=[
            ScoredDataGroup(
                environment_id="env-a",
                group_id="g-1",
                records=[{"sample_id": "2", "score": 0.1}],
            )
        ],
    )

    listed = store.list_scored_data(environment_id="env-a", limit=10)

    assert first.accepted_count == 1
    assert first.accepted_groups == 1
    assert first.deduplicated is False
    assert duplicate.accepted_count == 0
    assert duplicate.accepted_groups == 0
    assert duplicate.deduplicated is True
    assert listed == [{"sample_id": "1", "score": 0.7}]


def test_scored_data_partial_failure_resume_consistency(store: AtroposStore) -> None:
    first = store.ingest_scored_data(
        request_id="req-partial",
        groups=[
            ScoredDataGroup(
                environment_id="env-a",
                group_id="g-1",
                records=[{"sample_id": "1", "score": 0.7}],
            ),
            ScoredDataGroup(
                environment_id="env-a",
                group_id="g-2",
                records=["not-an-object"],  # type: ignore[list-item]
            ),
        ],
    )
    retry = store.ingest_scored_data(
        request_id="req-partial",
        groups=[
            ScoredDataGroup(
                environment_id="env-a",
                group_id="g-1",
                records=[{"sample_id": "1", "score": 0.7}],
            ),
            ScoredDataGroup(
                environment_id="env-a",
                group_id="g-2",
                records=[{"sample_id": "2", "score": 0.6}],
            ),
        ],
    )

    listed = store.list_scored_data(environment_id="env-a", limit=10)
    assert first.status == "partial_failed"
    assert first.accepted_groups == 1
    assert first.failed_groups == 1
    assert retry.status == "completed"
    assert retry.accepted_groups == 1
    assert listed == [{"sample_id": "1", "score": 0.7}, {"sample_id": "2", "score": 0.6}]
