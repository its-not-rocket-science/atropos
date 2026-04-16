"""Pluggable runtime storage backends for the FastAPI server."""

from __future__ import annotations

import json
from collections import deque
from collections.abc import MutableMapping
from dataclasses import dataclass
from datetime import datetime, timezone
from importlib.util import find_spec
from threading import Lock
from typing import Protocol


@dataclass(frozen=True, slots=True)
class RuntimeStatusRecord:
    """Status metadata for queued jobs."""

    job_id: str
    state: str
    created_at: datetime
    updated_at: datetime


@dataclass(frozen=True, slots=True)
class EnqueueResult:
    """Outcome of an enqueue request, including idempotency behavior."""

    job_id: str
    queue_depth: int
    deduplicated: bool


@dataclass(frozen=True, slots=True)
class IngestScoredDataResult:
    """Outcome of scored data ingestion request."""

    request_id: str
    accepted_count: int
    accepted_groups: int
    deduplicated: bool
    status: str
    failed_groups: int


@dataclass(frozen=True, slots=True)
class ScoredDataGroup:
    """A logical scored-data ingestion group."""

    environment_id: str
    records: list[dict[str, object]]
    group_id: str


class AtroposStore(Protocol):
    """Storage contract for runtime queue and status operations."""

    backend_name: str

    def enqueue_job(
        self,
        *,
        job_id: str,
        now: datetime,
        idempotency_key: str | None,
    ) -> EnqueueResult:
        """Persist a queued job and return queue depth with dedupe metadata."""

    def get_job_status(self, job_id: str) -> RuntimeStatusRecord | None:
        """Fetch status by job id."""

    def reset(self) -> None:
        """Clear runtime state."""

    def ingest_scored_data(
        self,
        *,
        request_id: str,
        groups: list[ScoredDataGroup],
    ) -> IngestScoredDataResult:
        """Ingest scored data groups with request-level deduplication."""

    def list_scored_data(self, *, environment_id: str, limit: int) -> list[dict[str, object]]:
        """List scored records for an environment."""


class RedisClient(Protocol):
    """Typed Redis client surface used by `RedisStore`."""

    def hset(self, name: str, mapping: MutableMapping[str, str]) -> int: ...

    def rpush(self, name: str, *values: str) -> int: ...

    def llen(self, name: str) -> int: ...
    def lrange(self, name: str, start: int, end: int) -> list[str]: ...

    def set(
        self,
        name: str,
        value: str,
        *,
        nx: bool | None = None,
        ex: int | None = None,
    ) -> bool | None: ...

    def get(self, name: str) -> str | None: ...

    def hgetall(self, name: str) -> MutableMapping[str, str]: ...

    def scan(
        self,
        cursor: int = 0,
        match: str | None = None,
        count: int | None = None,
    ) -> tuple[int, list[str]]: ...

    def delete(self, *names: str) -> int: ...


class InMemoryStore:
    """Single-process store that mirrors legacy app.state behavior."""

    backend_name = "memory"

    def __init__(self) -> None:
        self._queue: deque[str] = deque()
        self._status_by_job: dict[str, RuntimeStatusRecord] = {}
        self._job_by_idempotency_key: dict[str, str] = {}
        self._scored_by_environment: dict[str, list[dict[str, object]]] = {}
        self._accepted_group_keys: set[str] = set()
        self._request_status_by_id: dict[str, str] = {}
        self._request_accepted_groups: dict[str, set[str]] = {}
        self._lock = Lock()

    def enqueue_job(
        self,
        *,
        job_id: str,
        now: datetime,
        idempotency_key: str | None,
    ) -> EnqueueResult:
        with self._lock:
            if idempotency_key:
                existing = self._job_by_idempotency_key.get(idempotency_key)
                if existing is not None:
                    return EnqueueResult(
                        job_id=existing,
                        queue_depth=len(self._queue),
                        deduplicated=True,
                    )
                self._job_by_idempotency_key[idempotency_key] = job_id

            self._queue.append(job_id)
            self._status_by_job[job_id] = RuntimeStatusRecord(
                job_id=job_id,
                state="queued",
                created_at=now,
                updated_at=now,
            )
            return EnqueueResult(job_id=job_id, queue_depth=len(self._queue), deduplicated=False)

    def get_job_status(self, job_id: str) -> RuntimeStatusRecord | None:
        with self._lock:
            return self._status_by_job.get(job_id)

    def reset(self) -> None:
        with self._lock:
            self._queue.clear()
            self._status_by_job.clear()
            self._job_by_idempotency_key.clear()
            self._scored_by_environment.clear()
            self._accepted_group_keys.clear()
            self._request_status_by_id.clear()
            self._request_accepted_groups.clear()

    def ingest_scored_data(
        self,
        *,
        request_id: str,
        groups: list[ScoredDataGroup],
    ) -> IngestScoredDataResult:
        with self._lock:
            if self._request_status_by_id.get(request_id) == "completed":
                return IngestScoredDataResult(
                    request_id=request_id,
                    accepted_count=0,
                    accepted_groups=0,
                    deduplicated=True,
                    status="completed",
                    failed_groups=0,
                )

            self._request_status_by_id[request_id] = "processing"
            already_seen = self._request_accepted_groups.setdefault(request_id, set())
            accepted_count = 0
            accepted_groups = 0
            failed_groups = 0

            for group in groups:
                group_key = f"{group.environment_id}:{group.group_id}"
                if group_key in already_seen or group_key in self._accepted_group_keys:
                    continue
                if any(not isinstance(item, dict) for item in group.records):
                    failed_groups += 1
                    continue
                bucket = self._scored_by_environment.setdefault(group.environment_id, [])
                bucket.extend(group.records)
                self._accepted_group_keys.add(group_key)
                already_seen.add(group_key)
                accepted_count += len(group.records)
                accepted_groups += 1

            status = "completed" if failed_groups == 0 else "partial_failed"
            self._request_status_by_id[request_id] = status
            return IngestScoredDataResult(
                request_id=request_id,
                accepted_count=accepted_count,
                accepted_groups=accepted_groups,
                deduplicated=accepted_groups == 0 and failed_groups == 0 and bool(groups),
                status=status,
                failed_groups=failed_groups,
            )

    def list_scored_data(self, *, environment_id: str, limit: int) -> list[dict[str, object]]:
        with self._lock:
            records = self._scored_by_environment.get(environment_id, [])
            return [dict(item) for item in records[:limit]]


class RedisStore:
    """Redis-backed store for fault-tolerant, multi-instance runtime state."""

    backend_name = "redis"

    def __init__(
        self,
        *,
        redis_client: RedisClient,
        key_prefix: str = "atropos:runtime",
        idempotency_ttl_seconds: int = 24 * 60 * 60,
    ) -> None:
        self._redis = redis_client
        self._key_prefix = key_prefix
        self._idempotency_ttl_seconds = idempotency_ttl_seconds

    @classmethod
    def from_url(
        cls,
        redis_url: str,
        *,
        key_prefix: str = "atropos:runtime",
        idempotency_ttl_seconds: int = 24 * 60 * 60,
    ) -> RedisStore:
        if find_spec("redis") is None:  # pragma: no cover - exercised in runtime environments
            raise RuntimeError(
                "RedisStore requires the `redis` package. Install with `pip install redis`."
            )

        import redis

        client = redis.Redis.from_url(redis_url, decode_responses=True)
        return cls(
            redis_client=client,
            key_prefix=key_prefix,
            idempotency_ttl_seconds=idempotency_ttl_seconds,
        )

    def _queue_key(self) -> str:
        return f"{self._key_prefix}:queue"

    def _job_key(self, job_id: str) -> str:
        return f"{self._key_prefix}:job:{job_id}"

    def _idempotency_key(self, idempotency_key: str) -> str:
        return f"{self._key_prefix}:idempotency:{idempotency_key}"

    def _scored_request_key(self, request_id: str) -> str:
        return f"{self._key_prefix}:scored:request:{request_id}"

    def _scored_request_status_key(self, request_id: str) -> str:
        return f"{self._key_prefix}:scored:request_status:{request_id}"

    def _scored_request_groups_key(self, request_id: str) -> str:
        return f"{self._key_prefix}:scored:request_groups:{request_id}"

    def _scored_group_key(self, environment_id: str, group_id: str) -> str:
        return f"{self._key_prefix}:scored:group:{environment_id}:{group_id}"

    def _scored_list_key(self, environment_id: str) -> str:
        return f"{self._key_prefix}:scored:environment:{environment_id}"

    def enqueue_job(
        self,
        *,
        job_id: str,
        now: datetime,
        idempotency_key: str | None,
    ) -> EnqueueResult:
        created = now.astimezone(timezone.utc).isoformat()

        if idempotency_key is None:
            self._redis.hset(
                self._job_key(job_id),
                mapping={
                    "job_id": job_id,
                    "state": "queued",
                    "created_at": created,
                    "updated_at": created,
                },
            )
            self._redis.rpush(self._queue_key(), job_id)
            return EnqueueResult(
                job_id=job_id,
                queue_depth=int(self._redis.llen(self._queue_key())),
                deduplicated=False,
            )

        idem_key = self._idempotency_key(idempotency_key)
        was_created = bool(
            self._redis.set(
                idem_key,
                job_id,
                nx=True,
                ex=self._idempotency_ttl_seconds,
            )
        )
        if not was_created:
            existing_job_id = self._redis.get(idem_key)
            if existing_job_id is None:
                return self.enqueue_job(job_id=job_id, now=now, idempotency_key=idempotency_key)
            return EnqueueResult(
                job_id=str(existing_job_id),
                queue_depth=int(self._redis.llen(self._queue_key())),
                deduplicated=True,
            )

        self._redis.hset(
            self._job_key(job_id),
            mapping={
                "job_id": job_id,
                "state": "queued",
                "created_at": created,
                "updated_at": created,
            },
        )
        self._redis.rpush(self._queue_key(), job_id)
        return EnqueueResult(
            job_id=job_id,
            queue_depth=int(self._redis.llen(self._queue_key())),
            deduplicated=False,
        )

    def get_job_status(self, job_id: str) -> RuntimeStatusRecord | None:
        payload: MutableMapping[str, str] = self._redis.hgetall(self._job_key(job_id))
        if not payload:
            return None
        return RuntimeStatusRecord(
            job_id=payload["job_id"],
            state=payload["state"],
            created_at=datetime.fromisoformat(payload["created_at"]),
            updated_at=datetime.fromisoformat(payload["updated_at"]),
        )

    def reset(self) -> None:
        pattern = f"{self._key_prefix}:*"
        cursor = 0
        while True:
            cursor, keys = self._redis.scan(cursor=cursor, match=pattern, count=100)
            if keys:
                self._redis.delete(*keys)
            if cursor == 0:
                break

    def ingest_scored_data(
        self,
        *,
        request_id: str,
        groups: list[ScoredDataGroup],
    ) -> IngestScoredDataResult:
        status_key = self._scored_request_status_key(request_id)
        existing_status = self._redis.get(status_key)
        if existing_status == "completed":
            return IngestScoredDataResult(
                request_id=request_id,
                accepted_count=0,
                accepted_groups=0,
                deduplicated=True,
                status="completed",
                failed_groups=0,
            )

        self._redis.set(status_key, "processing", ex=self._idempotency_ttl_seconds)
        accepted_count = 0
        accepted_groups = 0
        failed_groups = 0
        request_groups_key = self._scored_request_groups_key(request_id)
        for group in groups:
            if any(not isinstance(item, dict) for item in group.records):
                failed_groups += 1
                continue
            group_key = self._scored_group_key(group.environment_id, group.group_id)
            claimed = bool(
                self._redis.set(
                    group_key,
                    request_id,
                    nx=True,
                    ex=self._idempotency_ttl_seconds,
                )
            )
            if not claimed:
                continue
            self._redis.set(
                f"{request_groups_key}:{group.group_id}",
                "1",
                ex=self._idempotency_ttl_seconds,
            )
            list_key = self._scored_list_key(group.environment_id)
            for record in group.records:
                self._redis.rpush(list_key, json.dumps(record, sort_keys=True))
            accepted_count += len(group.records)
            accepted_groups += 1

        final_status = "completed" if failed_groups == 0 else "partial_failed"
        self._redis.set(status_key, final_status, ex=self._idempotency_ttl_seconds)
        return IngestScoredDataResult(
            request_id=request_id,
            accepted_count=accepted_count,
            accepted_groups=accepted_groups,
            deduplicated=accepted_groups == 0 and failed_groups == 0 and bool(groups),
            status=final_status,
            failed_groups=failed_groups,
        )

    def list_scored_data(self, *, environment_id: str, limit: int) -> list[dict[str, object]]:
        if limit <= 0:
            return []
        encoded = self._redis.lrange(self._scored_list_key(environment_id), 0, limit - 1)
        return [json.loads(item) for item in encoded]


class PostgresStore:
    """Placeholder for a future Postgres-backed implementation."""

    backend_name = "postgres"

    def __init__(self, dsn: str) -> None:
        self.dsn = dsn

    def enqueue_job(
        self,
        *,
        job_id: str,
        now: datetime,
        idempotency_key: str | None,
    ) -> EnqueueResult:
        raise NotImplementedError("PostgresStore is not implemented yet")

    def get_job_status(self, job_id: str) -> RuntimeStatusRecord | None:
        raise NotImplementedError("PostgresStore is not implemented yet")

    def reset(self) -> None:
        raise NotImplementedError("PostgresStore is not implemented yet")

    def ingest_scored_data(
        self,
        *,
        request_id: str,
        groups: list[ScoredDataGroup],
    ) -> IngestScoredDataResult:
        raise NotImplementedError("PostgresStore is not implemented yet")

    def list_scored_data(self, *, environment_id: str, limit: int) -> list[dict[str, object]]:
        raise NotImplementedError("PostgresStore is not implemented yet")


# Backwards compatibility alias.
RuntimeStore = AtroposStore
