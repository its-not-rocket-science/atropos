"""FastAPI runtime server with pluggable storage and hardening tiers."""

from __future__ import annotations

import logging
import os
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from hashlib import sha256
from typing import Annotated, Any, cast
from uuid import uuid4

from fastapi import Depends, FastAPI, Header, HTTPException, Request, Response, status
from fastapi.middleware.cors import CORSMiddleware

from ..logging_utils import build_log_context, configure_logging
from ..observability import OBSERVABILITY, configure_tracing, render_metrics, tracing_span
from .storage import AtroposStore, InMemoryStore, RedisStore, RuntimeStatusRecord, ScoredDataGroup


class HardeningTier(str, Enum):
    """Deployment profiles that progressively tighten runtime controls."""

    RESEARCH_SAFE = "research-safe"
    INTERNAL_TEAM_SAFE = "internal-team-safe"
    PRODUCTION_SAFE = "production-safe"


RuntimeStatus = RuntimeStatusRecord


@dataclass(frozen=True, slots=True)
class RuntimePolicy:
    """Runtime controls derived from the selected hardening tier."""

    allow_reset_endpoint: bool
    allow_all_cors_origins: bool
    require_auth: bool


_POLICY_BY_TIER: dict[HardeningTier, RuntimePolicy] = {
    HardeningTier.RESEARCH_SAFE: RuntimePolicy(
        allow_reset_endpoint=True,
        allow_all_cors_origins=True,
        require_auth=False,
    ),
    HardeningTier.INTERNAL_TEAM_SAFE: RuntimePolicy(
        allow_reset_endpoint=False,
        allow_all_cors_origins=False,
        require_auth=True,
    ),
    HardeningTier.PRODUCTION_SAFE: RuntimePolicy(
        allow_reset_endpoint=False,
        allow_all_cors_origins=False,
        require_auth=True,
    ),
}


def get_runtime_state(request: Request) -> AtroposStore:
    """Compatibility accessor for the configured runtime store."""

    state = getattr(request.app.state, "runtime_store", None)
    if state is None:
        raise RuntimeError("Application runtime store was not initialized")
    return cast(AtroposStore, state)


def _build_auth_dependency(
    policy: RuntimePolicy,
    api_token: str | None,
) -> Callable[..., None]:
    if not policy.require_auth:

        def _allow_anonymous() -> None:
            return None

        return _allow_anonymous

    def _verify_token(
        x_api_token: Annotated[str | None, Header(alias="X-API-Token")] = None,
    ) -> None:
        if not api_token:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Authentication is required but no API token is configured",
            )
        if x_api_token != api_token:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid API token",
            )

    return _verify_token


def _default_store_for_tier(tier: HardeningTier) -> AtroposStore:
    if tier is HardeningTier.PRODUCTION_SAFE:
        redis_url = os.getenv("ATROPOS_REDIS_URL", "redis://localhost:6379/0")
        return RedisStore.from_url(redis_url)
    return InMemoryStore()


def build_runtime_app(
    tier: HardeningTier = HardeningTier.RESEARCH_SAFE,
    *,
    allowed_origins: list[str] | None = None,
    api_token: str | None = None,
    store: AtroposStore | None = None,
    log_format: str | None = None,
) -> FastAPI:
    """Build the FastAPI runtime app.

    The selected tier controls reset behavior, CORS openness, and whether
    write endpoints require an API token.
    """

    policy = _POLICY_BY_TIER[tier]
    app = FastAPI(title="Atropos Runtime API")
    runtime_store = store if store is not None else _default_store_for_tier(tier)
    api_logger = configure_logging(
        logger_name="atroposlib.api.server",
        level=logging.INFO,
        log_format=log_format,
    )

    async def _startup_store_binding() -> None:
        app.state.runtime_store = runtime_store
        configure_tracing()

    app.add_event_handler("startup", _startup_store_binding)

    async def _get_store() -> AtroposStore:
        return runtime_store

    cors_origins = ["*"] if policy.allow_all_cors_origins else list(allowed_origins or [])
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins,
        allow_credentials=not policy.allow_all_cors_origins,
        allow_methods=["GET", "POST"],
        allow_headers=[
            "Authorization",
            "Content-Type",
            "X-API-Token",
            "X-Idempotency-Key",
            "X-Request-ID",
        ],
    )

    write_access = _build_auth_dependency(policy, api_token)

    async def _metrics_middleware(
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        started_at = datetime.now(tz=timezone.utc)
        response = await call_next(request)
        duration = (datetime.now(tz=timezone.utc) - started_at).total_seconds()
        request_id = request.headers.get("X-Request-ID") or request.headers.get("X-Idempotency-Key")
        OBSERVABILITY.observe_api_request(
            method=request.method,
            path=request.url.path,
            status=response.status_code,
            duration_seconds=duration,
        )
        api_logger.info(
            "request_completed",
            extra=build_log_context(
                request_id=request_id,
                endpoint=request.url.path,
                env_id=request.headers.get("X-Env", "default"),
                duration_seconds=duration,
                status_code=response.status_code,
                method=request.method,
            ),
        )
        return response

    app.middleware("http")(_metrics_middleware)

    def health(store: Annotated[AtroposStore, Depends(_get_store)]) -> dict[str, str]:
        backend_name = store.backend_name
        return {"status": "ok", "tier": tier.value, "store": backend_name}

    def enqueue(
        job: dict[str, Any],
        request: Request,
        store: Annotated[AtroposStore, Depends(_get_store)],
        x_idempotency_key: Annotated[str | None, Header(alias="X-Idempotency-Key")] = None,
    ) -> dict[str, Any]:
        now = datetime.now(tz=timezone.utc)
        enqueue_result = store.enqueue_job(
            job_id=str(uuid4()),
            now=now,
            idempotency_key=x_idempotency_key,
        )
        env_name = request.headers.get("X-Env", "default")
        request_id = request.headers.get("X-Request-ID") or x_idempotency_key
        OBSERVABILITY.set_queue_depth(env=env_name, queue_depth=enqueue_result.queue_depth)

        api_logger.info(
            "job_enqueued",
            extra=build_log_context(
                env_id=env_name,
                request_id=request_id,
                batch_id=enqueue_result.job_id,
                endpoint="/jobs",
                deduplicated=enqueue_result.deduplicated,
                queue_depth=enqueue_result.queue_depth,
            ),
        )

        _ = job  # payload is accepted for future processing pipeline.
        return {
            "job_id": enqueue_result.job_id,
            "queue_depth": enqueue_result.queue_depth,
            "deduplicated": enqueue_result.deduplicated,
        }

    def get_job_status(
        job_id: str,
        store: Annotated[AtroposStore, Depends(_get_store)],
    ) -> dict[str, Any]:
        status_record = store.get_job_status(job_id)
        if status_record is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Unknown job_id")
        return {
            "job_id": status_record.job_id,
            "state": status_record.state,
            "created_at": status_record.created_at,
            "updated_at": status_record.updated_at,
        }

    def ingest_scored_data(
        payload: dict[str, Any],
        store: Annotated[AtroposStore, Depends(_get_store)],
        x_request_id: Annotated[str | None, Header(alias="X-Request-ID")] = None,
        x_idempotency_key: Annotated[str | None, Header(alias="X-Idempotency-Key")] = None,
    ) -> dict[str, Any]:
        request_id = x_request_id or x_idempotency_key
        if not request_id:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="X-Request-ID or X-Idempotency-Key header is required",
            )
        group = _coerce_scored_data_group(payload)
        with tracing_span(
            "runtime.ingest_scored_data",
            attributes={
                "atropos.env": group.environment_id,
                "atropos.request_id": request_id,
                "atropos.group_count": 1,
                "atropos.record_count": len(group.records),
            },
        ):
            result = store.ingest_scored_data(
                request_id=request_id,
                groups=[group],
            )
        env_name = group.environment_id
        _record_ingestion_metrics(
            store=store,
            env_name=env_name,
            endpoint="/scored_data",
            result=result,
            groups=[group],
        )
        api_logger.info(
            "scored_data_ingested",
            extra=build_log_context(
                env_id=env_name,
                request_id=request_id,
                batch_id=group.group_id,
                endpoint="/scored_data",
                accepted_count=result.accepted_count,
                deduplicated=result.deduplicated,
            ),
        )
        return {
            "request_id": result.request_id,
            "accepted_count": result.accepted_count,
            "accepted_groups": result.accepted_groups,
            "failed_groups": result.failed_groups,
            "status": result.status,
            "deduplicated": result.deduplicated,
        }

    def ingest_scored_data_list(
        payload: dict[str, Any],
        store: Annotated[AtroposStore, Depends(_get_store)],
        x_request_id: Annotated[str | None, Header(alias="X-Request-ID")] = None,
        x_idempotency_key: Annotated[str | None, Header(alias="X-Idempotency-Key")] = None,
    ) -> dict[str, Any]:
        request_id = x_request_id or x_idempotency_key
        if not request_id:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="X-Request-ID or X-Idempotency-Key header is required",
            )
        raw_groups = payload.get("groups")
        if not isinstance(raw_groups, list):
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="groups must be a list",
            )
        groups: list[ScoredDataGroup] = []
        for group_payload in raw_groups:
            if not isinstance(group_payload, dict):
                raise HTTPException(
                    status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                    detail="each group must be an object",
                )
            groups.append(_coerce_scored_data_group(group_payload))

        with tracing_span(
            "runtime.ingest_scored_data",
            attributes={
                "atropos.env": groups[0].environment_id if groups else "default",
                "atropos.request_id": request_id,
                "atropos.group_count": len(groups),
                "atropos.record_count": sum(len(group.records) for group in groups),
            },
        ):
            result = store.ingest_scored_data(
                request_id=request_id,
                groups=groups,
            )
        env_name = groups[0].environment_id if groups else "default"
        _record_ingestion_metrics(
            store=store,
            env_name=env_name,
            endpoint="/scored_data_list",
            result=result,
            groups=groups,
        )
        api_logger.info(
            "scored_data_list_ingested",
            extra=build_log_context(
                env_id=env_name,
                request_id=request_id,
                batch_id=(groups[0].group_id if groups else None),
                endpoint="/scored_data_list",
                accepted_count=result.accepted_count,
                deduplicated=result.deduplicated,
                group_count=len(groups),
            ),
        )
        return {
            "request_id": result.request_id,
            "accepted_count": result.accepted_count,
            "accepted_groups": result.accepted_groups,
            "failed_groups": result.failed_groups,
            "status": result.status,
            "deduplicated": result.deduplicated,
        }

    def scored_data_list(
        environment_id: str,
        store: Annotated[AtroposStore, Depends(_get_store)],
        limit: int = 100,
    ) -> dict[str, Any]:
        bounded_limit = max(0, min(limit, 1000))
        with tracing_span(
            "runtime.trainer_batch_fetch",
            attributes={
                "atropos.env": environment_id,
                "atropos.limit": bounded_limit,
            },
        ):
            records = store.list_scored_data(environment_id=environment_id, limit=bounded_limit)
        return {
            "environment_id": environment_id,
            "count": len(records),
            "records": records,
        }

    def metrics() -> Response:
        payload, content_type = render_metrics()
        return Response(content=payload, media_type=content_type)

    def reset_runtime(store: Annotated[AtroposStore, Depends(_get_store)]) -> dict[str, str]:
        if not policy.allow_reset_endpoint:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Not found")
        store.reset()
        api_logger.warning(
            "runtime_reset",
            extra=build_log_context(endpoint="/admin/reset"),
        )
        return {"status": "reset"}

    app.add_api_route("/health", health, methods=["GET"])
    app.add_api_route("/metrics", metrics, methods=["GET"])
    app.add_api_route(
        "/jobs",
        enqueue,
        methods=["POST"],
        dependencies=[Depends(write_access)],
    )
    app.add_api_route(
        "/jobs/{job_id}",
        get_job_status,
        methods=["GET"],
    )
    app.add_api_route(
        "/scored_data",
        ingest_scored_data,
        methods=["POST"],
        dependencies=[Depends(write_access)],
    )
    app.add_api_route(
        "/scored_data_list",
        scored_data_list,
        methods=["GET"],
        dependencies=[Depends(write_access)],
    )
    app.add_api_route(
        "/scored_data_list",
        ingest_scored_data_list,
        methods=["POST"],
        dependencies=[Depends(write_access)],
    )
    app.add_api_route(
        "/admin/reset",
        reset_runtime,
        methods=["POST"],
        dependencies=[Depends(write_access)],
    )

    return app


def _coerce_scored_data_group(payload: dict[str, Any]) -> ScoredDataGroup:
    environment_id = str(payload.get("environment_id", "default"))
    raw_records = payload.get("records")
    if not isinstance(raw_records, list):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="records must be a list",
        )
    records: list[dict[str, object]] = []
    for item in raw_records:
        if not isinstance(item, dict):
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="each record must be an object",
            )
        records.append(item)
    raw_group_id = payload.get("group_id")
    if raw_group_id is not None:
        group_id = str(raw_group_id)
    else:
        digest = sha256()
        digest.update(environment_id.encode("utf-8"))
        digest.update(b":")
        digest.update(str(sorted(records, key=lambda r: str(sorted(r.items())))).encode("utf-8"))
        group_id = digest.hexdigest()
    return ScoredDataGroup(environment_id=environment_id, records=records, group_id=group_id)


def _record_ingestion_metrics(
    *,
    store: AtroposStore,
    env_name: str,
    endpoint: str,
    result: Any,
    groups: list[ScoredDataGroup],
) -> None:
    queue_metrics = store.get_scored_queue_metrics(now=datetime.now(tz=timezone.utc))
    OBSERVABILITY.set_queue_depth(env=env_name, queue_depth=queue_metrics.depth)
    OBSERVABILITY.set_queue_oldest_age(
        env=env_name,
        oldest_age_seconds=queue_metrics.oldest_age_seconds,
    )
    OBSERVABILITY.set_buffered_groups(env=env_name, group_count=result.accepted_groups)
    OBSERVABILITY.observe_ingestion(env=env_name, accepted_count=result.accepted_count)
    if result.deduplicated:
        OBSERVABILITY.observe_duplicate_rejection(env=env_name, endpoint=endpoint)
    for group in groups:
        group_status = store.get_scored_group_status(
            environment_id=group.environment_id,
            group_id=group.group_id,
        )
        if (
            group_status is None
            or group_status.buffered_at is None
            or group_status.batched_at is None
        ):
            continue
        latency_seconds = (
            group_status.batched_at.astimezone(timezone.utc)
            - group_status.buffered_at.astimezone(timezone.utc)
        ).total_seconds()
        OBSERVABILITY.observe_batch_formation_latency(
            env=group.environment_id,
            latency_seconds=latency_seconds,
        )
