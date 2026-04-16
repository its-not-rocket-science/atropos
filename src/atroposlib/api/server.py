"""FastAPI runtime server with pluggable storage and hardening tiers."""

from __future__ import annotations

import os
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from hashlib import sha256
from typing import Annotated, Any
from uuid import uuid4

from fastapi import Depends, FastAPI, Header, HTTPException, Request, Response, status
from fastapi.middleware.cors import CORSMiddleware

from ..observability import OBSERVABILITY, render_metrics
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
    return state


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
) -> FastAPI:
    """Build the FastAPI runtime app.

    The selected tier controls reset behavior, CORS openness, and whether
    write endpoints require an API token.
    """

    policy = _POLICY_BY_TIER[tier]
    app = FastAPI(title="Atropos Runtime API")
    runtime_store = store if store is not None else _default_store_for_tier(tier)

    @app.on_event("startup")
    async def _startup_store_binding() -> None:
        app.state.runtime_store = runtime_store

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
        OBSERVABILITY.observe_api_request(
            method=request.method,
            path=request.url.path,
            status=response.status_code,
            duration_seconds=duration,
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
        OBSERVABILITY.set_queue_depth(env=env_name, queue_depth=enqueue_result.queue_depth)

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
        result = store.ingest_scored_data(
            request_id=request_id,
            groups=[group],
        )
        env_name = group.environment_id
        queue_metrics = store.get_scored_queue_metrics(now=datetime.now(tz=timezone.utc))
        OBSERVABILITY.set_queue_depth(env=env_name, queue_depth=queue_metrics.depth)
        OBSERVABILITY.set_queue_oldest_age(
            env=env_name,
            oldest_age_seconds=queue_metrics.oldest_age_seconds,
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

        result = store.ingest_scored_data(
            request_id=request_id,
            groups=groups,
        )
        env_name = groups[0].environment_id if groups else "default"
        queue_metrics = store.get_scored_queue_metrics(now=datetime.now(tz=timezone.utc))
        OBSERVABILITY.set_queue_depth(env=env_name, queue_depth=queue_metrics.depth)
        OBSERVABILITY.set_queue_oldest_age(
            env=env_name,
            oldest_age_seconds=queue_metrics.oldest_age_seconds,
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
