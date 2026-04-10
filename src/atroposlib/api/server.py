"""FastAPI runtime server with pluggable storage and hardening tiers."""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Annotated, Any
from uuid import uuid4

from fastapi import Depends, FastAPI, Header, HTTPException, Request, Response, status
from fastapi.middleware.cors import CORSMiddleware

from atroposlib.observability import RuntimeMetrics, log_json_event

from .storage import InMemoryStore, RedisStore, RuntimeStatusRecord, RuntimeStore


class HardeningTier(str, Enum):
    """Deployment profiles that progressively tighten runtime controls."""

    RESEARCH_SAFE = "research-safe"
    INTERNAL_TEAM_SAFE = "internal-team-safe"
    PRODUCTION_SAFE = "production-safe"


RuntimeStatus = RuntimeStatusRecord


@dataclass(slots=True)
class AppRuntimeState:
    """Typed runtime state bound once on `app.state.runtime`."""

    store: RuntimeStore
    metrics: RuntimeMetrics


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


def get_runtime_state(request: Request) -> AppRuntimeState:
    """Typed accessor for runtime state.

    Raises a clear error if startup wiring is incomplete.
    """
    state = getattr(request.app.state, "runtime", None)
    if not isinstance(state, AppRuntimeState):
        raise RuntimeError("Application runtime state was not initialized")
    return state


def _build_auth_dependency(
    policy: RuntimePolicy,
    api_token: str | None,
):
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


def _default_store_for_tier(tier: HardeningTier) -> RuntimeStore:
    if tier is HardeningTier.PRODUCTION_SAFE:
        redis_url = os.getenv("ATROPOS_REDIS_URL", "redis://localhost:6379/0")
        return RedisStore.from_url(redis_url)
    return InMemoryStore()


def build_runtime_app(
    tier: HardeningTier = HardeningTier.RESEARCH_SAFE,
    *,
    allowed_origins: list[str] | None = None,
    api_token: str | None = None,
    store: RuntimeStore | None = None,
) -> FastAPI:
    """Build the FastAPI runtime app.

    The selected tier controls reset behavior, CORS openness, and whether
    write endpoints require an API token.
    """

    policy = _POLICY_BY_TIER[tier]
    app = FastAPI(title="Atropos Runtime API")
    runtime_store = store if store is not None else _default_store_for_tier(tier)
    runtime_metrics = RuntimeMetrics()
    app.state.runtime = AppRuntimeState(store=runtime_store, metrics=runtime_metrics)

    cors_origins = ["*"] if policy.allow_all_cors_origins else list(allowed_origins or [])
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins,
        allow_credentials=not policy.allow_all_cors_origins,
        allow_methods=["GET", "POST"],
        allow_headers=["Authorization", "Content-Type", "X-API-Token", "X-Idempotency-Key"],
    )

    write_access = _build_auth_dependency(policy, api_token)

    @app.middleware("http")
    async def observability_middleware(request: Request, call_next):
        runtime = get_runtime_state(request)
        request_started = datetime.now(tz=timezone.utc)
        start = time.perf_counter()
        env_name = request.headers.get("X-Env", "default")

        response = await call_next(request)

        latency = max(0.0, time.perf_counter() - start)
        status_code = str(response.status_code)
        path = request.url.path
        method = request.method

        runtime.metrics.api_requests_total.labels(
            method=method,
            path=path,
            status_code=status_code,
        ).inc()
        runtime.metrics.rollout_latency_seconds.labels(
            method=method,
            path=path,
            env=env_name,
        ).observe(latency)
        if response.status_code >= status.HTTP_400_BAD_REQUEST:
            runtime.metrics.api_errors_total.labels(
                method=method,
                path=path,
                status_code=status_code,
            ).inc()

        log_json_event(
            logger=logging.getLogger("atropos.runtime"),
            event="http_request",
            method=method,
            path=path,
            status_code=response.status_code,
            latency_seconds=latency,
            env=env_name,
            started_at=request_started.isoformat(),
        )

        return response


    def health(request: Request) -> dict[str, str]:
        backend_name = get_runtime_state(request).store.backend_name
        return {"status": "ok", "tier": tier.value, "store": backend_name}

    def enqueue(
        job: dict[str, Any],
        request: Request,
        x_idempotency_key: Annotated[str | None, Header(alias="X-Idempotency-Key")] = None,
    ) -> dict[str, Any]:
        runtime = get_runtime_state(request)
        now = datetime.now(tz=timezone.utc)
        env_name = request.headers.get("X-Env", "default")
        enqueue_result = runtime.store.enqueue_job(
            job_id=str(uuid4()),
            now=now,
            idempotency_key=x_idempotency_key,
        )

        _ = job  # payload is accepted for future processing pipeline.
        runtime.metrics.queue_size.labels(env=env_name).set(float(enqueue_result.queue_depth))
        return {
            "job_id": enqueue_result.job_id,
            "queue_depth": enqueue_result.queue_depth,
            "deduplicated": enqueue_result.deduplicated,
        }

    def get_job_status(job_id: str, request: Request) -> dict[str, Any]:
        runtime = get_runtime_state(request)
        status_record = runtime.store.get_job_status(job_id)
        if status_record is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Unknown job_id")
        return {
            "job_id": status_record.job_id,
            "state": status_record.state,
            "created_at": status_record.created_at,
            "updated_at": status_record.updated_at,
        }

    def reset_runtime(request: Request) -> dict[str, str]:
        if not policy.allow_reset_endpoint:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Not found")
        runtime = get_runtime_state(request)
        runtime.store.reset()
        return {"status": "reset"}

    def metrics_endpoint(request: Request) -> Response:
        metrics = get_runtime_state(request).metrics
        return Response(content=metrics.generate_latest(), media_type=metrics.content_type)

    app.add_api_route("/health", health, methods=["GET"])
    app.add_api_route("/metrics", metrics_endpoint, methods=["GET"])
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
        "/admin/reset",
        reset_runtime,
        methods=["POST"],
        dependencies=[Depends(write_access)],
    )

    return app
