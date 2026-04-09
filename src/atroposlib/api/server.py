"""FastAPI runtime server with typed state and configurable hardening tiers.

The app intentionally keeps the runtime small but strongly typed so
`app.state` usage is centralized through `AppRuntimeState`.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from threading import Lock
from typing import Annotated, Any
from uuid import uuid4

from fastapi import Depends, FastAPI, Header, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field


class HardeningTier(StrEnum):
    """Deployment profiles that progressively tighten runtime controls."""

    RESEARCH_SAFE = "research-safe"
    INTERNAL_TEAM_SAFE = "internal-team-safe"
    PRODUCTION_SAFE = "production-safe"


class RuntimeStatus(BaseModel):
    """Status metadata for queued jobs."""

    job_id: str
    state: str
    created_at: datetime
    updated_at: datetime


class JobRequest(BaseModel):
    """Minimal job payload for enqueue endpoints."""

    payload: dict[str, Any] = Field(default_factory=dict)


class EnqueueResponse(BaseModel):
    """Response emitted when a job enters the queue."""

    job_id: str
    queue_depth: int


@dataclass(slots=True)
class AppRuntimeState:
    """Typed runtime state bound once on `app.state.runtime`."""

    queue: deque[str]
    status_by_job: dict[str, RuntimeStatus]
    lock: Lock


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


def _build_auth_dependency(policy: RuntimePolicy, api_token: str | None):
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


def build_runtime_app(
    tier: HardeningTier = HardeningTier.RESEARCH_SAFE,
    *,
    allowed_origins: list[str] | None = None,
    api_token: str | None = None,
) -> FastAPI:
    """Build the FastAPI runtime app.

    The selected tier controls reset behavior, CORS openness, and whether
    write endpoints require an API token.
    """

    policy = _POLICY_BY_TIER[tier]
    app = FastAPI(title="Atropos Runtime API")
    app.state.runtime = AppRuntimeState(queue=deque(), status_by_job={}, lock=Lock())

    cors_origins = ["*"] if policy.allow_all_cors_origins else list(allowed_origins or [])
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins,
        allow_credentials=not policy.allow_all_cors_origins,
        allow_methods=["GET", "POST"],
        allow_headers=["Authorization", "Content-Type", "X-API-Token"],
    )

    write_access = _build_auth_dependency(policy, api_token)

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok", "tier": tier.value}

    @app.post("/jobs", response_model=EnqueueResponse, dependencies=[Depends(write_access)])
    def enqueue(job: JobRequest, request: Request) -> EnqueueResponse:
        runtime = get_runtime_state(request)
        now = datetime.now(tz=UTC)
        job_id = str(uuid4())

        with runtime.lock:
            runtime.queue.append(job_id)
            runtime.status_by_job[job_id] = RuntimeStatus(
                job_id=job_id,
                state="queued",
                created_at=now,
                updated_at=now,
            )
            queue_depth = len(runtime.queue)

        _ = job  # payload is accepted for future processing pipeline.
        return EnqueueResponse(job_id=job_id, queue_depth=queue_depth)

    @app.get("/jobs/{job_id}", response_model=RuntimeStatus)
    def get_job_status(job_id: str, request: Request) -> RuntimeStatus:
        runtime = get_runtime_state(request)
        with runtime.lock:
            status_record = runtime.status_by_job.get(job_id)
        if status_record is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Unknown job_id")
        return status_record

    @app.post("/admin/reset", dependencies=[Depends(write_access)])
    def reset_runtime(request: Request) -> dict[str, str]:
        if not policy.allow_reset_endpoint:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Not found")
        runtime = get_runtime_state(request)
        with runtime.lock:
            runtime.queue.clear()
            runtime.status_by_job.clear()
        return {"status": "reset"}

    return app
