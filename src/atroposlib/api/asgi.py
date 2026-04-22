"""ASGI entrypoint for runtime API deployments."""

from __future__ import annotations

from fastapi import FastAPI

from .runtime_config import load_runtime_deployment_config_from_env
from .server import HardeningTier, build_runtime_app
from .storage import AtroposStore, InMemoryStore, RedisStore


def _assert_required_health_routes(app: FastAPI) -> None:
    route_paths = {getattr(route, "path", "") for route in app.routes}
    required_paths = {"/health", "/health/live", "/health/ready", "/health/dependencies"}
    missing = sorted(required_paths - route_paths)
    if missing:
        raise ValueError(f"Missing required health endpoints: {', '.join(missing)}")


def _build_app_from_env() -> object:
    config = load_runtime_deployment_config_from_env()
    store: AtroposStore
    if config.store_backend == "memory":
        store = InMemoryStore()
    elif config.store_backend == "redis":
        if not config.redis_url:
            raise ValueError("ATROPOS_REDIS_URL is required when ATROPOS_STORE_BACKEND=redis")
        store = RedisStore.from_url(config.redis_url)
    else:
        raise ValueError(
            f"ATROPOS_STORE_BACKEND must be one of: memory, redis; got {config.store_backend!r}"
        )
    app = build_runtime_app(
        tier=HardeningTier(config.tier.value),
        allowed_origins=config.allowed_origins,
        api_token=config.api_token,
        store=store,
        log_format=config.log_format,
        redis_url=config.redis_url,
    )
    if config.require_health_endpoints:
        _assert_required_health_routes(app)
    return app


app = _build_app_from_env()
