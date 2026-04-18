"""ASGI entrypoint for runtime API deployments."""

from __future__ import annotations

from .runtime_config import load_runtime_deployment_config_from_env
from .server import HardeningTier, build_runtime_app
from .storage import InMemoryStore


def _build_app_from_env() -> object:
    config = load_runtime_deployment_config_from_env()
    store = InMemoryStore() if config.store_backend == "memory" else None
    return build_runtime_app(
        tier=HardeningTier(config.tier.value),
        allowed_origins=config.allowed_origins,
        api_token=config.api_token,
        store=store,
        log_format=config.log_format,
        redis_url=config.redis_url,
    )


app = _build_app_from_env()
