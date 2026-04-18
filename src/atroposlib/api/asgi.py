"""ASGI entrypoint for production runtime API deployments."""

from __future__ import annotations

import os

from .server import HardeningTier, build_runtime_app


def _coerce_tier(value: str | None) -> HardeningTier:
    if value is None:
        return HardeningTier.PRODUCTION_SAFE
    normalized = value.strip().lower()
    for tier in HardeningTier:
        if tier.value == normalized:
            return tier
    raise ValueError(
        "ATROPOS_HARDENING_TIER must be one of: " + ", ".join(tier.value for tier in HardeningTier)
    )


def _split_csv(value: str | None) -> list[str]:
    if value is None:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def _build_app_from_env() -> object:
    tier = _coerce_tier(os.getenv("ATROPOS_HARDENING_TIER"))
    api_token = os.getenv("ATROPOS_API_TOKEN")
    allowed_origins = _split_csv(os.getenv("ATROPOS_ALLOWED_ORIGINS"))
    return build_runtime_app(
        tier=tier,
        allowed_origins=allowed_origins,
        api_token=api_token,
        log_format=os.getenv("ATROPOS_LOG_FORMAT"),
    )


app = _build_app_from_env()
