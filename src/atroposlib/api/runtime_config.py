"""Runtime deployment configuration profiles for the FastAPI API."""

from __future__ import annotations

import os
from dataclasses import dataclass
from enum import Enum


class RuntimeProfile(str, Enum):
    """Named deployment profiles with explicit defaults."""

    LOCAL_DEV = "local-dev"
    CI = "ci"
    PRODUCTION = "production"


class RuntimeHardeningTier(str, Enum):
    """String tier values mapped by the runtime server."""

    RESEARCH_SAFE = "research-safe"
    INTERNAL_TEAM_SAFE = "internal-team-safe"
    PRODUCTION_SAFE = "production-safe"


@dataclass(frozen=True, slots=True)
class RuntimeDeploymentConfig:
    """Configuration resolved from profile defaults plus environment overrides."""

    profile: RuntimeProfile
    tier: RuntimeHardeningTier
    store_backend: str
    redis_url: str | None
    api_token: str | None
    allowed_origins: list[str]
    log_format: str | None
    tracing_enabled: bool
    tracing_exporter: str | None
    tracing_endpoint: str | None

    def validate_for_runtime(self) -> None:
        """Validate profile-specific requirements before app startup."""

        if self.profile is RuntimeProfile.PRODUCTION:
            if self.store_backend != "redis":
                raise ValueError(
                    "Production profile requires ATROPOS_STORE_BACKEND=redis for durable storage"
                )
            if not self.redis_url:
                raise ValueError(
                    "Production profile requires ATROPOS_REDIS_URL "
                    "and does not permit localhost defaults"
                )
            if not self.api_token:
                raise ValueError("Production profile requires ATROPOS_API_TOKEN")
            if not self.allowed_origins:
                raise ValueError(
                    "Production profile requires ATROPOS_ALLOWED_ORIGINS with at least one origin"
                )
            if self.tracing_enabled and not self.tracing_endpoint:
                raise ValueError(
                    "Production profile requires ATROPOS_TRACING_ENDPOINT when tracing is enabled"
                )


def _split_csv(value: str | None) -> list[str]:
    if value is None:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def _coerce_profile(value: str | None) -> RuntimeProfile:
    normalized = (value or RuntimeProfile.LOCAL_DEV.value).strip().lower()
    for profile in RuntimeProfile:
        if profile.value == normalized:
            return profile
    raise ValueError(
        "ATROPOS_RUNTIME_PROFILE must be one of: "
        + ", ".join(profile.value for profile in RuntimeProfile)
    )


def _coerce_tier(value: str | None, *, default: RuntimeHardeningTier) -> RuntimeHardeningTier:
    if value is None:
        return default
    normalized = value.strip().lower()
    for tier in RuntimeHardeningTier:
        if tier.value == normalized:
            return tier
    raise ValueError(
        "ATROPOS_HARDENING_TIER must be one of: "
        + ", ".join(tier.value for tier in RuntimeHardeningTier)
    )


def _bool_env(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def load_runtime_deployment_config_from_env() -> RuntimeDeploymentConfig:
    """Load runtime deployment config with profile defaults and env overrides."""

    profile = _coerce_profile(os.getenv("ATROPOS_RUNTIME_PROFILE"))

    profile_defaults: dict[RuntimeProfile, dict[str, str]] = {
        RuntimeProfile.LOCAL_DEV: {
            "tier": RuntimeHardeningTier.RESEARCH_SAFE.value,
            "store_backend": "memory",
        },
        RuntimeProfile.CI: {
            "tier": RuntimeHardeningTier.INTERNAL_TEAM_SAFE.value,
            "store_backend": "memory",
        },
        RuntimeProfile.PRODUCTION: {
            "tier": RuntimeHardeningTier.PRODUCTION_SAFE.value,
            "store_backend": "redis",
        },
    }
    defaults = profile_defaults[profile]

    tier = _coerce_tier(
        os.getenv("ATROPOS_HARDENING_TIER"),
        default=RuntimeHardeningTier(defaults["tier"]),
    )
    store_backend = os.getenv("ATROPOS_STORE_BACKEND", defaults["store_backend"]).strip().lower()
    redis_url = os.getenv("ATROPOS_REDIS_URL")
    api_token_default = "ci-test-token" if profile is RuntimeProfile.CI else None
    api_token = os.getenv("ATROPOS_API_TOKEN", api_token_default)
    allowed_origins = _split_csv(os.getenv("ATROPOS_ALLOWED_ORIGINS"))

    config = RuntimeDeploymentConfig(
        profile=profile,
        tier=tier,
        store_backend=store_backend,
        redis_url=redis_url,
        api_token=api_token,
        allowed_origins=allowed_origins,
        log_format=os.getenv("ATROPOS_LOG_FORMAT"),
        tracing_enabled=_bool_env(
            "ATROPOS_TRACING_ENABLED",
            default=profile is RuntimeProfile.PRODUCTION,
        ),
        tracing_exporter=os.getenv("ATROPOS_TRACING_EXPORTER", "otlp"),
        tracing_endpoint=os.getenv("ATROPOS_TRACING_ENDPOINT"),
    )
    config.validate_for_runtime()
    return config
