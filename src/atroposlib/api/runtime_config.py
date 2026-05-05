"""Runtime deployment configuration profiles for the FastAPI API."""

from __future__ import annotations

import os
from dataclasses import dataclass
from enum import Enum

from ..logging_utils import resolve_log_format


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


class RuntimeMode(str, Enum):
    """High-level runtime behavior mode."""

    DEV = "dev"
    PRODUCTION = "production"


@dataclass(frozen=True, slots=True)
class RuntimeDeploymentConfig:
    """Configuration resolved from profile defaults plus environment overrides."""

    profile: RuntimeProfile
    mode: RuntimeMode
    tier: RuntimeHardeningTier
    store_backend: str
    redis_url: str | None
    api_token: str | None
    allowed_origins: list[str]
    log_format: str | None
    tracing_enabled: bool
    tracing_exporter: str | None
    tracing_endpoint: str | None
    require_health_endpoints: bool
    allow_unsafe_localhost_defaults: bool

    def validate_for_runtime(self) -> None:
        """Validate profile-specific requirements before app startup."""

        if (
            self.tier is RuntimeHardeningTier.PRODUCTION_SAFE
            and self.mode is not RuntimeMode.PRODUCTION
        ):
            raise ValueError(
                "Production-safe hardening tier requires ATROPOS_RUNTIME_PROFILE=production "
                "so production mode is explicit"
            )

        if (
            self.mode is RuntimeMode.PRODUCTION
            and self.tier is not RuntimeHardeningTier.PRODUCTION_SAFE
        ):
            raise ValueError("Production profile requires ATROPOS_HARDENING_TIER=production-safe")

        if self.mode is RuntimeMode.PRODUCTION:
            if self.store_backend != "redis":
                raise ValueError(
                    "Production profile requires ATROPOS_STORE_BACKEND=redis for durable storage"
                )
            if not self.redis_url:
                raise ValueError(
                    "Production profile requires ATROPOS_REDIS_URL "
                    "and does not permit localhost defaults"
                )
            if not self.allow_unsafe_localhost_defaults and _looks_like_localhost_url(
                self.redis_url
            ):
                raise ValueError(
                    "Production profile disallows localhost Redis defaults; "
                    "set ATROPOS_REDIS_URL to a non-localhost address or "
                    "ATROPOS_ALLOW_UNSAFE_LOCALHOST_DEFAULTS=true to override"
                )
            if not self.api_token:
                raise ValueError("Production profile requires ATROPOS_API_TOKEN")
            if not self.allowed_origins:
                raise ValueError(
                    "Production profile requires ATROPOS_ALLOWED_ORIGINS with at least one origin"
                )
            if not self.allow_unsafe_localhost_defaults and any(
                _looks_like_localhost_origin(origin) for origin in self.allowed_origins
            ):
                raise ValueError(
                    "Production profile disallows localhost CORS origins unless "
                    "ATROPOS_ALLOW_UNSAFE_LOCALHOST_DEFAULTS=true"
                )
            if resolve_log_format(self.log_format) != "json":
                raise ValueError(
                    "Production profile requires structured logs; set ATROPOS_LOG_FORMAT=json"
                )
            if self.tracing_enabled and not self.tracing_endpoint:
                raise ValueError(
                    "Production profile requires ATROPOS_TRACING_ENDPOINT when tracing is enabled"
                )
            if not self.require_health_endpoints:
                raise ValueError("Production profile requires health endpoints to remain enabled")


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


def _looks_like_localhost_url(url: str) -> bool:
    normalized = url.strip().lower()
    return any(
        token in normalized for token in ("localhost", "127.0.0.1", "://0.0.0.0", "::1", "[::1]")
    )


def _looks_like_localhost_origin(origin: str) -> bool:
    normalized = origin.strip().lower()
    return any(
        token in normalized for token in ("localhost", "127.0.0.1", "://0.0.0.0", "::1", "[::1]")
    )


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
    mode = RuntimeMode.PRODUCTION if profile is RuntimeProfile.PRODUCTION else RuntimeMode.DEV
    allow_unsafe_localhost_defaults = _bool_env(
        "ATROPOS_ALLOW_UNSAFE_LOCALHOST_DEFAULTS",
        default=False,
    )
    require_health_endpoints = _bool_env(
        "ATROPOS_REQUIRE_HEALTH_ENDPOINTS",
        default=mode is RuntimeMode.PRODUCTION,
    )

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
        mode=mode,
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
        require_health_endpoints=require_health_endpoints,
        allow_unsafe_localhost_defaults=allow_unsafe_localhost_defaults,
    )
    config.validate_for_runtime()
    return config
