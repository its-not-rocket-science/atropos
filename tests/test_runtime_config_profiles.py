from __future__ import annotations

import pytest

from atroposlib.api.runtime_config import (
    RuntimeProfile,
    load_runtime_deployment_config_from_env,
)


@pytest.fixture(autouse=True)
def _clear_runtime_env(monkeypatch: pytest.MonkeyPatch) -> None:
    keys = [
        "ATROPOS_RUNTIME_PROFILE",
        "ATROPOS_HARDENING_TIER",
        "ATROPOS_STORE_BACKEND",
        "ATROPOS_REDIS_URL",
        "ATROPOS_API_TOKEN",
        "ATROPOS_ALLOWED_ORIGINS",
        "ATROPOS_LOG_FORMAT",
        "ATROPOS_TRACING_ENABLED",
        "ATROPOS_TRACING_EXPORTER",
        "ATROPOS_TRACING_ENDPOINT",
        "ATROPOS_REQUIRE_HEALTH_ENDPOINTS",
        "ATROPOS_ALLOW_UNSAFE_LOCALHOST_DEFAULTS",
    ]
    for key in keys:
        monkeypatch.delenv(key, raising=False)


def test_local_dev_profile_defaults_to_memory_and_open_defaults() -> None:
    config = load_runtime_deployment_config_from_env()

    assert config.profile is RuntimeProfile.LOCAL_DEV
    assert config.mode.value == "dev"
    assert config.store_backend == "memory"
    assert config.redis_url is None
    assert config.api_token is None
    assert config.allowed_origins == []
    assert config.tracing_enabled is False
    assert config.require_health_endpoints is False


def test_ci_profile_sets_memory_and_default_ci_token(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ATROPOS_RUNTIME_PROFILE", "ci")

    config = load_runtime_deployment_config_from_env()

    assert config.profile is RuntimeProfile.CI
    assert config.store_backend == "memory"
    assert config.api_token == "ci-test-token"


def test_production_profile_requires_explicit_durable_and_observability_settings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ATROPOS_RUNTIME_PROFILE", "production")

    with pytest.raises(ValueError, match="ATROPOS_REDIS_URL"):
        load_runtime_deployment_config_from_env()

    monkeypatch.setenv("ATROPOS_REDIS_URL", "redis://redis:6379/0")
    with pytest.raises(ValueError, match="ATROPOS_API_TOKEN"):
        load_runtime_deployment_config_from_env()

    monkeypatch.setenv("ATROPOS_API_TOKEN", "prod-token")
    with pytest.raises(ValueError, match="ATROPOS_ALLOWED_ORIGINS"):
        load_runtime_deployment_config_from_env()

    monkeypatch.setenv("ATROPOS_ALLOWED_ORIGINS", "https://api.example.com")
    monkeypatch.setenv("ATROPOS_TRACING_ENABLED", "true")
    with pytest.raises(ValueError, match="ATROPOS_TRACING_ENDPOINT"):
        load_runtime_deployment_config_from_env()


def test_production_profile_rejects_unsafe_localhost_defaults(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ATROPOS_RUNTIME_PROFILE", "production")
    monkeypatch.setenv("ATROPOS_REDIS_URL", "redis://localhost:6379/0")
    monkeypatch.setenv("ATROPOS_API_TOKEN", "prod-token")
    monkeypatch.setenv("ATROPOS_ALLOWED_ORIGINS", "https://api.example.com")
    monkeypatch.setenv("ATROPOS_TRACING_ENABLED", "false")
    monkeypatch.setenv("ATROPOS_LOG_FORMAT", "json")

    with pytest.raises(ValueError, match="localhost Redis defaults"):
        load_runtime_deployment_config_from_env()

    monkeypatch.setenv("ATROPOS_REDIS_URL", "redis://redis.internal:6379/0")
    monkeypatch.setenv("ATROPOS_ALLOWED_ORIGINS", "http://localhost:3000")
    with pytest.raises(ValueError, match="localhost CORS origins"):
        load_runtime_deployment_config_from_env()

    monkeypatch.setenv("ATROPOS_ALLOW_UNSAFE_LOCALHOST_DEFAULTS", "true")
    config = load_runtime_deployment_config_from_env()
    assert config.allow_unsafe_localhost_defaults is True


def test_production_profile_requires_structured_logs_and_health_endpoints(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ATROPOS_RUNTIME_PROFILE", "production")
    monkeypatch.setenv("ATROPOS_REDIS_URL", "redis://redis.internal:6379/0")
    monkeypatch.setenv("ATROPOS_API_TOKEN", "prod-token")
    monkeypatch.setenv("ATROPOS_ALLOWED_ORIGINS", "https://api.example.com")
    monkeypatch.setenv("ATROPOS_TRACING_ENABLED", "false")
    monkeypatch.setenv("ATROPOS_LOG_FORMAT", "pretty")

    with pytest.raises(ValueError, match="structured logs"):
        load_runtime_deployment_config_from_env()

    monkeypatch.setenv("ATROPOS_LOG_FORMAT", "json")
    monkeypatch.setenv("ATROPOS_REQUIRE_HEALTH_ENDPOINTS", "false")
    with pytest.raises(ValueError, match="health endpoints"):
        load_runtime_deployment_config_from_env()


def test_production_profile_allows_complete_configuration(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ATROPOS_RUNTIME_PROFILE", "production")
    monkeypatch.setenv("ATROPOS_REDIS_URL", "redis://redis:6379/0")
    monkeypatch.setenv("ATROPOS_API_TOKEN", "prod-token")
    monkeypatch.setenv("ATROPOS_ALLOWED_ORIGINS", "https://api.example.com")
    monkeypatch.setenv("ATROPOS_TRACING_ENABLED", "true")
    monkeypatch.setenv("ATROPOS_TRACING_ENDPOINT", "http://otel-collector:4318/v1/traces")
    monkeypatch.setenv("ATROPOS_LOG_FORMAT", "json")

    config = load_runtime_deployment_config_from_env()

    assert config.profile is RuntimeProfile.PRODUCTION
    assert config.mode.value == "production"
    assert config.store_backend == "redis"
    assert config.redis_url == "redis://redis:6379/0"
    assert config.api_token == "prod-token"
    assert config.allowed_origins == ["https://api.example.com"]
    assert config.tracing_enabled is True
    assert config.tracing_endpoint == "http://otel-collector:4318/v1/traces"
    assert config.require_health_endpoints is True
