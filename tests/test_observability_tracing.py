from __future__ import annotations

from atroposlib.observability import configure_tracing, tracing_config_from_env, tracing_span


def test_tracing_config_from_env_defaults_disabled(monkeypatch) -> None:
    monkeypatch.delenv("ATROPOS_TRACING_ENABLED", raising=False)
    monkeypatch.delenv("ATROPOS_TRACING_EXPORTER", raising=False)
    monkeypatch.delenv("ATROPOS_TRACING_ENDPOINT", raising=False)
    monkeypatch.delenv("ATROPOS_TRACING_SAMPLE_RATIO", raising=False)

    cfg = tracing_config_from_env()

    assert cfg.enabled is False
    assert cfg.exporter == "otlp"
    assert cfg.sample_ratio == 1.0


def test_configure_tracing_noop_when_disabled(monkeypatch) -> None:
    monkeypatch.setenv("ATROPOS_TRACING_ENABLED", "false")

    configured = configure_tracing()

    assert configured is False


def test_tracing_span_noop_context_manager() -> None:
    with tracing_span("test.span", attributes={"atropos.test": 1}) as span:
        _ = span
