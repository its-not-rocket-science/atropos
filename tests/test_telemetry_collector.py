from __future__ import annotations

import urllib.error
from unittest.mock import Mock

import pytest

from atropos.telemetry_collector import (
    CollectionConfig,
    TelemetryRequestError,
    TGICollector,
    VLLMCollector,
)


class _FakeResponse:
    def __init__(self, body: str):
        self._body = body.encode("utf-8")

    def read(self) -> bytes:
        return self._body

    def __enter__(self) -> _FakeResponse:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:  # noqa: ANN001
        return None


def test_vllm_collect_sample_healthy_metrics(monkeypatch: pytest.MonkeyPatch) -> None:
    collector = VLLMCollector("http://localhost", CollectionConfig(max_tokens=20))
    metrics_text = "\n".join(
        [
            "vllm:gpu_cache_usage_perc 0.5",
            "vllm:time_per_output_token_seconds 0.01",
            "vllm:num_requests_running 3",
        ]
    )

    monkeypatch.setattr(
        "atropos.telemetry_collector.urllib.request.urlopen",
        lambda *args, **kwargs: _FakeResponse(metrics_text),
    )

    sample = collector._collect_sample()

    assert sample is not None
    assert sample.memory_gb == pytest.approx(40.0)
    assert sample.throughput_toks_per_sec == pytest.approx(100.0)
    assert (
        sample.raw_metrics["metric_quality"]["throughput_toks_per_sec"]
        == "estimated_from_metric"
    )


def test_request_helper_retries_on_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    collector = VLLMCollector("http://localhost")
    sleep_calls: list[float] = []

    sequence = [
        urllib.error.URLError("timeout"),
        urllib.error.URLError("timeout"),
        _FakeResponse('{"status": "healthy"}'),
    ]

    def _fake_urlopen(*args, **kwargs):  # noqa: ANN002, ANN003
        item = sequence.pop(0)
        if isinstance(item, Exception):
            raise item
        return item

    monkeypatch.setattr("atropos.telemetry_collector.time.sleep", lambda s: sleep_calls.append(s))
    monkeypatch.setattr("atropos.telemetry_collector.urllib.request.urlopen", _fake_urlopen)

    response = collector._make_request("/health")

    assert response["status"] == "healthy"
    assert sleep_calls == [0.25, 0.5]


def test_tgi_malformed_metrics_endpoint_returns_none(monkeypatch: pytest.MonkeyPatch) -> None:
    collector = TGICollector("http://localhost")
    monkeypatch.setattr(
        "atropos.telemetry_collector.urllib.request.urlopen",
        lambda *args, **kwargs: _FakeResponse("tgi_queue_size not-a-number"),
    )

    sample = collector._collect_sample()

    assert sample is None


def test_vllm_fallback_metadata_marks_measured_vs_estimated(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    collector = VLLMCollector("http://localhost", CollectionConfig(max_tokens=64))

    monkeypatch.setattr(
        collector,
        "_request_text",
        Mock(side_effect=TelemetryRequestError("metrics timeout")),
    )
    monkeypatch.setattr(
        collector,
        "_make_request",
        Mock(return_value={"usage": {"completion_tokens": 12}}),
    )

    sample = collector._collect_sample()

    assert sample is not None
    quality = sample.raw_metrics["metric_quality"]
    assert quality["memory_gb"] == "unavailable"
    assert quality["throughput_toks_per_sec"] == "measured_from_request"
    assert quality["latency_ms_per_request"] == "measured_from_request"
    assert quality["tokens_per_request"] == "measured_from_request"
