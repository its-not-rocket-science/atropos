from __future__ import annotations

from importlib.util import find_spec

import pytest

if find_spec("fastapi") is None:
    pytestmark = pytest.mark.skip(reason="fastapi is not installed")


def test_worker_metrics_endpoint_exposes_dependency_metrics() -> None:
    from fastapi.testclient import TestClient

    from atroposlib.workers.runtime import RuntimeWorker, build_worker_app

    worker = RuntimeWorker(
        api_base_url="http://runtime.invalid",
        poll_interval_seconds=0.01,
        request_timeout_seconds=0.01,
    )
    app = build_worker_app(worker)
    client = TestClient(app)

    with client:
        _ = client.get("/readyz")
        metrics = client.get("/metrics")

    assert metrics.status_code == 200
    assert "atropos_worker_dependency_checks_total" in metrics.text
    assert "atropos_worker_dependency_failures_total" in metrics.text
    assert "atropos_worker_dependency_ready" in metrics.text


def test_worker_health_endpoints_report_degraded_dependency() -> None:
    from fastapi.testclient import TestClient

    from atroposlib.workers.runtime import RuntimeWorker, build_worker_app

    worker = RuntimeWorker(
        api_base_url="http://runtime.invalid",
        poll_interval_seconds=0.01,
        request_timeout_seconds=0.01,
    )
    app = build_worker_app(worker)
    client = TestClient(app)

    with client:
        ready = client.get("/readyz")
        dependency = client.get("/depz")
        dependency_alias = client.get("/health/dependencies")

    assert ready.status_code == 503
    assert ready.json()["status"] == "not_ready"
    assert ready.json()["health_state"] in {"degraded", "unavailable"}
    assert ready.json()["control_plane_ready"] is True
    assert ready.json()["dependency_healthy"] is False

    assert dependency.status_code == 503
    assert dependency.json()["status"] == "degraded"
    assert dependency.json()["dependency"] == "runtime_api"
    assert dependency.json()["health_state"] in {"degraded", "unavailable"}

    assert dependency_alias.status_code == dependency.status_code
    assert dependency_alias.json()["dependency"] == "runtime_api"
