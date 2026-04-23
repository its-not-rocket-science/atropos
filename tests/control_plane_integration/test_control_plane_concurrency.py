from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from importlib.util import find_spec
from threading import Lock
from typing import Any

import pytest

if find_spec("fastapi") is None:
    pytestmark = pytest.mark.skip(reason="fastapi is not installed")


def test_concurrent_environment_submissions() -> None:
    from fastapi.testclient import TestClient

    from atroposlib.api.server import build_runtime_app
    from atroposlib.api.storage import InMemoryStore

    app = build_runtime_app(store=InMemoryStore())
    with TestClient(app) as client:

        def _register_env(idx: int) -> dict[str, Any]:
            env_id = f"env-{idx % 12}"
            response = client.post("/environments", json={"environment_id": env_id})
            assert response.status_code == 200
            return response.json()

        with ThreadPoolExecutor(max_workers=24) as executor:
            responses = [
                future.result()
                for future in [executor.submit(_register_env, idx) for idx in range(240)]
            ]

        listed = client.get("/environments")

    assert listed.status_code == 200
    assert listed.json()["count"] == 12
    assert len(responses) == 240
    assert sum(1 for item in responses if item["created"]) == 12


def test_simultaneous_register_and_disconnect_cycles() -> None:
    from fastapi.testclient import TestClient

    from atroposlib.api.server import build_runtime_app
    from atroposlib.api.storage import InMemoryStore

    app = build_runtime_app(store=InMemoryStore())

    def _ephemeral_register(idx: int) -> int:
        with TestClient(app) as transient_client:
            response = transient_client.post(
                "/environments",
                json={"environment_id": f"disconnect-env-{idx % 20}"},
            )
            return response.status_code

    with ThreadPoolExecutor(max_workers=32) as executor:
        status_codes = [
            future.result()
            for future in [executor.submit(_ephemeral_register, idx) for idx in range(120)]
        ]

    with TestClient(app) as verifier:
        listed = verifier.get("/environments")

    assert all(code == 200 for code in status_codes)
    assert listed.status_code == 200
    assert listed.json()["count"] == 20


def test_mixed_group_sizes_under_pressure() -> None:
    from fastapi.testclient import TestClient

    from atroposlib.api.server import build_runtime_app
    from atroposlib.api.storage import InMemoryStore

    app = build_runtime_app(store=InMemoryStore())
    with TestClient(app) as client:

        def _submit(idx: int) -> dict[str, Any]:
            record_count = (idx % 7) + 1
            records = [
                {
                    "sample_id": f"sample-{idx}-{record_idx}",
                    "score": float(record_idx) / max(record_count, 1),
                }
                for record_idx in range(record_count)
            ]
            response = client.post(
                "/scored_data",
                headers={"X-Request-ID": f"mixed-request-{idx}"},
                json={
                    "environment_id": "pressure-env",
                    "group_id": f"mixed-group-{idx}",
                    "records": records,
                },
            )
            assert response.status_code == 200
            return response.json()

        with ThreadPoolExecutor(max_workers=20) as executor:
            payloads = [
                future.result() for future in [executor.submit(_submit, idx) for idx in range(180)]
            ]

        records = client.get(
            "/scored_data_list",
            params={"environment_id": "pressure-env", "limit": 2000},
        )

    assert records.status_code == 200
    assert payloads
    assert all(item["status"] == "completed" for item in payloads)
    assert sum(int(item["accepted_count"]) for item in payloads) == records.json()["count"]


def test_repeated_duplicate_retries_are_idempotent() -> None:
    from fastapi.testclient import TestClient

    from atroposlib.api.server import build_runtime_app
    from atroposlib.api.storage import InMemoryStore

    app = build_runtime_app(store=InMemoryStore())
    with TestClient(app) as client:

        def _retry_duplicate() -> dict[str, Any]:
            response = client.post(
                "/scored_data",
                headers={"X-Request-ID": "dup-retry-request"},
                json={
                    "environment_id": "retry-env",
                    "group_id": "retry-group",
                    "records": [{"sample_id": "same", "score": 0.8}],
                },
            )
            assert response.status_code == 200
            return response.json()

        with ThreadPoolExecutor(max_workers=64) as executor:
            payloads = [
                future.result()
                for future in [executor.submit(_retry_duplicate) for _ in range(300)]
            ]

    assert sum(int(item["accepted_groups"]) for item in payloads) == 1
    assert sum(int(item["accepted_count"]) for item in payloads) == 1
    assert sum(1 for item in payloads if bool(item["deduplicated"])) == 299


def test_batch_consumers_can_poll_during_ingestion_spike() -> None:
    from fastapi.testclient import TestClient

    from atroposlib.api.server import build_runtime_app
    from atroposlib.api.storage import InMemoryStore

    app = build_runtime_app(store=InMemoryStore())
    with TestClient(app) as client:
        producer_total = 220

        def _produce(idx: int) -> int:
            response = client.post(
                "/scored_data",
                headers={"X-Request-ID": f"spike-request-{idx}"},
                json={
                    "environment_id": "poll-spike-env",
                    "group_id": f"spike-group-{idx}",
                    "records": [{"sample_id": f"spike-{idx}", "score": 0.5}],
                },
            )
            return response.status_code

        def _poll() -> int:
            response = client.get(
                "/scored_data_list",
                params={"environment_id": "poll-spike-env", "limit": 1000},
            )
            return response.status_code

        futures = []
        with ThreadPoolExecutor(max_workers=48) as executor:
            for idx in range(producer_total):
                futures.append(executor.submit(_produce, idx))
                if idx % 4 == 0:
                    futures.append(executor.submit(_poll))

            status_codes = [future.result() for future in as_completed(futures)]

        final_records = client.get(
            "/scored_data_list",
            params={"environment_id": "poll-spike-env", "limit": 1000},
        )

    assert all(code == 200 for code in status_codes)
    assert final_records.status_code == 200
    assert final_records.json()["count"] == producer_total


def test_store_reconnect_during_load_keeps_ingestion_available() -> None:
    from fastapi.testclient import TestClient

    from atroposlib.api.server import build_runtime_app
    from atroposlib.api.storage import InMemoryStore

    class FlakyReconnectStore(InMemoryStore):
        """In-memory store that simulates transient dependency reconnects."""

        def __init__(self, *, fail_metric_calls: int) -> None:
            super().__init__()
            self._fail_metric_calls = fail_metric_calls
            self._metric_failures_seen = 0
            self._metric_lock = Lock()

        def get_scored_queue_metrics(self, *, now: datetime):  # type: ignore[override]
            _ = now
            with self._metric_lock:
                if self._metric_failures_seen < self._fail_metric_calls:
                    self._metric_failures_seen += 1
                    raise RuntimeError("transient store reconnect")
            return super().get_scored_queue_metrics(now=datetime.now(tz=timezone.utc))

    app = build_runtime_app(store=FlakyReconnectStore(fail_metric_calls=30))
    with TestClient(app) as client:

        def _submit(idx: int) -> int:
            response = client.post(
                "/scored_data",
                headers={"X-Request-ID": f"reconnect-request-{idx}"},
                json={
                    "environment_id": "reconnect-env",
                    "group_id": f"reconnect-group-{idx}",
                    "records": [{"sample_id": f"reconnect-{idx}", "score": 0.42}],
                },
            )
            return response.status_code

        with ThreadPoolExecutor(max_workers=32) as executor:
            statuses = [
                future.result() for future in [executor.submit(_submit, idx) for idx in range(160)]
            ]

        final_records = client.get(
            "/scored_data_list",
            params={"environment_id": "reconnect-env", "limit": 500},
        )

    assert all(code == 200 for code in statuses)
    assert final_records.status_code == 200
    assert final_records.json()["count"] == 160
