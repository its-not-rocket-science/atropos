from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from importlib.util import find_spec

import pytest

if find_spec("fastapi") is None:
    pytestmark = pytest.mark.skip(reason="fastapi is not installed")


@pytest.mark.stress
def test_many_concurrent_workers_submitting_groups() -> None:
    from fastapi.testclient import TestClient

    from atroposlib.api.server import build_runtime_app
    from atroposlib.api.storage import InMemoryStore

    app = build_runtime_app(store=InMemoryStore())
    client = TestClient(app)

    worker_count = 32
    groups_per_worker = 16

    def _submit(worker_idx: int, group_idx: int) -> tuple[int, bool]:
        request_id = f"worker-{worker_idx}-group-{group_idx}"
        response = client.post(
            "/scored_data",
            json={
                "environment_id": f"env-{worker_idx % 4}",
                "group_id": request_id,
                "records": [
                    {"sample_id": request_id, "score": 0.5},
                    {"sample_id": f"{request_id}-2", "score": 0.7},
                ],
            },
            headers={"X-Request-ID": request_id},
        )
        payload = response.json()
        return (response.status_code, bool(payload.get("deduplicated", False)))

    futures = []
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        for worker_idx in range(worker_count):
            for group_idx in range(groups_per_worker):
                futures.append(executor.submit(_submit, worker_idx, group_idx))

    status_codes = []
    dedupe_flags = []
    for future in as_completed(futures):
        status_code, dedupe = future.result()
        status_codes.append(status_code)
        dedupe_flags.append(dedupe)

    assert len(status_codes) == worker_count * groups_per_worker
    assert all(code == 200 for code in status_codes)
    assert not any(dedupe_flags)


@pytest.mark.stress
def test_partial_batch_pressure_marks_partial_failed() -> None:
    from fastapi.testclient import TestClient

    from atroposlib.api.server import build_runtime_app
    from atroposlib.api.storage import InMemoryStore

    app = build_runtime_app(store=InMemoryStore())
    client = TestClient(app)

    response = client.post(
        "/scored_data_list",
        json={
            "groups": [
                {
                    "environment_id": "partial-pressure",
                    "group_id": "valid-1",
                    "records": [{"sample_id": "ok-1", "score": 0.8}],
                },
                {
                    "environment_id": "partial-pressure",
                    "group_id": "invalid-1",
                    "records": ["not-a-record"],
                },
                {
                    "environment_id": "partial-pressure",
                    "group_id": "valid-2",
                    "records": [{"sample_id": "ok-2", "score": 0.9}],
                },
            ]
        },
        headers={"X-Request-ID": "partial-batch-1"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "partial_failed"
    assert payload["accepted_groups"] == 2
    assert payload["failed_groups"] == 1
    assert payload["accepted_count"] == 2


@pytest.mark.stress
def test_duplicate_submission_storm_is_deduplicated() -> None:
    from fastapi.testclient import TestClient

    from atroposlib.api.server import build_runtime_app
    from atroposlib.api.storage import InMemoryStore

    app = build_runtime_app(store=InMemoryStore())
    client = TestClient(app)

    def _submit() -> dict[str, object]:
        response = client.post(
            "/scored_data",
            json={
                "environment_id": "dup-storm",
                "group_id": "g-1",
                "records": [{"sample_id": "a", "score": 0.4}],
            },
            headers={"X-Request-ID": "dup-storm-request"},
        )
        assert response.status_code == 200
        return response.json()

    with ThreadPoolExecutor(max_workers=40) as executor:
        payloads = [future.result() for future in [executor.submit(_submit) for _ in range(120)]]

    deduplicated = [bool(item["deduplicated"]) for item in payloads]
    accepted = [int(item["accepted_count"]) for item in payloads]
    accepted_groups = [int(item["accepted_groups"]) for item in payloads]

    assert sum(accepted) == 1
    assert sum(accepted_groups) == 1
    assert deduplicated.count(False) == 1
    assert deduplicated.count(True) == 119


@pytest.mark.stress
def test_backpressure_under_full_queues_reduces_rate_limit() -> None:
    from atroposlib.envs.worker_runtime import WorkerRuntime

    runtime = WorkerRuntime(min_workers=1, max_workers=64, target_queue_depth=8)

    for idx in range(16):
        runtime.enqueue({"work": idx})

    baseline = runtime.recommended_workers(requested_workers=32, env="bp")
    runtime.update_trainer_feedback("bp", {"queue_depth": 128})
    throttled = runtime.recommended_workers(requested_workers=64, env="bp")

    for _ in range(20):
        runtime.update_trainer_feedback("bp", {"queue_depth": 256})

    max_throttled = runtime.recommended_workers(requested_workers=32, env="bp")

    assert baseline == 32
    assert throttled < baseline
    assert runtime.environment_rates["bp"] <= runtime.max_rate_limit
    assert runtime.environment_rates["bp"] >= runtime.min_rate_limit
    assert max_throttled <= int(round(32 * runtime.min_rate_limit)) + 1


@pytest.mark.stress
def test_rapid_env_register_disconnect_cycles_keep_runtime_ready() -> None:
    from fastapi.testclient import TestClient

    from atroposlib.api.server import build_runtime_app
    from atroposlib.api.storage import InMemoryStore

    cycles = 80

    for cycle in range(cycles):
        app = build_runtime_app(store=InMemoryStore())
        with TestClient(app) as client:
            ready = client.get("/health/ready")
            write = client.post(
                "/scored_data",
                json={
                    "environment_id": f"env-cycle-{cycle % 5}",
                    "group_id": f"group-{cycle}",
                    "records": [{"sample_id": f"sample-{cycle}", "score": 0.3}],
                },
                headers={"X-Request-ID": f"req-{cycle}"},
            )
        assert ready.status_code == 200
        assert write.status_code == 200
