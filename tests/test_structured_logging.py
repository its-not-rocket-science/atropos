from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from importlib.util import find_spec
from typing import Any

import pytest

if find_spec("fastapi") is not None:
    from fastapi.testclient import TestClient

    from atroposlib.api.server import build_runtime_app
    from atroposlib.api.storage import InMemoryStore
else:  # pragma: no cover - optional dependency in CI variants
    TestClient = None

from atroposlib.envs.checkpoint_manager import CheckpointManager
from atroposlib.envs.metrics_logger import MetricsLogger
from atroposlib.envs.runtime_controller import RuntimeController
from atroposlib.envs.runtime_interfaces import BacklogManager, ItemSource, RolloutCollector
from atroposlib.logging_utils import (
    STABLE_LOG_FIELDS,
    StructuredJsonFormatter,
    build_log_context,
    resolve_log_format,
)


@dataclass
class ToyItemSource(ItemSource):
    def prepare_item(self, payload: dict[str, Any]) -> dict[str, Any]:
        samples = payload.get("samples", [])
        return {"samples": list(samples), "sample_count": len(samples)}


@dataclass
class InMemoryBacklog(BacklogManager):
    backlog: list[dict[str, Any]] = field(default_factory=list)
    max_workers: int = 4

    def orchestrate(
        self,
        work_item: dict[str, Any],
        requested_workers: int = 1,
        *,
        env: str = "default",
        trainer_feedback: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        self.backlog.append(dict(work_item))
        next_item = self.backlog.pop(0)
        selected = max(1, min(requested_workers, self.max_workers))
        return {
            "worker_count": selected,
            "work_item": next_item,
            "env": env,
            "trainer_queue_depth": int((trainer_feedback or {}).get("queue_depth", 0)),
            "rate_limit": 1.0,
        }


@dataclass
class ToyApiSender:
    def send(self, payload: dict[str, Any]) -> dict[str, Any]:
        item = payload["work_item"]
        return {
            "ok": True,
            "records": [
                {"text": sample["text"], "score": len(sample["text"])} for sample in item["samples"]
            ],
            "payload": payload,
        }


@dataclass
class ToyRolloutCollector(RolloutCollector):
    def collect(self, transport_result: dict[str, Any]) -> dict[str, Any]:
        return {
            "ok": transport_result["ok"],
            "scored_records": transport_result["records"],
            "worker_count": transport_result["payload"]["worker_count"],
        }


def test_structured_json_formatter_includes_stable_fields() -> None:
    logger = logging.getLogger("atroposlib.test.logging")
    record = logger.makeRecord(
        name=logger.name,
        level=logging.INFO,
        fn="test.py",
        lno=10,
        msg="runtime_step_finished",
        args=(),
        exc_info=None,
        extra=build_log_context(
            env_id="toy-env",
            request_id="req-123",
            batch_id="batch-456",
            worker_id=2,
            endpoint="/scored_data",
            current_step=7,
            queue_depth=4,
        ),
    )
    formatted = StructuredJsonFormatter().format(record)
    payload = json.loads(formatted)

    for key in STABLE_LOG_FIELDS:
        assert key in payload
    assert payload["env_id"] == "toy-env"
    assert payload["request_id"] == "req-123"
    assert payload["extra"]["queue_depth"] == 4


def test_resolve_log_format_supports_pretty_alias() -> None:
    assert resolve_log_format("text") == "pretty"
    assert resolve_log_format("json") == "json"


def test_runtime_controller_logs_structured_context(caplog) -> None:  # type: ignore[no-untyped-def]
    runtime = RuntimeController(
        item_source=ToyItemSource(),
        backlog_manager=InMemoryBacklog(),
        send_to_api=ToyApiSender(),
        rollout_collector=ToyRolloutCollector(),
        metrics_logger=MetricsLogger(),
        checkpoint_manager=CheckpointManager(),
    )

    with caplog.at_level(logging.INFO, logger="atroposlib.envs.runtime"):
        runtime.run_step(
            {
                "env": "toy",
                "samples": [{"text": "alpha"}],
                "request_id": "req-runtime-1",
                "batch_id": "batch-runtime-1",
                "current_step": 42,
            },
            worker_count=2,
        )

    events = [record for record in caplog.records if record.name == "atroposlib.envs.runtime"]
    assert events
    started = next(record for record in events if record.msg == "runtime_step_started")
    assert started.env_id == "toy"
    assert started.request_id == "req-runtime-1"
    assert started.batch_id == "batch-runtime-1"
    assert started.current_step == 42


@pytest.mark.skipif(TestClient is None, reason="fastapi is not installed")
def test_api_server_json_logs_include_request_context(capsys) -> None:  # type: ignore[no-untyped-def]
    app = build_runtime_app(store=InMemoryStore(), log_format="json")
    client = TestClient(app)

    response = client.post(
        "/scored_data",
        headers={"X-Request-ID": "req-api-1"},
        json={
            "environment_id": "toy-env",
            "group_id": "batch-api-1",
            "records": [{"sample_id": "s-1", "score": 1.0}],
        },
    )
    assert response.status_code == 200

    captured = capsys.readouterr().err.strip().splitlines()
    assert captured
    parsed_logs = [json.loads(line) for line in captured if line.strip().startswith("{")]
    scored_log = next(entry for entry in parsed_logs if entry["message"] == "scored_data_ingested")

    assert scored_log["env_id"] == "toy-env"
    assert scored_log["request_id"] == "req-api-1"
    assert scored_log["batch_id"] == "batch-api-1"
    assert scored_log["endpoint"] == "/scored_data"
