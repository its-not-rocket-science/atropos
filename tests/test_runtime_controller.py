from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from atroposlib.envs.checkpoint_manager import CheckpointManager
from atroposlib.envs.metrics_logger import MetricsLogger
from atroposlib.envs.runtime_controller import RuntimeController
from atroposlib.envs.runtime_interfaces import BacklogManager, ItemSource, RolloutCollector


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


@dataclass
class ToyEnvironment:
    """Toy env that does not inherit BaseEnv and only exposes user-facing API."""

    runtime: RuntimeController

    def run(self, samples: list[dict[str, str]]) -> dict[str, Any]:
        return self.runtime.run_step({"env": "toy", "samples": samples}, worker_count=2)


def test_runtime_controller_runs_toy_env_end_to_end_without_baseenv() -> None:
    runtime = RuntimeController(
        item_source=ToyItemSource(),
        backlog_manager=InMemoryBacklog(),
        send_to_api=ToyApiSender(),
        rollout_collector=ToyRolloutCollector(),
        metrics_logger=MetricsLogger(),
        checkpoint_manager=CheckpointManager(),
    )
    env = ToyEnvironment(runtime=runtime)

    result = env.run(
        [
            {"text": "alpha"},
            {"text": "beta beta"},
        ]
    )

    assert result["ok"] is True
    assert [item["score"] for item in result["scored_records"]] == [5, 9]
    assert result["worker_count"] == 2
    assert runtime.checkpoint_manager.latest() == result
