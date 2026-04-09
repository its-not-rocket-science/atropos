from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

from atroposlib.envs.base import BaseEnv
from atroposlib.envs.components import EnvTransportClient


@dataclass
class DeterministicScoringTransport(EnvTransportClient):
    """In-memory scoring transport that avoids any external model calls."""

    def send(self, payload: dict[str, Any]) -> dict[str, Any]:
        work_item = payload["work_item"]
        samples = work_item.get("samples", [])
        scored_records = [
            {
                "sample_id": sample["sample_id"],
                "text": sample["text"],
                "score": len(sample["text"]),
            }
            for sample in samples
        ]
        return {
            "ok": True,
            "payload": payload,
            "scored_records": scored_records,
        }


@dataclass
class ToyRuntimeAPI:
    """Minimal in-process API facade used for CI-safe integration tests."""

    running: bool = False
    env_registry: dict[str, BaseEnv] = field(default_factory=dict)
    scored_store: defaultdict[str, list[dict[str, Any]]] = field(
        default_factory=lambda: defaultdict(list)
    )

    def start(self) -> None:
        self.running = True

    def register_environment(self, environment_id: str) -> str:
        if not self.running:
            raise RuntimeError("API must be started before registering environments")

        self.env_registry[environment_id] = BaseEnv(transport=DeterministicScoringTransport())
        return environment_id

    def produce_scored_data(self, environment_id: str, payload: dict[str, Any]) -> int:
        env = self.env_registry.get(environment_id)
        if env is None:
            raise KeyError(f"Unknown environment: {environment_id}")

        result = env.step(payload, worker_count=1)
        records = result["scored_records"]
        self.scored_store[environment_id].extend(records)
        return len(records)

    def fetch_batch(self, environment_id: str, limit: int = 2) -> list[dict[str, Any]]:
        if environment_id not in self.env_registry:
            raise KeyError(f"Unknown environment: {environment_id}")

        batch = self.scored_store[environment_id][:limit]
        del self.scored_store[environment_id][:limit]
        return batch


@dataclass
class ToyTrainerPreprocessor:
    """Trainer-side preprocessing step used by the integration test."""

    def run(self, batch: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if not batch:
            return []

        max_score = max(item["score"] for item in batch)
        if max_score <= 0:
            return [{**item, "normalized_score": 0.0} for item in batch]

        return [
            {
                **item,
                "normalized_score": round(item["score"] / max_score, 3),
            }
            for item in batch
        ]


def test_toy_environment_end_to_end_flow() -> None:
    """Validate API start -> env register -> scoring -> batch fetch -> preprocessing."""

    api = ToyRuntimeAPI()

    api.start()
    assert api.running is True

    env_id = api.register_environment("toy-env")
    assert env_id == "toy-env"

    produced = api.produce_scored_data(
        env_id,
        {
            "samples": [
                {"sample_id": "a", "text": "alpha"},
                {"sample_id": "b", "text": "beta beta"},
                {"sample_id": "c", "text": "gamma"},
            ]
        },
    )
    assert produced == 3

    batch = api.fetch_batch(env_id, limit=2)
    assert [item["sample_id"] for item in batch] == ["a", "b"]

    preprocessed = ToyTrainerPreprocessor().run(batch)
    assert [item["normalized_score"] for item in preprocessed] == [0.556, 1.0]
