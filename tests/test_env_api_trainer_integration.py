from __future__ import annotations

from dataclasses import dataclass
from importlib.util import find_spec
from typing import Any

import pytest

from atroposlib.envs.base import BaseEnv
from atroposlib.envs.transport_client import TransportClient

if find_spec("fastapi") is None:
    pytestmark = pytest.mark.skip(reason="fastapi is not installed")


@dataclass
class ScoringTransport(TransportClient):
    def send(self, payload: dict[str, Any]) -> dict[str, Any]:
        samples = payload["work_item"]["samples"]
        scored = [
            {"sample_id": item["sample_id"], "score": float(len(item["text"]))}
            for item in samples
        ]
        return {"ok": True, "scored_records": scored, "payload": payload}


@dataclass
class ToyTrainer:
    def run_epoch(self, records: list[dict[str, object]]) -> dict[str, float]:
        if not records:
            return {"count": 0.0, "avg_score": 0.0}
        total = sum(float(item["score"]) for item in records)
        return {"count": float(len(records)), "avg_score": total / len(records)}


def test_env_to_api_to_trainer_loop() -> None:
    from fastapi.testclient import TestClient
    
    from atroposlib.api.server import build_runtime_app
    from atroposlib.api.storage import InMemoryStore

    client = TestClient(build_runtime_app(store=InMemoryStore()))
    env = BaseEnv(transport=ScoringTransport())

    rollout = env.step(
        {
            "env": "toy",
            "samples": [
                {"sample_id": "s-1", "text": "alpha"},
                {"sample_id": "s-2", "text": "beta beta"},
                {"sample_id": "s-3", "text": "gamma"},
            ],
        },
        worker_count=4,
    )

    ingest = client.post(
        "/scored_data",
        headers={"X-Request-ID": "req-1"},
        json={"environment_id": "toy", "records": rollout["scored_records"]},
    )
    assert ingest.status_code == 200
    assert ingest.json()["accepted_count"] == 3

    fetched = client.get("/scored_data_list", params={"environment_id": "toy", "limit": 10})
    assert fetched.status_code == 200
    payload = fetched.json()
    assert payload["count"] == 3

    trainer_metrics = ToyTrainer().run_epoch(payload["records"])
    assert trainer_metrics == {"count": 3.0, "avg_score": 6.333333333333333}
