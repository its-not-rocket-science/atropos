from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from atroposlib.envs.base import BaseEnv
from atroposlib.envs.components import EnvRuntime, EnvTransportClient
from atroposlib.envs.env_logic import EnvLogic


@dataclass
class PrefixLogic(EnvLogic):
    prefix: str = "prep-"

    def prepare_step(self, payload: dict[str, Any]) -> dict[str, Any]:
        return {"payload": f"{self.prefix}{payload['payload']}"}

    def finalize_step(self, transport_result: dict[str, Any]) -> dict[str, Any]:
        return {**transport_result, "finalized": True}


@dataclass
class FlakyTransport(EnvTransportClient):
    fail_until_attempt: int = 1
    attempts: int = 0

    def _send_once(self, payload: dict[str, Any]) -> dict[str, Any]:
        self.attempts += 1
        if self.attempts <= self.fail_until_attempt:
            raise RuntimeError("transient")
        return {"ok": True, "payload": payload, "attempts": self.attempts}


def test_step_delegates_to_runtime_transport_logs_and_checkpoint() -> None:
    env = BaseEnv()
    result = env.step({"task": "x"}, worker_count=2)

    assert result["ok"] is True
    assert result["payload"]["worker_count"] == 2
    assert len(env.logger.events) == 2
    assert env.checkpoint_manager.snapshots[-1] == result


def test_merge_yaml_and_cli_cli_takes_precedence() -> None:
    env = BaseEnv()
    merged = env.merge_yaml_and_cli(
        {"model": "baseline", "workers": 1},
        {"workers": 4},
    )
    assert merged == {"model": "baseline", "workers": 4}


def test_build_cli_args_is_deterministic() -> None:
    env = BaseEnv()
    args = env.build_cli_args({"workers": 2, "model_name": "gpt"})
    assert args == ["--model-name", "gpt", "--workers", "2"]


def test_dependency_injection_works_for_env_logic() -> None:
    env = BaseEnv(env_logic=PrefixLogic())

    result = env.step({"payload": "demo"})

    assert result["payload"]["work_item"] == {"payload": "prep-demo"}
    assert result["finalized"] is True


def test_transport_retries_then_succeeds() -> None:
    env = BaseEnv(transport_client=FlakyTransport(max_retries=2, fail_until_attempt=1))

    result = env.step({"task": "retry"})

    assert result["ok"] is True
    assert result["attempts"] == 2


def test_backward_compatibility_shim_types_are_usable() -> None:
    runtime = EnvRuntime()
    output = runtime.orchestrate({"task": "compat"}, requested_workers=3)

    transport = EnvTransportClient()
    response = transport.send(output)

    assert output["worker_count"] == 3
    assert response["ok"] is True


def test_trainer_feedback_reduces_env_rate_limit() -> None:
    env = BaseEnv()

    overloaded = env.step(
        {"env": "math", "task": "x", "trainer_feedback": {"queue_depth": 32}},
        worker_count=8,
    )
    recovered = env.step(
        {"env": "math", "task": "y", "trainer_feedback": {"queue_depth": 0}},
        worker_count=8,
    )

    overloaded_runtime = overloaded["payload"]
    recovered_runtime = recovered["payload"]
    assert overloaded_runtime["rate_limit"] < 1.0
    assert overloaded_runtime["worker_count"] < 8
    assert recovered_runtime["rate_limit"] > overloaded_runtime["rate_limit"]


def test_dynamic_scaling_accounts_for_trainer_queue_depth() -> None:
    runtime = EnvRuntime(max_workers=10)
    baseline = runtime.orchestrate({"task": "base"}, requested_workers=2, env="qa")
    for idx in range(6):
        runtime.enqueue({"task": f"queued-{idx}"})
    saturated = runtime.orchestrate(
        {"task": "heavy"},
        requested_workers=2,
        env="qa",
        trainer_feedback={"queue_depth": 6},
    )

    assert baseline["worker_count"] == 2
    assert saturated["worker_count"] > baseline["worker_count"]
    assert saturated["trainer_queue_depth"] == 6
