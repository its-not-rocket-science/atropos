from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from atroposlib.envs.base import BaseEnv
from atroposlib.envs.components import EnvRuntime, EnvTransportClient
from atroposlib.envs.env_logic import EnvLogic


class DummySpan:
    def __init__(self) -> None:
        self.attributes: dict[str, object] = {}

    def set_attribute(self, key: str, value: object) -> None:
        self.attributes[key] = value


class DummyTracingHooks:
    def __init__(self) -> None:
        self.span = DummySpan()

    def step_span(self, *, worker_count: int, payload: dict[str, Any]):
        _ = (worker_count, payload)

        tracing_span = self.span

        class _Ctx:
            def __enter__(self):
                return tracing_span

            def __exit__(self, exc_type, exc, tb):
                _ = (exc_type, exc, tb)
                return False

        return _Ctx()

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


def test_step_emits_worker_utilization_and_trace_attributes() -> None:
    tracing = DummyTracingHooks()
    env = BaseEnv(tracing_hooks=tracing, env_name="prod")

    _ = env.step({"task": "x"}, worker_count=4)

    assert tracing.span.attributes["atropos.worker.requested"] == 4
    assert "atropos.worker.utilization" in tracing.span.attributes
    assert env.logger.events[-1]["metadata"]["env"] == "prod"
