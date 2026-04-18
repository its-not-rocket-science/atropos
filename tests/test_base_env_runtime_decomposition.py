from __future__ import annotations

from dataclasses import dataclass

from atroposlib.envs.compatibility_adapter import BaseEnvCompatibilityAdapter
from atroposlib.envs.dependency_factory import BaseEnvDependencyFactory
from atroposlib.envs.env_logic import PassthroughEnvLogic
from atroposlib.envs.metrics_logger import MetricsLogger
from atroposlib.envs.transport_client import TransportClient


class EchoTransport(TransportClient):
    def _send_once(self, payload: dict[str, object]) -> dict[str, object]:
        return {"ok": True, "payload": payload}


@dataclass
class StepStub:
    calls: list[dict[str, object]]

    def __call__(self, payload: dict[str, object], worker_count: int = 1) -> dict[str, object]:
        self.calls.append({"payload": dict(payload), "worker_count": worker_count})
        return {"ok": True, "payload": payload, "worker_count": worker_count}


def test_dependency_factory_wires_seed_and_controller_dependencies() -> None:
    factory = BaseEnvDependencyFactory()

    wiring = factory.build(seed=7)

    assert wiring.seed_manager is not None
    assert wiring.collaborators.worker_runtime.seed_manager is wiring.seed_manager
    assert wiring.runtime_controller.backlog_manager is wiring.collaborators.worker_runtime
    assert wiring.runtime_controller.send_to_api is wiring.collaborators.transport_client
    assert isinstance(wiring.collaborators.env_logic, PassthroughEnvLogic)


def test_compatibility_adapter_serve_supports_single_payload_and_stream() -> None:
    adapter = BaseEnvCompatibilityAdapter(
        worker_runtime=BaseEnvDependencyFactory().build().collaborators.worker_runtime,
        transport_client=EchoTransport(),
        metrics_logger=MetricsLogger(),
        checkpoint_manager=BaseEnvDependencyFactory().build().collaborators.checkpoint_manager,
        cli_adapter=BaseEnvDependencyFactory().build().collaborators.cli_adapter,
    )
    step = StepStub(calls=[])

    one = adapter.serve(step, {"task": "single"}, worker_count=2)
    many = adapter.serve(step, [{"task": "a"}, {"task": "b"}], worker_count=3)

    assert isinstance(one, dict)
    assert isinstance(many, list)
    assert [call["worker_count"] for call in step.calls] == [2, 3, 3]


def test_compatibility_adapter_delegates_legacy_apis() -> None:
    wiring = BaseEnvDependencyFactory().build()
    adapter = BaseEnvCompatibilityAdapter(
        worker_runtime=wiring.collaborators.worker_runtime,
        transport_client=EchoTransport(),
        metrics_logger=wiring.collaborators.metrics_logger,
        checkpoint_manager=wiring.collaborators.checkpoint_manager,
        cli_adapter=wiring.collaborators.cli_adapter,
    )

    runtime_result = adapter.orchestrate_workers({"task": "x"}, worker_count=2)
    response = adapter.call_api(runtime_result)
    adapter.log_event("compat_event", source="test")
    adapter.checkpoint({"ok": True})

    assert runtime_result["worker_count"] == 2
    assert response["ok"] is True
    assert adapter.metrics_logger.events[-1]["event"] == "compat_event"
    assert adapter.checkpoint_manager.latest() == {"ok": True}
