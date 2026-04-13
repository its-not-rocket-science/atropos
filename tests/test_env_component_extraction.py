from __future__ import annotations

from atroposlib.envs.base import BaseEnv
from atroposlib.envs.checkpoint_manager import CheckpointManager
from atroposlib.envs.cli_adapter import CliAdapter
from atroposlib.envs.metrics_logger import MetricsLogger
from atroposlib.envs.transport_client import TransportClient
from atroposlib.envs.worker_runtime import WorkerRuntime


class EchoTransport(TransportClient):
    def _send_once(self, payload: dict[str, object]) -> dict[str, object]:
        return {"ok": True, "payload": payload}


def test_worker_runtime_orchestrate_reports_scaling_state() -> None:
    runtime = WorkerRuntime(max_workers=6)
    for idx in range(3):
        runtime.enqueue({"idx": idx})

    result = runtime.orchestrate({"task": "x"}, requested_workers=2, env="qa")

    assert result["worker_count"] == 4
    assert result["env"] == "qa"


def test_transport_client_send_uses_overridden_send_once() -> None:
    transport = EchoTransport()

    result = transport.send({"x": 1})

    assert result == {"ok": True, "payload": {"x": 1}}


def test_checkpoint_manager_latest_returns_copy() -> None:
    manager = CheckpointManager()
    manager.save({"iteration": 1})

    latest = manager.latest()

    assert latest == {"iteration": 1}
    assert latest is not None
    latest["iteration"] = 99
    assert manager.latest() == {"iteration": 1}


def test_metrics_logger_tracks_events_and_metrics() -> None:
    logger = MetricsLogger()

    logger.log_event("step_started", env="qa")
    logger.record_metrics({"reward": 1.5}, step=4)

    assert logger.events[0]["event"] == "step_started"
    assert logger.metrics[0] == {"reward": 1.5, "step": 4.0}


def test_cli_adapter_builds_and_merges_configs() -> None:
    cli = CliAdapter()

    args = cli.build_cli_args({"model_name": "demo", "workers": 2})
    merged = cli.merge_yaml_and_cli({"workers": 1, "lr": 0.01}, {"workers": 2})

    assert args == ["--model-name", "demo", "--workers", "2"]
    assert merged == {"workers": 2, "lr": 0.01}


def test_base_env_serve_process_and_evaluate_are_compatible_aliases() -> None:
    env = BaseEnv(transport_client=EchoTransport())

    process_result = env.process({"task": "process"}, worker_count=2)
    evaluate_result = env.evaluate({"task": "evaluate"}, worker_count=1)
    serve_result = env.serve({"task": "serve"}, worker_count=1)

    assert process_result["ok"] is True
    assert evaluate_result["ok"] is True
    assert isinstance(serve_result, dict)
    assert serve_result["ok"] is True
