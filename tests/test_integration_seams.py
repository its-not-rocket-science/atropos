from __future__ import annotations

from datetime import datetime
from pathlib import Path
from unittest.mock import Mock

import pytest

from atropos.abtesting.models import (
    ABTestConfig,
    ExperimentStatus,
    StatisticalTestType,
    Variant,
    VariantMetrics,
)
from atropos.abtesting.runner import ExperimentRunner
from atropos.abtesting.store import ExperimentStore
from atropos.cli import main
from atropos.deployment.models import (
    DeploymentRequest,
    DeploymentResult,
    DeploymentStatus,
    DeploymentStrategyType,
)
from atropos.deployment.strategies import CanaryStrategy
from atropos.models import DeploymentScenario
from atropos.pipeline.models import PipelineResult, PipelineStage, StageResult, StageStatus
from atropos.telemetry import get_parser, telemetry_to_scenario
from atropos.telemetry_collector import CollectionResult, TelemetryData, collect_and_save


def _scenario(name: str = "seam-scenario") -> DeploymentScenario:
    return DeploymentScenario(
        name=name,
        parameters_b=7.0,
        memory_gb=16.0,
        throughput_toks_per_sec=45.0,
        power_watts=280.0,
        requests_per_day=50_000,
        tokens_per_request=900,
        electricity_cost_per_kwh=0.15,
        annual_hardware_cost_usd=24_000.0,
        one_time_project_cost_usd=27_000.0,
    )


def _pipeline_result(status: StageStatus) -> PipelineResult:
    result = PipelineResult(
        pipeline_name="seam-pipeline",
        scenario_name="demo",
        strategy_name="mild_pruning",
    )
    result.final_status = status
    result.stages = [
        StageResult(
            stage=PipelineStage.ASSESS,
            status=StageStatus.SUCCESS,
            message="assessment complete",
        )
    ]
    return result


def test_cli_pipeline_wires_config_into_runner(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Protects the CLI->config->runner seam from silently dropping parsed config values."""
    cfg = tmp_path / "pipeline.yaml"
    cfg.write_text("pipeline:\n  name: seam-pipeline\n")

    observed: dict[str, object] = {}

    def _fake_run_pipeline(*, config, scenario, strategy, grid_co2e, dry_run):  # type: ignore[no-untyped-def]
        observed["config_name"] = config.name
        observed["scenario_name"] = scenario.name
        observed["strategy_name"] = strategy.name
        observed["dry_run"] = dry_run
        observed["grid_co2e"] = grid_co2e
        return _pipeline_result(StageStatus.SUCCESS)

    monkeypatch.setattr("atropos.cli.run_pipeline", _fake_run_pipeline)
    monkeypatch.setattr(
        "atropos.cli._load_scenario_input",
        lambda _arg: ("demo", _scenario("demo")),
    )

    exit_code = main(
        [
            "pipeline",
            "demo",
            "--config",
            str(cfg),
            "--strategy",
            "mild_pruning",
            "--dry-run",
        ]
    )

    assert exit_code == 0
    assert observed["config_name"] == "seam-pipeline"
    assert observed["scenario_name"] == "demo"
    assert observed["strategy_name"] == "mild_pruning"
    assert observed["dry_run"] is True
    assert observed["grid_co2e"] == 0.35


def test_cli_pipeline_returns_nonzero_when_runner_fails(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Protects callers that rely on CLI exit codes when pipeline execution fails."""
    cfg = tmp_path / "pipeline.yaml"
    cfg.write_text("pipeline:\n  name: seam-pipeline\n")

    monkeypatch.setattr(
        "atropos.cli._load_scenario_input",
        lambda _arg: ("demo", _scenario("demo")),
    )
    monkeypatch.setattr(
        "atropos.cli.run_pipeline",
        lambda **_kwargs: _pipeline_result(StageStatus.FAILED),
    )

    exit_code = main(["pipeline", "demo", "--config", str(cfg), "--strategy", "mild_pruning"])
    assert exit_code == 1


def test_ab_runner_to_store_to_analysis_with_control_fallback(tmp_path: Path) -> None:
    """Protects analysis/store integration when the configured control variant has no metrics."""
    config = ABTestConfig(
        experiment_id="exp-seam",
        name="Seam test",
        variants=[
            Variant("variant-a", "A", "/models/a", 0.5),
            Variant("variant-b", "B", "/models/b", 0.5),
        ],
        primary_metric="throughput_toks_per_sec",
        secondary_metrics=[],
        traffic_allocation=0.2,
        significance_level=0.05,
        statistical_power=0.8,
        test_type=StatisticalTestType.T_TEST,
        min_sample_size_per_variant=5,
        max_duration_hours=1.0,
        deployment_platform="vllm",
    )
    runner = ExperimentRunner(config, platform=Mock())

    now = datetime.now().isoformat()
    metrics = {
        # Intentionally omit configured control "variant-a" to exercise fallback logic.
        "variant-b": VariantMetrics(
            variant_id="variant-b",
            sample_count=25,
            metrics={"throughput_toks_per_sec": {"mean": 100.0, "std": 5.0, "count": 25}},
            timestamp_start=now,
            timestamp_end=now,
        ),
        "variant-shadow": VariantMetrics(
            variant_id="variant-shadow",
            sample_count=25,
            metrics={"throughput_toks_per_sec": {"mean": 120.0, "std": 5.0, "count": 25}},
            timestamp_start=now,
            timestamp_end=now,
        ),
    }

    def _fake_collect() -> dict[str, VariantMetrics]:
        runner._variant_metrics = metrics
        return metrics

    runner._collect_metrics = _fake_collect  # type: ignore[method-assign]

    analysis = runner.analyze()
    assert "throughput_toks_per_sec_variant-b_vs_variant-shadow" in analysis

    store = ExperimentStore(base_dir=tmp_path)
    store.save_config(config)
    store.update_experiment(
        config.experiment_id,
        status=ExperimentStatus.RUNNING,
        variant_metrics=metrics,
        statistical_results=analysis,
    )

    persisted = store.load_experiment(config.experiment_id)
    assert persisted is not None
    assert persisted["status"] == ExperimentStatus.RUNNING.value
    assert "throughput_toks_per_sec_variant-b_vs_variant-shadow" in persisted["statistical_results"]


def test_collect_and_import_telemetry_json_contract(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Protects collector->import seam so saved telemetry remains consumable by scenario import."""

    class _FakeCollector:
        def collect(self) -> CollectionResult:
            return CollectionResult(
                success=True,
                aggregated=TelemetryData(
                    source="vllm",
                    memory_gb=12.0,
                    throughput_toks_per_sec=88.0,
                    latency_ms_per_request=30.0,
                    tokens_per_request=512.0,
                    parameters_b=13.0,
                    power_watts=220.0,
                    requests_per_day=12_000,
                ),
                metadata={"sample_count": 3},
            )

    output = tmp_path / "telemetry.json"
    monkeypatch.setattr(
        "atropos.telemetry_collector.get_collector",
        lambda *_args, **_kwargs: _FakeCollector(),
    )

    result = collect_and_save("vllm", "http://server", output)
    assert result.success is True
    assert output.exists()

    telemetry = get_parser("json").parse_file(output)
    scenario = telemetry_to_scenario(telemetry, name="from-collector", requests_per_day=20_000)

    assert scenario.memory_gb == 12.0
    assert scenario.throughput_toks_per_sec == 88.0
    assert scenario.requests_per_day == 20_000


def test_collect_and_save_does_not_write_file_on_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Protects import workflows from reading stale partial files after collection failures."""

    class _FailCollector:
        def collect(self) -> CollectionResult:
            return CollectionResult(success=False, error_message="endpoint down")

    output = tmp_path / "telemetry.json"
    monkeypatch.setattr(
        "atropos.telemetry_collector.get_collector",
        lambda *_args, **_kwargs: _FailCollector(),
    )

    result = collect_and_save("vllm", "http://server", output)

    assert result.success is False
    assert not output.exists()


def test_canary_strategy_failure_triggers_rollback_with_context() -> None:
    """Protects deployment rollback seam by asserting failure context survives strategy handling."""
    platform = Mock()
    request = DeploymentRequest(
        model_path="/models/new",
        platform="vllm",
        strategy=DeploymentStrategyType.CANARY,
    )

    platform.deploy.return_value = DeploymentResult(
        request=request,
        deployment_id="dep-123",
        status=DeploymentStatus.SUCCESS,
        message="deployed",
    )
    platform.get_status.return_value = DeploymentResult(
        request=request,
        status=DeploymentStatus.FAILED,
        message="pod unhealthy",
    )
    platform.rollback.return_value = DeploymentResult(
        request=request,
        status=DeploymentStatus.FAILED,
        message="rollback API down",
    )

    strategy = CanaryStrategy(
        {
            "initial_percent": 10.0,
            "increment_percent": 30.0,
            "poll_interval_seconds": 0.0,
            "timeout_seconds": 5.0,
            "max_errors": 1,
        }
    )
    result = strategy.execute(platform, request)

    assert result.status == DeploymentStatus.FAILED
    assert result.deployment_id == "dep-123"
    assert "rollback api down" in result.message.lower()
    assert result.metrics["failure_reason"] == "pod unhealthy"
    platform.rollback.assert_called_once_with("dep-123")
