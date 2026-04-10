"""Tests for pipeline module."""

import sys
from pathlib import Path

import pytest

from atropos.models import DeploymentScenario, OptimizationStrategy
from atropos.pipeline.config import PipelineConfig
from atropos.pipeline.models import PipelineStage, StageStatus
from atropos.pipeline.runner import PipelineRunner, _execute_external_command


def make_scenario() -> DeploymentScenario:
    """Create a test deployment scenario."""
    return DeploymentScenario(
        name="test-scenario",
        parameters_b=10.0,
        memory_gb=8.0,
        throughput_toks_per_sec=20.0,
        power_watts=200.0,
        requests_per_day=1000,
        tokens_per_request=1000,
        electricity_cost_per_kwh=0.15,
        annual_hardware_cost_usd=1000.0,
        one_time_project_cost_usd=500.0,
    )


def make_strategy() -> OptimizationStrategy:
    """Create a test optimization strategy."""
    return OptimizationStrategy(
        name="test-strategy",
        parameter_reduction_fraction=0.2,
        memory_reduction_fraction=0.2,
        throughput_improvement_fraction=0.1,
        power_reduction_fraction=0.1,
        quality_risk="low",
    )


def test_pipeline_config_from_dict() -> None:
    """Test PipelineConfig creation from dictionary."""
    data = {
        "name": "test-pipeline",
        "auto_execute": True,
        "thresholds": {
            "max_break_even_months": 6,
            "min_annual_savings_usd": 5000.0,
            "max_quality_risk": "low",
            "min_expected_quality": 0.8,
        },
        "quality_prediction": {
            "method": "obd_obs",
            "uncertainty_method": "quantile",
            "metric": "perplexity",
            "confidence_level": 0.9,
            "baseline_quality": 1.0,
        },
        "pruning": {
            "framework": "llm-pruner",
            "target_sparsity": 0.25,
            "structured": True,
        },
        "recovery": {
            "enabled": True,
            "epochs": 2,
            "learning_rate": 1e-5,
            "batch_size": 16,
        },
        "validation": {
            "tolerance_percent": 5.0,
            "quality_benchmark": "humaneval",
        },
        "deployment": {
            "auto_deploy": False,
            "strategy": "immediate",
        },
    }

    config = PipelineConfig.from_dict(data)
    assert config.name == "test-pipeline"
    assert config.auto_execute is True
    assert config.thresholds is not None
    assert config.thresholds.max_break_even_months == 6
    assert config.thresholds.min_annual_savings_usd == 5000.0
    assert config.thresholds.max_quality_risk == "low"
    assert config.thresholds.min_expected_quality == 0.8
    assert config.quality_prediction is not None
    assert config.quality_prediction.method == "obd_obs"
    assert config.pruning is not None
    assert config.pruning.framework == "llm-pruner"
    assert config.pruning.target_sparsity == 0.25
    assert config.recovery is not None
    assert config.recovery.enabled is True
    assert config.recovery.epochs == 2
    assert config.validation is not None
    assert config.validation.tolerance_percent == 5.0
    assert config.deployment is not None
    assert config.deployment.auto_deploy is False


def test_pipeline_config_to_dict() -> None:
    """Test PipelineConfig serialization to dictionary."""
    config = PipelineConfig(
        name="test-config",
        auto_execute=False,
    )
    # Default configs should be created in __post_init__
    assert config.thresholds is not None
    assert config.pruning is not None
    assert config.recovery is not None
    assert config.validation is not None
    assert config.deployment is not None

    data = config.to_dict()
    assert data["name"] == "test-config"
    assert data["auto_execute"] is False
    assert "thresholds" in data
    assert "pruning" in data
    assert "recovery" in data
    assert "validation" in data
    assert "deployment" in data
    assert "quality_prediction" in data


def test_pipeline_runner_dry_run() -> None:
    """Test PipelineRunner with dry run mode."""
    scenario = make_scenario()
    strategy = make_strategy()
    config = PipelineConfig(name="test-runner")

    runner = PipelineRunner(config, dry_run=True)
    result = runner.run(scenario, strategy, grid_co2e=0.35)

    assert result.pipeline_name == "test-runner"
    assert result.scenario_name == "test-scenario"
    assert result.strategy_name == "test-strategy"
    assert result.final_status is not None
    # In dry run mode with positive optimization, should succeed or pass gate
    assert len(result.stages) > 0

    # Check that we have at least Assess stage
    assess_stages = [s for s in result.stages if s.stage.name == "ASSESS"]
    assert len(assess_stages) == 1
    assert assess_stages[0].status.name in ["SUCCESS", "FAILED"]

    # ROI outcome should be set if assess succeeded
    if assess_stages[0].status.name == "SUCCESS":
        assert result.roi_outcome is not None


def test_pipeline_config_from_yaml(tmp_path: Path) -> None:
    """Test loading PipelineConfig from YAML file."""
    yaml_content = """
pipeline:
  name: "yaml-test"
  auto_execute: true
  thresholds:
    max_break_even_months: 8
    min_annual_savings_usd: 8000.0
    max_quality_risk: "medium"
  pruning:
    framework: "wanda"
    target_sparsity: 0.35
    structured: false
"""
    yaml_file = tmp_path / "test-config.yaml"
    yaml_file.write_text(yaml_content)

    config = PipelineConfig.from_yaml(yaml_file)
    assert config.name == "yaml-test"
    assert config.auto_execute is True
    assert config.thresholds is not None
    assert config.thresholds.max_break_even_months == 8
    assert config.pruning is not None
    assert config.pruning.framework == "wanda"
    assert config.pruning.target_sparsity == 0.35
    assert config.pruning.structured is False


def test_execute_external_command_success() -> None:
    """External command helper should report successful execution."""
    result = _execute_external_command(
        stage=PipelineStage.PRUNE,
        command=[sys.executable, "-c", "print('ok')"],
        timeout_seconds=2,
    )
    assert result["succeeded"] is True
    assert result["exit_code"] == 0
    assert result["timed_out"] is False
    assert "ok" in str(result["stdout_excerpt"])


def test_execute_external_command_nonzero_exit() -> None:
    """External command helper should report non-zero exit details."""
    result = _execute_external_command(
        stage=PipelineStage.VALIDATE,
        command=[sys.executable, "-c", "import sys; print('boom', file=sys.stderr); sys.exit(5)"],
        timeout_seconds=2,
    )
    assert result["succeeded"] is False
    assert result["exit_code"] == 5
    assert result["timed_out"] is False
    assert "boom" in str(result["stderr_excerpt"])


def test_execute_external_command_timeout() -> None:
    """External command helper should surface timeout in metadata."""
    result = _execute_external_command(
        stage=PipelineStage.RECOVER,
        command=[sys.executable, "-c", "import time; time.sleep(1.5)"],
        timeout_seconds=1,
    )
    assert result["succeeded"] is False
    assert result["timed_out"] is True
    assert result["exit_code"] == -1


def test_pipeline_runner_dry_run_custom_command_unchanged() -> None:
    """Dry run should skip command execution even when commands are configured."""
    scenario = make_scenario()
    strategy = make_strategy()
    config = PipelineConfig.from_dict(
        {
            "name": "dry-run-command-check",
            "thresholds": {
                "max_break_even_months": 1200,
                "min_annual_savings_usd": -1.0,
                "max_quality_risk": "high",
                "min_expected_quality": 0.0,
            },
            "pruning": {"custom_command": f"{sys.executable} -c \"raise SystemExit(77)\""},
        }
    )
    runner = PipelineRunner(config, dry_run=True)
    result = runner.run(scenario, strategy, grid_co2e=0.35)
    prune_stage = next(stage for stage in result.stages if stage.stage == PipelineStage.PRUNE)
    assert prune_stage.status == StageStatus.SUCCESS
    assert "[DRY RUN]" in prune_stage.message


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
