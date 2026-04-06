"""Tests for CLI functionality."""

import logging
from pathlib import Path
from unittest.mock import patch

import pytest

from atropos.cli import main


def test_list_presets_runs(capsys) -> None:  # type: ignore[no-untyped-def]
    """Test that list-presets command runs successfully."""
    result = main(["list-presets"])
    captured = capsys.readouterr()
    assert result == 0
    assert "medium-coder" in captured.out
    assert "structured_pruning" in captured.out


def test_preset_json_runs(capsys) -> None:  # type: ignore[no-untyped-def]
    """Test that preset command with JSON report runs successfully."""
    result = main(["preset", "medium-coder", "--report", "json"])
    captured = capsys.readouterr()
    assert result == 0
    assert '"scenario_name": "medium-coder"' in captured.out


def test_compare_markdown_writes_file(tmp_path: Path) -> None:
    """Test that compare command writes markdown output to file."""
    out = tmp_path / "compare.md"
    result = main(
        [
            "compare",
            "medium-coder",
            "--strategies",
            "mild_pruning",
            "structured_pruning",
            "--format",
            "markdown",
            "--output",
            str(out),
        ]
    )
    assert result == 0
    assert out.exists()
    assert "| Strategy |" in out.read_text()


def test_cli_verbose_flag(capsys: pytest.CaptureFixture) -> None:
    """Test that --verbose flag enables INFO logging."""
    # Clear existing handlers to avoid interference
    logger = logging.getLogger("atropos")
    original_handlers = logger.handlers.copy()
    logger.handlers.clear()

    try:
        result = main(["--verbose", "preset", "medium-coder"])
        assert result == 0
        # With verbose flag, logger should be at INFO level
        assert logger.level == logging.INFO
    finally:
        # Restore original handlers
        logger.handlers = original_handlers


def test_cli_debug_flag(capsys: pytest.CaptureFixture) -> None:
    """Test that --debug flag enables DEBUG logging."""
    logger = logging.getLogger("atropos")
    original_handlers = logger.handlers.copy()
    logger.handlers.clear()

    try:
        result = main(["--debug", "preset", "medium-coder"])
        assert result == 0
        assert logger.level == logging.DEBUG
    finally:
        logger.handlers = original_handlers


def test_cli_traceback_flag() -> None:
    """Test that --traceback flag sets SHOW_TRACEBACK."""
    import atropos.logging_config as config_module

    original_value = config_module.SHOW_TRACEBACK
    config_module.SHOW_TRACEBACK = False

    try:
        result = main(["--traceback", "preset", "medium-coder"])
        assert result == 0
        # Should be True after setup_logging with traceback=True
        assert config_module.SHOW_TRACEBACK is True
    finally:
        config_module.SHOW_TRACEBACK = original_value


def test_tune_command_basic(capsys: pytest.CaptureFixture) -> None:
    """Test that tune command runs successfully."""
    # Mock the HyperparameterTuner to avoid actual optimization
    with patch("atropos.cli.HyperparameterTuner") as mock_tuner_class:
        mock_tuner = mock_tuner_class.return_value
        mock_result = mock_tuner.tune.return_value

        # Set up mock result attributes needed for CLI output
        mock_result.strategy.name = "tuned_test"
        mock_result.strategy.parameter_reduction_fraction = 0.25
        mock_result.strategy.memory_reduction_fraction = 0.2
        mock_result.strategy.throughput_improvement_fraction = 0.15
        mock_result.strategy.power_reduction_fraction = 0.1
        mock_result.strategy.quality_risk = "medium"
        mock_result.recommended_framework = "wanda"
        mock_result.quality_risk = "medium"
        mock_result.expected_memory_gb = 12.5
        mock_result.expected_throughput_toks_per_sec = 28.0
        mock_result.expected_latency_ms_per_request = 53.6
        mock_result.expected_power_watts = 270.0
        mock_result.tuning_metadata = {"architecture": "llama"}

        result = main(["tune", "medium-coder"])
        assert result == 0

        captured = capsys.readouterr()
        assert "Hyperparameter Tuning Results" in captured.out
        assert "tuned_test" in captured.out
        assert "wanda" in captured.out

        mock_tuner_class.assert_called_once()
        mock_tuner.tune.assert_called_once()


def test_tune_command_with_verbose(capsys: pytest.CaptureFixture) -> None:
    """Test tune command with verbose flag."""
    with patch("atropos.cli.HyperparameterTuner") as mock_tuner_class:
        mock_tuner = mock_tuner_class.return_value
        mock_result = mock_tuner.tune.return_value

        # Set up mock result attributes needed for CLI output formatting
        mock_result.strategy.name = "tuned_test"
        mock_result.strategy.parameter_reduction_fraction = 0.25
        mock_result.strategy.memory_reduction_fraction = 0.2
        mock_result.strategy.throughput_improvement_fraction = 0.15
        mock_result.strategy.power_reduction_fraction = 0.1
        mock_result.strategy.quality_risk = "medium"
        mock_result.recommended_framework = "wanda"
        mock_result.quality_risk = "medium"
        mock_result.expected_memory_gb = 12.5
        mock_result.expected_throughput_toks_per_sec = 28.0
        mock_result.expected_latency_ms_per_request = 53.6
        mock_result.expected_power_watts = 270.0
        mock_result.tuning_metadata = {"architecture": "llama"}

        result = main(["--verbose", "tune", "medium-coder"])
        assert result == 0

        captured = capsys.readouterr()
        # Should have some output (not empty)
        assert captured.out.strip() != ""


def test_tune_command_with_constraints(capsys: pytest.CaptureFixture) -> None:
    """Test tune command with constraint flags."""
    with patch("atropos.cli.HyperparameterTuner") as mock_tuner_class:
        mock_tuner = mock_tuner_class.return_value
        mock_result = mock_tuner.tune.return_value
        mock_result.strategy.name = "tuned_constrained"
        mock_result.strategy.parameter_reduction_fraction = 0.25
        mock_result.strategy.memory_reduction_fraction = 0.2
        mock_result.strategy.throughput_improvement_fraction = 0.15
        mock_result.strategy.power_reduction_fraction = 0.1
        mock_result.strategy.quality_risk = "medium"
        mock_result.recommended_framework = "wanda"
        mock_result.quality_risk = "medium"
        mock_result.expected_memory_gb = 12.5
        mock_result.expected_throughput_toks_per_sec = 28.0
        mock_result.expected_latency_ms_per_request = 53.6
        mock_result.expected_power_watts = 270.0
        mock_result.tuning_metadata = {"architecture": "llama"}

        result = main(
            [
                "tune",
                "medium-coder",
                "--max-memory",
                "10.0",
                "--min-throughput",
                "30.0",
                "--max-latency",
                "100.0",
                "--max-power",
                "250.0",
                "--use-quantization",
                "--fast-pruning",
            ]
        )
        assert result == 0

        # Check that constraints were passed to tuner
        args, kwargs = mock_tuner_class.call_args
        # Constructor accepts scenario, model_characteristics=None, constraints=None
        # CLI passes keyword arguments: scenario=..., constraints=...
        constraints = kwargs.get("constraints")
        if constraints is None and len(args) > 2:
            constraints = args[2]

        assert constraints is not None
        assert constraints.max_memory_gb == 10.0
        assert constraints.min_throughput_toks_per_sec == 30.0
        assert constraints.max_latency_ms_per_request == 100.0
        assert constraints.max_power_watts == 250.0
        assert constraints.use_quantization is True
        assert constraints.prefer_fast_pruning is True


def test_cloud_pricing_list_providers(capsys: pytest.CaptureFixture) -> None:
    """Test cloud-pricing list-providers command."""
    result = main(["cloud-pricing", "list-providers"])
    assert result == 0
    captured = capsys.readouterr()
    assert "aws" in captured.out


def test_cloud_pricing_estimate(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    """Test cloud-pricing estimate command."""
    scenario = tmp_path / "scenario.yaml"
    scenario.write_text(
        """
name: cloud-cli-test
parameters_b: 34
memory_gb: 20
throughput_toks_per_sec: 40
power_watts: 300
requests_per_day: 20000
tokens_per_request: 1200
electricity_cost_per_kwh: 0.15
one_time_project_cost_usd: 15000
deployment:
  platform: aws
  instance_type: p4d.24xlarge
  purchase_option: spot
  region: us-east-1
monthly_runtime_hours: 100
"""
    )
    result = main(["cloud-pricing", "estimate", "--scenario", str(scenario)])
    assert result == 0
    captured = capsys.readouterr()
    assert "Monthly" in captured.out




def test_cloud_pricing_estimate_reserved_shows_buyout(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    """Test reserved estimate prints commitment buyout."""
    scenario = tmp_path / "scenario_reserved.yaml"
    scenario.write_text(
        """
name: cloud-cli-reserved
parameters_b: 34
memory_gb: 20
throughput_toks_per_sec: 40
power_watts: 300
requests_per_day: 20000
tokens_per_request: 1200
electricity_cost_per_kwh: 0.15
one_time_project_cost_usd: 15000
deployment:
  platform: azure
  instance_type: Standard_NC24_A100_v2
  purchase_option: reserved
  commitment_years: 1
monthly_runtime_hours: 100
"""
    )
    result = main(
        [
            "cloud-pricing",
            "estimate",
            "--scenario",
            str(scenario),
            "--fetch-live-pricing",
            "--mock-pricing-api",
        ]
    )
    assert result == 0
    captured = capsys.readouterr()
    assert "Commitment buyout" in captured.out


def test_cloud_pricing_compare(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    """Test cloud-pricing compare command."""
    scenario = tmp_path / "scenario.yaml"
    scenario.write_text(
        """
name: cloud-compare-test
parameters_b: 34
memory_gb: 20
throughput_toks_per_sec: 40
power_watts: 300
requests_per_day: 20000
tokens_per_request: 1200
electricity_cost_per_kwh: 0.15
one_time_project_cost_usd: 15000
deployment:
  platform: aws
  instance_type: p4d.24xlarge
  purchase_option: ondemand
"""
    )
    result = main(
        [
            "cloud-pricing",
            "compare",
            "--scenario",
            str(scenario),
            "--providers",
            "aws,azure",
        ]
    )
    assert result == 0
    captured = capsys.readouterr()
    assert "Provider comparison" in captured.out
