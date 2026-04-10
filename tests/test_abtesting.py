"""Tests for A/B testing module."""

from __future__ import annotations

import sys
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import Mock

import pytest

# Add src to path for local development
SRC_DIR = Path(__file__).parent.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from atropos.abtesting.models import (  # noqa: E402
    ABTestConfig,
    ExperimentResult,
    ExperimentStatus,
    StatisticalTestType,
    Variant,
    VariantMetrics,
)
from atropos.abtesting.runner import ExperimentRunner, analyze_experiment_results  # noqa: E402
from atropos.abtesting.statistics import (  # noqa: E402
    confidence_interval,
    independent_t_test,
)
from atropos.abtesting.store import ExperimentStore  # noqa: E402
from atropos.deployment.models import (  # noqa: E402
    DeploymentRequest,
    DeploymentResult,
    DeploymentStatus,
)


# Fixtures
@pytest.fixture
def mock_platform() -> Mock:
    """Mock deployment platform."""
    platform = Mock()
    platform.deploy.return_value = DeploymentResult(
        request=Mock(spec=DeploymentRequest),
        deployment_id="test-deployment-123",
        status=DeploymentStatus.SUCCESS,
        message="Deployment successful",
        metrics={"variant_deployments": {}},
    )
    return platform


@pytest.fixture
def sample_variants() -> list[Variant]:
    """Create sample variants for testing."""
    return [
        Variant(
            variant_id="variant-a",
            name="Variant A",
            model_path="/models/variant-a",
            traffic_weight=0.5,
        ),
        Variant(
            variant_id="variant-b",
            name="Variant B",
            model_path="/models/variant-b",
            traffic_weight=0.5,
        ),
    ]


@pytest.fixture
def sample_config(sample_variants: list[Variant]) -> ABTestConfig:
    """Create sample AB test configuration."""
    return ABTestConfig(
        experiment_id="test-experiment-123",
        name="Test Experiment",
        variants=sample_variants,
        primary_metric="throughput_toks_per_sec",
        secondary_metrics=["latency_p95", "error_rate"],
        traffic_allocation=0.1,
        significance_level=0.05,
        statistical_power=0.8,
        test_type=StatisticalTestType.T_TEST,
        min_sample_size_per_variant=1000000,
        max_duration_hours=24.0,
        auto_stop_conditions={},
        deployment_platform="vllm",
        deployment_strategy="immediate",
        health_checks={"http_endpoint": "http://localhost:8000/health"},
        metadata={"environment": "test"},
    )


# Store tests
def test_store_basic_operations(tmp_path: Path) -> None:
    """Test basic ExperimentStore operations."""
    store = ExperimentStore(base_dir=tmp_path)
    experiment_id = "test-exp-1"
    data = {
        "experiment_id": experiment_id,
        "status": "draft",
        "config": {"name": "Test"},
    }

    # Save and load
    store.save_experiment(experiment_id, data)
    loaded = store.load_experiment(experiment_id)
    assert loaded is not None
    assert loaded["experiment_id"] == experiment_id
    assert loaded["status"] == "draft"
    assert "created_at" in loaded
    assert "updated_at" in loaded

    # Update
    data["status"] = "running"
    store.save_experiment(experiment_id, data)
    updated = store.load_experiment(experiment_id)
    assert updated is not None
    assert updated["status"] == "running"

    # Delete
    deleted = store.delete_experiment(experiment_id)
    assert deleted is True
    assert store.load_experiment(experiment_id) is None

    # Delete non-existent
    assert store.delete_experiment("non-existent") is False


def test_store_list_experiments(tmp_path: Path) -> None:
    """Test listing experiments with status filter."""
    store = ExperimentStore(base_dir=tmp_path)

    # Create multiple experiments with different statuses
    for i in range(3):
        data = {
            "experiment_id": f"exp-{i}",
            "status": "running" if i % 2 == 0 else "draft",
            "config": {"name": f"Experiment {i}"},
        }
        store.save_experiment(f"exp-{i}", data)

    # List all
    all_exps = store.list_experiments()
    assert len(all_exps) == 3
    # Should be sorted newest first (by created_at)

    # Filter by status
    running = store.list_experiments(status_filter="running")
    assert len(running) == 2  # exp-0 and exp-2
    assert all(exp["status"] == "running" for exp in running)

    draft = store.list_experiments(status_filter="draft")
    assert len(draft) == 1
    assert draft[0]["status"] == "draft"

    # Invalid filter (case-insensitive)
    filtered = store.list_experiments(status_filter="RUNNING")
    assert len(filtered) == 2


def test_store_save_and_load_config(sample_config: ABTestConfig, tmp_path: Path) -> None:
    """Test saving and loading ABTestConfig."""
    store = ExperimentStore(base_dir=tmp_path)
    store.save_config(sample_config)

    loaded_data = store.load_experiment(sample_config.experiment_id)
    assert loaded_data is not None
    assert loaded_data["type"] == "config"
    assert loaded_data["status"] == ExperimentStatus.DRAFT.value
    assert loaded_data["config"]["experiment_id"] == sample_config.experiment_id
    assert loaded_data["config"]["name"] == sample_config.name
    assert len(loaded_data["config"]["variants"]) == 2


def test_store_save_result(tmp_path: Path) -> None:
    """Test saving ExperimentResult."""
    store = ExperimentStore(base_dir=tmp_path)
    result = ExperimentResult(
        experiment_id="result-test",
        status=ExperimentStatus.COMPLETED,
        start_time="2026-01-01T00:00:00",
        variant_metrics={},
        statistical_results={},
    )
    store.save_result(result)

    loaded_data = store.load_experiment(result.experiment_id)
    assert loaded_data is not None
    assert loaded_data["type"] == "result"
    assert loaded_data["status"] == ExperimentStatus.COMPLETED.value
    assert loaded_data["result"]["experiment_id"] == result.experiment_id


# Statistics tests
def test_independent_t_test_basic() -> None:
    """Test independent t-test with simple data."""
    # Two distinct samples
    sample_a = [1.0, 2.0, 3.0, 4.0, 5.0]
    sample_b = [6.0, 7.0, 8.0, 9.0, 10.0]

    result = independent_t_test(sample_a, sample_b)
    assert "t_statistic" in result
    assert "p_value" in result
    assert "degrees_of_freedom" in result
    assert "method" in result

    # p-value should be small (difference is significant)
    if result["p_value"] is not None:
        assert result["p_value"] < 0.05

    # Insufficient data
    result2 = independent_t_test([1.0], [2.0])
    assert result2["t_statistic"] is None
    assert result2["p_value"] is None
    assert result2["method"] == "insufficient_data"


def test_independent_t_test_equal_samples() -> None:
    """Test t-test with identical samples."""
    sample = [1.0, 2.0, 3.0, 4.0, 5.0]
    result = independent_t_test(sample, sample.copy())
    assert result["t_statistic"] == 0.0 or abs(result["t_statistic"]) < 1e-10
    # p-value should be ~1.0 (no difference)
    if result["p_value"] is not None:
        assert abs(result["p_value"] - 1.0) < 0.1


def test_confidence_interval() -> None:
    """Test confidence interval calculation."""
    sample = [1.0, 2.0, 3.0, 4.0, 5.0]
    ci = confidence_interval(sample, confidence_level=0.95)
    assert ci is not None
    lower, upper = ci
    assert lower < upper
    # Mean is 3.0, should be within interval
    assert lower < 3.0 < upper

    # Insufficient data
    assert confidence_interval([1.0]) is None


# Runner tests
def test_runner_initialization(sample_config: ABTestConfig, mock_platform: Mock) -> None:
    """Test ExperimentRunner initialization."""
    runner = ExperimentRunner(sample_config, mock_platform)
    assert runner.config == sample_config
    assert runner.platform == mock_platform
    assert runner.status == ExperimentStatus.DRAFT
    assert runner.start_time is None
    assert runner.end_time is None
    assert runner.deployment_ids == {}


def test_runner_start_success(sample_config: ABTestConfig, mock_platform: Mock) -> None:
    """Test starting an experiment."""
    runner = ExperimentRunner(sample_config, mock_platform)
    result = runner.start()

    assert result.experiment_id == sample_config.experiment_id
    assert result.status == ExperimentStatus.RUNNING
    assert result.start_time is not None
    assert result.variant_metrics == {}
    assert result.statistical_results == {}

    # Platform should have been called twice (once per variant)
    assert mock_platform.deploy.call_count == 2

    # Runner state updated
    assert runner.status == ExperimentStatus.RUNNING
    assert runner.start_time == result.start_time
    assert runner.deployment_ids is not None


def test_runner_cannot_start_twice(sample_config: ABTestConfig, mock_platform: Mock) -> None:
    """Test that experiment cannot be started twice."""
    runner = ExperimentRunner(sample_config, mock_platform)
    runner.start()

    with pytest.raises(RuntimeError, match="Cannot start experiment in status"):
        runner.start()


def test_runner_stop(sample_config: ABTestConfig, mock_platform: Mock) -> None:
    """Test stopping an experiment."""
    runner = ExperimentRunner(sample_config, mock_platform)
    runner.start()

    # Mock platform.undeploy
    mock_platform.undeploy = Mock(
        return_value=DeploymentResult(
            request=Mock(spec=DeploymentRequest),
            deployment_id="test-deployment-123",
            status=DeploymentStatus.SUCCESS,
            message="Undeployed",
        )
    )

    result = runner.stop(reason="test")
    assert result.status == ExperimentStatus.STOPPED
    assert result.end_time is not None

    # Runner state updated
    assert runner.status == ExperimentStatus.STOPPED
    assert runner.end_time == result.end_time


def test_runner_pause_resume(sample_config: ABTestConfig, mock_platform: Mock) -> None:
    """Test pausing and resuming an experiment."""
    runner = ExperimentRunner(sample_config, mock_platform)
    runner.start()

    # Mock platform pause_experiment and resume_experiment
    mock_platform.pause_experiment = Mock()
    mock_platform.resume_experiment = Mock()

    # Pause experiment
    runner.pause()
    assert runner.status == ExperimentStatus.PAUSED
    mock_platform.pause_experiment.assert_called_once_with(
        sample_config.experiment_id, runner.deployment_ids
    )

    # Resume experiment
    runner.resume()
    assert runner.status == ExperimentStatus.RUNNING
    mock_platform.resume_experiment.assert_called_once_with(
        sample_config.experiment_id, runner.deployment_ids
    )

    # Stop experiment to clean up monitoring thread
    mock_platform.undeploy = Mock(
        return_value=DeploymentResult(
            request=Mock(spec=DeploymentRequest),
            deployment_id="test-deployment-123",
            status=DeploymentStatus.SUCCESS,
            message="Undeployed",
        )
    )
    runner.stop(reason="test")


def test_runner_pause_without_platform_support(
    sample_config: ABTestConfig, mock_platform: Mock
) -> None:
    """Test pausing when platform doesn't support traffic pausing."""
    runner = ExperimentRunner(sample_config, mock_platform)
    runner.start()

    # Platform does not have pause_experiment method
    # Should still update status and log warning
    runner.pause()
    assert runner.status == ExperimentStatus.PAUSED

    # Resume without platform support
    runner.resume()
    assert runner.status == ExperimentStatus.RUNNING

    # Stop experiment to clean up monitoring thread
    mock_platform.undeploy = Mock(
        return_value=DeploymentResult(
            request=Mock(spec=DeploymentRequest),
            deployment_id="test-deployment-123",
            status=DeploymentStatus.SUCCESS,
            message="Undeployed",
        )
    )
    runner.stop(reason="test")


def test_analyze_experiment_results_prefers_raw_samples() -> None:
    """Test analysis path that uses raw observations when available."""
    config = ABTestConfig(
        experiment_id="raw-analysis-test",
        name="Raw Analysis Test",
        variants=[
            Variant(variant_id="control", name="Control", model_path="/models/control"),
            Variant(variant_id="treatment", name="Treatment", model_path="/models/treatment"),
        ],
        primary_metric="throughput_toks_per_sec",
        test_type=StatisticalTestType.T_TEST,
    )
    variant_metrics = {
        "control": VariantMetrics(
            variant_id="control",
            sample_count=4,
            metrics={"throughput_toks_per_sec": {"mean": 100.0, "std": 5.0, "count": 4}},
            raw_observations={"throughput_toks_per_sec": [97.0, 99.0, 101.0, 103.0]},
        ),
        "treatment": VariantMetrics(
            variant_id="treatment",
            sample_count=4,
            metrics={"throughput_toks_per_sec": {"mean": 120.0, "std": 5.0, "count": 4}},
            raw_observations={"throughput_toks_per_sec": [117.0, 119.0, 121.0, 123.0]},
        ),
    }

    results = analyze_experiment_results(variant_metrics, config)
    assert len(results) == 1
    result = next(iter(results.values()))
    assert result.metadata["analysis_mode"] == "raw_observations"
    assert result.metadata["warnings"] == []
    assert result.sample_sizes == {"control": 4, "treatment": 4}


def test_analyze_experiment_results_aggregate_fallback_warns() -> None:
    """Test aggregate-only fallback path when raw observations are unavailable."""
    config = ABTestConfig(
        experiment_id="aggregate-analysis-test",
        name="Aggregate Analysis Test",
        variants=[
            Variant(variant_id="control", name="Control", model_path="/models/control"),
            Variant(variant_id="treatment", name="Treatment", model_path="/models/treatment"),
        ],
        primary_metric="throughput_toks_per_sec",
        test_type=StatisticalTestType.T_TEST,
    )
    variant_metrics = {
        "control": VariantMetrics(
            variant_id="control",
            sample_count=3,
            metrics={"throughput_toks_per_sec": {"mean": 100.0, "std": 0.0, "count": 3}},
        ),
        "treatment": VariantMetrics(
            variant_id="treatment",
            sample_count=3,
            metrics={"throughput_toks_per_sec": {"mean": 110.0, "std": 0.0, "count": 3}},
        ),
    }

    results = analyze_experiment_results(variant_metrics, config)
    assert len(results) == 1
    result = next(iter(results.values()))
    assert result.metadata["analysis_mode"] == "aggregate_only_fallback"
    assert len(result.metadata["warnings"]) == 1
    assert "Aggregate-only fallback used" in result.metadata["warnings"][0]
    assert result.metadata["data_sources"] == {
        "control": "aggregated_mean_repeated",
        "treatment": "aggregated_mean_repeated",
    }


def test_auto_stop_does_not_stop_at_min_sample_size_alone(mock_platform: Mock) -> None:
    """Test auto-stop requires more than just minimum sample size."""
    config = ABTestConfig(
        experiment_id="autostop-min-sample-test",
        name="Auto-stop Min Sample Test",
        variants=[
            Variant(variant_id="control", name="Control", model_path="/models/control"),
            Variant(variant_id="treatment", name="Treatment", model_path="/models/treatment"),
        ],
        primary_metric="throughput_toks_per_sec",
        min_sample_size_per_variant=10,
        max_duration_hours=48.0,
        auto_stop_conditions={},
    )
    runner = ExperimentRunner(config, mock_platform)
    runner._variant_metrics = {
        "control": VariantMetrics(
            variant_id="control",
            sample_count=12,
            metrics={"throughput_toks_per_sec": {"mean": 100.0, "std": 10.0, "count": 12}},
        ),
        "treatment": VariantMetrics(
            variant_id="treatment",
            sample_count=14,
            metrics={"throughput_toks_per_sec": {"mean": 101.0, "std": 10.0, "count": 14}},
        ),
    }
    runner._start_time = (datetime.now() - timedelta(hours=1)).isoformat()
    runner._statistical_results = {}

    should_stop, reason = runner._check_auto_stop_conditions()
    assert should_stop is False
    assert reason in {
        "No statistical results available yet",
        "Statistical target not reached yet",
    }


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
