"""Tests for visualization module."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from atropos.visualization import (
    ReportType,
    detect_report_type,
    load_framework_comparison,
    load_pruning_report,
    load_tradeoff_analysis,
    load_validation_report,
    visualize_json,
)


# Sample data fixtures
def sample_framework_comparison_data() -> dict[str, Any]:
    """Sample framework comparison JSON data."""
    return {
        "total_tests": 3,
        "successful": 2,
        "failed": 1,
        "duration_sec": 120.5,
        "start_time": "2024-01-01T10:00:00",
        "end_time": "2024-01-01T10:02:00",
        "config": {"target_sparsity": 0.3, "models": ["gpt2", "bloom-560m"]},
        "results": [
            {
                "model": "gpt2",
                "framework": "magnitude",
                "status": "success",
                "target_sparsity": 0.3,
                "achieved_sparsity": 0.28,
                "original_params": 124000000,
                "pruned_params": 89280000,
                "pruning_time_sec": 45.2,
                "output_path": "/tmp/gpt2_magnitude",
                "error_message": "",
                "timestamp": "2024-01-01T10:00:30",
            },
            {
                "model": "gpt2",
                "framework": "wanda",
                "status": "success",
                "target_sparsity": 0.3,
                "achieved_sparsity": 0.31,
                "original_params": 124000000,
                "pruned_params": 85560000,
                "pruning_time_sec": 60.1,
                "output_path": "/tmp/gpt2_wanda",
                "error_message": "",
                "timestamp": "2024-01-01T10:01:30",
            },
            {
                "model": "bloom-560m",
                "framework": "magnitude",
                "status": "failed",
                "target_sparsity": 0.3,
                "achieved_sparsity": 0.0,
                "original_params": 560000000,
                "pruned_params": 560000000,
                "pruning_time_sec": 0.0,
                "output_path": "",
                "error_message": "Unsupported architecture",
                "timestamp": "2024-01-01T10:02:00",
            },
        ],
    }


def sample_tradeoff_analysis_data() -> dict[str, Any]:
    """Sample trade-off analysis JSON data."""
    return {
        "total_tests": 2,
        "successful": 2,
        "failed": 0,
        "duration_sec": 300.0,
        "start_time": "2024-01-01T11:00:00",
        "end_time": "2024-01-01T11:05:00",
        "config": {"target_sparsity": 0.3, "quality_metric": "perplexity"},
        "results": [
            {
                "model": "gpt2",
                "framework": "magnitude",
                "status": "success",
                "target_sparsity": 0.3,
                "achieved_sparsity": 0.28,
                "original_params": 124000000,
                "pruned_params": 89280000,
                "pruning_time_sec": 45.2,
                "pruning_output_path": "/tmp/gpt2_magnitude",
                "baseline_perplexity": 25.4,
                "pruned_perplexity": 26.8,
                "perplexity_change_pct": 5.5,
                "baseline_completion_score": 0.85,
                "pruned_completion_score": 0.82,
                "completion_score_change_pct": -3.5,
                "baseline_inference_time_ms": 120.0,
                "pruned_inference_time_ms": 105.0,
                "inference_speedup_pct": 12.5,
                "quality_speed_ratio": 0.44,
                "error_message": "",
                "timestamp": "2024-01-01T11:01:00",
            },
            {
                "model": "bloom-560m",
                "framework": "wanda",
                "status": "success",
                "target_sparsity": 0.3,
                "achieved_sparsity": 0.31,
                "original_params": 560000000,
                "pruned_params": 386400000,
                "pruning_time_sec": 120.5,
                "pruning_output_path": "/tmp/bloom_wanda",
                "baseline_perplexity": 18.2,
                "pruned_perplexity": 19.1,
                "perplexity_change_pct": 4.9,
                "baseline_completion_score": 0.88,
                "pruned_completion_score": 0.86,
                "completion_score_change_pct": -2.3,
                "baseline_inference_time_ms": 350.0,
                "pruned_inference_time_ms": 310.0,
                "inference_speedup_pct": 11.4,
                "quality_speed_ratio": 0.43,
                "error_message": "",
                "timestamp": "2024-01-01T11:04:00",
            },
        ],
    }


def sample_pruning_report_data() -> dict[str, Any]:
    """Sample pruning report JSON data."""
    return {
        "total_models": 2,
        "successful": 2,
        "failed": 0,
        "duration_sec": 180.0,
        "output_dir": "/tmp/pruned_models",
        "start_time": "2024-01-01T12:00:00",
        "end_time": "2024-01-01T12:03:00",
        "results": [
            {
                "model_id": "gpt2",
                "strategy": "structured_pruning",
                "status": "success",
                "original_params": 124000000,
                "pruned_params": 99200000,
                "target_sparsity": 0.2,
                "actual_sparsity": 0.2,
                "pruning_time_sec": 30.5,
                "output_path": "/tmp/pruned_models/gpt2",
                "error_message": "",
                "timestamp": "2024-01-01T12:01:00",
            },
            {
                "model_id": "bloom-560m",
                "strategy": "magnitude_pruning",
                "status": "success",
                "original_params": 560000000,
                "pruned_params": 448000000,
                "target_sparsity": 0.2,
                "actual_sparsity": 0.2,
                "pruning_time_sec": 90.2,
                "output_path": "/tmp/pruned_models/bloom",
                "error_message": "",
                "timestamp": "2024-01-01T12:02:30",
            },
        ],
    }


def sample_validation_report_data() -> dict[str, Any]:
    """Sample validation report JSON data."""
    return {
        "summary": {"passed": 2, "failed": 1, "total": 3},
        "criteria": {
            "perplexity_tolerance_pct": 20.0,
            "generation_similarity_threshold": 0.7,
        },
        "results": [
            {
                "model": "gpt2",
                "strategy": "structured_pruning",
                "original_path": "/tmp/original/gpt2",
                "pruned_path": "/tmp/pruned/gpt2",
                "passed": True,
                "metrics": {
                    "perplexity_increase_pct": 5.2,
                    "avg_generation_similarity": 0.85,
                },
                "errors": [],
                "perplexity_passed": True,
                "generation_passed": True,
            },
            {
                "model": "bloom-560m",
                "strategy": "magnitude_pruning",
                "original_path": "/tmp/original/bloom",
                "pruned_path": "/tmp/pruned/bloom",
                "passed": True,
                "metrics": {
                    "perplexity_increase_pct": 12.3,
                    "avg_generation_similarity": 0.72,
                },
                "errors": [],
                "perplexity_passed": True,
                "generation_passed": True,
            },
            {
                "model": "opt-125m",
                "strategy": "structured_pruning",
                "original_path": "/tmp/original/opt",
                "pruned_path": "/tmp/pruned/opt",
                "passed": False,
                "metrics": {
                    "perplexity_increase_pct": 25.5,
                    "avg_generation_similarity": 0.45,
                },
                "errors": ["perplexity increase too high"],
                "perplexity_passed": False,
                "generation_passed": False,
            },
        ],
    }


# Detection tests
def test_detect_report_type_framework_comparison() -> None:
    """Test detection of framework comparison report."""
    data = sample_framework_comparison_data()
    assert detect_report_type(data) == ReportType.FRAMEWORK_COMPARISON


def test_detect_report_type_tradeoff_analysis() -> None:
    """Test detection of trade-off analysis report."""
    data = sample_tradeoff_analysis_data()
    assert detect_report_type(data) == ReportType.TRADEOFF_ANALYSIS


def test_detect_report_type_pruning_report() -> None:
    """Test detection of pruning report."""
    data = sample_pruning_report_data()
    assert detect_report_type(data) == ReportType.PRUNING_REPORT


def test_detect_report_type_validation_report() -> None:
    """Test detection of validation report."""
    data = sample_validation_report_data()
    assert detect_report_type(data) == ReportType.VALIDATION_REPORT


def test_detect_report_type_unknown() -> None:
    """Test detection of unknown report type."""
    assert detect_report_type({}) == ReportType.UNKNOWN


# Load tests
def test_load_framework_comparison() -> None:
    """Test loading framework comparison data."""
    data = sample_framework_comparison_data()
    report = load_framework_comparison(data)
    assert report.total_tests == 3
    assert report.successful == 2
    assert report.failed == 1
    assert len(report.results) == 3
    assert report.results[0].model == "gpt2"
    assert report.results[0].framework == "magnitude"
    assert report.results[0].status == "success"
    assert report.results[0].parameter_reduction_fraction == pytest.approx(
        (124000000 - 89280000) / 124000000
    )


def test_load_tradeoff_analysis() -> None:
    """Test loading trade-off analysis data."""
    data = sample_tradeoff_analysis_data()
    report = load_tradeoff_analysis(data)
    assert report.total_tests == 2
    assert report.successful == 2
    assert len(report.results) == 2
    assert report.results[0].model == "gpt2"
    assert report.results[0].framework == "magnitude"
    assert report.results[0].perplexity_change_pct == 5.5
    assert report.results[0].quality_degradation_pct == pytest.approx((5.5 + (-3.5)) / 2)


def test_load_pruning_report() -> None:
    """Test loading pruning report data."""
    data = sample_pruning_report_data()
    report = load_pruning_report(data)
    assert report.total_models == 2
    assert report.successful == 2
    assert len(report.results) == 2
    assert report.results[0].model_id == "gpt2"
    assert report.results[0].strategy == "structured_pruning"
    assert report.results[0].target_sparsity == 0.2


def test_load_validation_report() -> None:
    """Test loading validation report data."""
    data = sample_validation_report_data()
    report = load_validation_report(data)
    assert report.summary["passed"] == 2
    assert report.summary["failed"] == 1
    assert len(report.results) == 3
    assert report.results[0].model == "gpt2"
    assert report.results[0].strategy == "structured_pruning"
    assert report.results[0].passed is True


# Figure creation tests (with mocked Plotly)
@pytest.fixture
def mock_plotly():
    """Mock Plotly modules."""
    with (
        patch("atropos.visualization.go", MagicMock()) as mock_go,
        patch("atropos.visualization.make_subplots", MagicMock()) as mock_make_subplots,
    ):
        # Create mock Figure class
        mock_fig = MagicMock()
        mock_go.Figure.return_value = mock_fig
        mock_go.Bar = MagicMock()
        mock_go.Scatter = MagicMock()
        mock_go.Heatmap = MagicMock()
        mock_go.Pie = MagicMock()
        mock_go.Figure = MagicMock(return_value=mock_fig)
        mock_make_subplots.return_value = mock_fig
        yield {
            "go": mock_go,
            "make_subplots": mock_make_subplots,
            "fig": mock_fig,
        }


def test_create_framework_comparison_figures(mock_plotly):
    """Test creation of framework comparison figures."""
    from atropos.visualization import (
        create_framework_comparison_figures,
        load_framework_comparison,
    )

    data = sample_framework_comparison_data()
    report = load_framework_comparison(data)
    figures = create_framework_comparison_figures(report)
    # Should return dict with figure names
    assert isinstance(figures, dict)
    # Since Plotly is mocked, figures dict may be empty
    # We just ensure no exception


def test_create_tradeoff_analysis_figures(mock_plotly):
    """Test creation of trade-off analysis figures."""
    from atropos.visualization import (
        create_tradeoff_analysis_figures,
        load_tradeoff_analysis,
    )

    data = sample_tradeoff_analysis_data()
    report = load_tradeoff_analysis(data)
    figures = create_tradeoff_analysis_figures(report)
    assert isinstance(figures, dict)


def test_create_pruning_report_figures(mock_plotly):
    """Test creation of pruning report figures."""
    from atropos.visualization import (
        create_pruning_report_figures,
        load_pruning_report,
    )

    data = sample_pruning_report_data()
    report = load_pruning_report(data)
    figures = create_pruning_report_figures(report)
    assert isinstance(figures, dict)


def test_create_validation_report_figures(mock_plotly):
    """Test creation of validation report figures."""
    from atropos.visualization import (
        create_validation_report_figures,
        load_validation_report,
    )

    data = sample_validation_report_data()
    report = load_validation_report(data)
    figures = create_validation_report_figures(report)
    assert isinstance(figures, dict)


# Integration test for visualize_json
def test_visualize_json_framework_comparison(tmp_path: Path):
    """Test visualize_json with framework comparison data."""
    data = sample_framework_comparison_data()
    input_file = tmp_path / "input.json"
    output_dir = tmp_path / "output"
    with open(input_file, "w", encoding="utf-8") as f:
        json.dump(data, f)

    # Mock Plotly to avoid actual dependency
    with (
        patch("atropos.visualization.go", None),
        patch("atropos.visualization.make_subplots", None),
    ):
        visualize_json(input_file, output_dir, formats=["html"])
        # Should print warning about Plotly not installed
        # No exception expected


def test_visualize_json_with_explicit_report_type(tmp_path: Path):
    """Test visualize_json with explicit report type."""
    data = sample_framework_comparison_data()
    input_file = tmp_path / "input.json"
    output_dir = tmp_path / "output"
    with open(input_file, "w", encoding="utf-8") as f:
        json.dump(data, f)

    with (
        patch("atropos.visualization.go", None),
        patch("atropos.visualization.make_subplots", None),
    ):
        visualize_json(
            input_file,
            output_dir,
            formats=["html"],
            report_type="framework-comparison",
        )


def test_visualize_json_file_not_found():
    """Test visualize_json with non-existent file."""
    with pytest.raises(FileNotFoundError):
        visualize_json(Path("/nonexistent.json"), Path("."))


def test_visualize_json_unknown_report_type(tmp_path: Path):
    """Test visualize_json with unknown report type."""
    input_file = tmp_path / "input.json"
    with open(input_file, "w", encoding="utf-8") as f:
        json.dump({"invalid": "data"}, f)

    with pytest.raises(ValueError, match="Could not detect report type"):
        visualize_json(input_file, tmp_path)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
