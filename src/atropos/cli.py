"""Command-line interface for Atropos."""

from __future__ import annotations

import argparse
import dataclasses
import logging
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any, cast

import yaml  # type: ignore[import-untyped]

from .abtesting.models import (
    ABTestConfig,
    ExperimentStatus,
    StatisticalTestType,
    Variant,
    VariantMetrics,
)
from .abtesting.runner import (
    ExperimentRunner,
    analyze_experiment_results,
)
from .abtesting.store import get_default_store
from .batch import BatchExecutionReport, batch_process
from .calculations import combine_strategies, estimate_outcome
from .calibration import calibrate_scenario, generate_calibration_report
from .carbon_presets import CARBON_PRESETS, get_carbon_intensity, list_regions
from .config import AtroposConfig
from .core.calculator import ROICalculator
from .core.uncertainty import ParameterDistribution
from .costs.cloud_pricing import CloudPricingEngine, request_from_scenario_yaml
from .deployment.platforms import get_platform
from .exceptions import AtroposError
from .integrations import TRACKERS, get_tracker, run_to_scenario
from .io import csv_to_markdown, export_to_csv, load_scenario, render_report
from .logging_config import SHOW_TRACEBACK, setup_logging
from .model_tester import (
    generate_catalog,
    get_recommended_test_models,
    run_test_suite,
)
from .models import DeploymentScenario
from .pipeline import PipelineConfig, run_pipeline
from .presets import QUANTIZATION_BONUS, SCENARIOS, STRATEGIES
from .pruning.manager import setup_pruning_environment, test_pruning_framework
from .reporting import generate_comparison_json, generate_comparison_table
from .telemetry import (
    PARSERS,
    get_parser,
    telemetry_to_scenario,
    validate_telemetry,
)
from .telemetry_collector import (
    CollectionConfig,
    collect_and_save,
)
from .tuning import HyperparameterTuner, TuningConstraints, TuningResult
from .visualization import visualize_json


def build_parser() -> argparse.ArgumentParser:
    """Build and configure the argument parser for the CLI."""
    parser = argparse.ArgumentParser(
        prog="atropos-llm",
        description=(
            "Estimate the ROI of pruning and related optimizations for coding LLM deployments."
        ),
    )
    # Global logging and debugging flags
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose output (INFO level logging)",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug output (DEBUG level logging)",
    )
    parser.add_argument(
        "--traceback",
        action="store_true",
        help="Show full traceback for exceptions",
    )
    parser.add_argument(
        "--log-file",
        type=Path,
        help="Write logs to specified file (overrides ATROPOS_LOG_FILE)",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("list-presets", help="List built-in deployment and strategy presets.")

    preset_parser = subparsers.add_parser("preset", help="Run a built-in scenario preset.")
    preset_parser.add_argument("name", choices=sorted(SCENARIOS.keys()))
    _add_strategy_args(preset_parser)
    preset_parser.add_argument(
        "--region", help="Region for carbon intensity (ISO code or cloud region like us-east-1)"
    )

    scenario_parser = subparsers.add_parser("scenario", help="Run a scenario from a YAML file.")
    scenario_parser.add_argument("path", type=Path)
    _add_strategy_args(scenario_parser)
    scenario_parser.add_argument(
        "--region", help="Region for carbon intensity (ISO code or cloud region like us-east-1)"
    )

    compare_parser = subparsers.add_parser("compare", help="Compare multiple strategies.")
    compare_parser.add_argument("scenario", help="Scenario name or path to YAML")
    compare_parser.add_argument(
        "--strategies", nargs="+", required=True, choices=sorted(STRATEGIES.keys())
    )
    compare_parser.add_argument("--with-quantization", action="store_true")
    compare_parser.add_argument("--format", choices=["text", "markdown", "json"], default="text")
    compare_parser.add_argument(
        "--sort-by", choices=["savings", "breakeven", "risk"], default="savings"
    )
    compare_parser.add_argument("--ascending", action="store_true", help="Sort in ascending order")
    compare_parser.add_argument("--output", "-o", type=Path)
    compare_parser.add_argument(
        "--region", help="Region for carbon intensity (ISO code or cloud region like us-east-1)"
    )

    batch_parser = subparsers.add_parser("batch", help="Process multiple scenario files.")
    batch_parser.add_argument("directory", type=Path)
    batch_parser.add_argument(
        "--strategies", nargs="+", required=True, choices=sorted(STRATEGIES.keys())
    )
    batch_parser.add_argument("--with-quantization", action="store_true")
    batch_parser.add_argument("--output", "-o", type=Path, required=True)
    batch_parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop batch processing on first error (default: continue).",
    )
    batch_parser.add_argument(
        "--max-errors",
        type=int,
        default=None,
        help="Stop after N errors (default: unlimited).",
    )
    batch_parser.add_argument(
        "--error-log",
        type=Path,
        default=None,
        help="Optional JSON file for detailed error entries.",
    )
    batch_parser.add_argument(
        "--retry-attempts",
        type=int,
        default=3,
        help="Retry attempts for recoverable errors (default: 3).",
    )
    batch_parser.add_argument(
        "--scenario-timeout-seconds",
        type=int,
        default=600,
        help="Per-scenario timeout in seconds (default: 600).",
    )
    batch_parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=5,
        help="Persist checkpoint every N processed rows (default: 5).",
    )
    batch_parser.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="Resume from an existing batch CSV output file.",
    )

    sensitivity_parser = subparsers.add_parser("sensitivity", help="Run sensitivity analysis.")
    sensitivity_parser.add_argument("scenario", help="Scenario name or path to YAML")
    sensitivity_parser.add_argument("--strategy", required=True, choices=sorted(STRATEGIES.keys()))
    sensitivity_parser.add_argument(
        "--param",
        required=True,
        choices=[
            "memory_reduction_fraction",
            "throughput_improvement_fraction",
            "power_reduction_fraction",
        ],
    )
    sensitivity_parser.add_argument("--variations", type=int, default=5)
    sensitivity_parser.add_argument("--step", type=float, default=0.1)
    sensitivity_parser.add_argument("--output", "-o", type=Path)
    sensitivity_parser.add_argument("--format", choices=["text", "csv", "json"], default="text")

    mc_parser = subparsers.add_parser("monte-carlo", help="Run Monte Carlo uncertainty analysis.")
    mc_parser.add_argument("scenario", help="Scenario name or path to YAML")
    mc_parser.add_argument("--strategy", required=True, choices=sorted(STRATEGIES.keys()))
    mc_parser.add_argument(
        "--params",
        nargs="+",
        default=["memory_reduction_fraction", "throughput_improvement_fraction"],
        help=(
            "Parameters to vary "
            "(default: memory_reduction_fraction throughput_improvement_fraction)"
        ),
    )
    mc_parser.add_argument(
        "--distribution",
        choices=["normal", "uniform", "triangular"],
        default="normal",
        help="Distribution type for parameter variation",
    )
    mc_parser.add_argument(
        "--std-dev",
        type=float,
        default=0.1,
        help="Standard deviation for normal distribution (as fraction of mean)",
    )
    mc_parser.add_argument(
        "--range",
        type=float,
        default=0.2,
        dest="range_fraction",
        help="Range for uniform/triangular distribution (as +/- fraction of mean)",
    )
    mc_parser.add_argument(
        "--simulations", type=int, default=1000, help="Number of simulations to run"
    )
    mc_parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    mc_parser.add_argument("--output", "-o", type=Path, help="Output file path")
    mc_parser.add_argument(
        "--format", choices=["text", "json", "csv"], default="text", help="Output format"
    )

    # Hyperparameter tuning command
    tune_parser = subparsers.add_parser(
        "tune",
        help="Automatically tune pruning strategy hyperparameters for optimal ROI.",
    )
    tune_parser.add_argument("scenario", help="Scenario name or path to YAML")
    tune_parser.add_argument(
        "--max-memory",
        type=float,
        help="Maximum memory constraint in GB",
    )
    tune_parser.add_argument(
        "--min-throughput",
        type=float,
        help="Minimum throughput constraint in tokens/sec",
    )
    tune_parser.add_argument(
        "--max-latency",
        type=float,
        help="Maximum latency constraint in ms per request",
    )
    tune_parser.add_argument(
        "--max-power",
        type=float,
        help="Maximum power constraint in watts",
    )
    tune_parser.add_argument(
        "--max-risk",
        choices=["low", "medium", "high"],
        default="medium",
        help="Maximum acceptable quality risk (default: medium)",
    )
    tune_parser.add_argument(
        "--use-quantization",
        action="store_true",
        help="Include quantization in optimization",
    )
    tune_parser.add_argument(
        "--fast-pruning",
        action="store_true",
        help="Prefer fast pruning frameworks",
    )
    tune_parser.add_argument(
        "--output",
        "-o",
        type=Path,
        help="Output file path for tuned strategy (YAML format)",
    )
    tune_parser.add_argument(
        "--format",
        choices=["text", "json", "yaml"],
        default="text",
        help="Output format (default: text)",
    )

    csv_md_parser = subparsers.add_parser(
        "csv-to-markdown", help="Convert CSV results to markdown report."
    )
    csv_md_parser.add_argument("input", type=Path, help="Path to CSV file")
    csv_md_parser.add_argument("--output", "-o", type=Path, help="Output markdown file path")

    telemetry_parser = subparsers.add_parser(
        "import-telemetry", help="Import benchmark telemetry to create a scenario."
    )
    telemetry_parser.add_argument("input", type=Path, help="Path to telemetry file")
    telemetry_parser.add_argument(
        "--format",
        choices=list(PARSERS.keys()),
        required=True,
        help="Telemetry format",
    )
    telemetry_parser.add_argument("--name", required=True, help="Scenario name")
    telemetry_parser.add_argument(
        "--mapping",
        type=str,
        help='Field mapping as JSON (e.g., \'{"memory_gb": "gpu_memory"}\')',
    )
    telemetry_parser.add_argument(
        "--electricity-cost", type=float, default=0.15, help="Electricity cost per kWh"
    )
    telemetry_parser.add_argument(
        "--hardware-cost", type=float, default=24000.0, help="Annual hardware cost in USD"
    )
    telemetry_parser.add_argument(
        "--project-cost", type=float, default=27000.0, help="One-time project cost in USD"
    )
    telemetry_parser.add_argument("--requests-per-day", type=int, help="Expected requests per day")
    telemetry_parser.add_argument("--output", "-o", type=Path, help="Output YAML file path")
    telemetry_parser.add_argument(
        "--preview", action="store_true", help="Preview scenario params without saving"
    )

    collect_parser = subparsers.add_parser(
        "collect-telemetry",
        help="Collect telemetry from a running inference server.",
    )
    collect_parser.add_argument(
        "--server-type",
        choices=["vllm", "tgi", "triton"],
        required=True,
        help="Type of inference server",
    )
    collect_parser.add_argument(
        "--url",
        default="http://localhost:8000",
        help="Server URL (default: http://localhost:8000)",
    )
    collect_parser.add_argument(
        "--duration",
        type=float,
        default=60.0,
        help="Collection duration in seconds (default: 60)",
    )
    collect_parser.add_argument(
        "--interval",
        type=float,
        default=5.0,
        help="Sampling interval in seconds (default: 5)",
    )
    collect_parser.add_argument(
        "--warmup",
        type=int,
        default=10,
        help="Number of warmup requests (default: 10)",
    )
    collect_parser.add_argument(
        "--output",
        "-o",
        type=Path,
        required=True,
        help="Output JSON file for telemetry",
    )
    collect_parser.add_argument(
        "--name",
        help="Scenario name for importing after collection",
    )
    collect_parser.add_argument(
        "--create-scenario",
        action="store_true",
        help="Create Atropos scenario from collected telemetry",
    )

    exp_parser = subparsers.add_parser(
        "import-experiment", help="Import scenario from experiment tracker (wandb/mlflow)."
    )
    exp_parser.add_argument(
        "--tracker",
        choices=list(TRACKERS.keys()),
        required=True,
        help="Experiment tracker type",
    )
    exp_parser.add_argument("--run-id", help="Specific run ID to import")
    exp_parser.add_argument("--experiment", help="Experiment/project name")
    exp_parser.add_argument("--entity", help="Entity/team name (wandb)")
    exp_parser.add_argument("--project", help="Project name (wandb)")
    exp_parser.add_argument("--limit", type=int, default=1, help="Number of runs to import")
    exp_parser.add_argument("--api-key", help="API key for authentication")
    exp_parser.add_argument("--host", help="Tracker host URL")
    exp_parser.add_argument("--name", help="Scenario name (or use run ID)")
    exp_parser.add_argument(
        "--electricity-cost", type=float, default=0.15, help="Electricity cost per kWh"
    )
    exp_parser.add_argument(
        "--hardware-cost", type=float, default=24000.0, help="Annual hardware cost in USD"
    )
    exp_parser.add_argument(
        "--project-cost", type=float, default=27000.0, help="One-time project cost in USD"
    )
    exp_parser.add_argument("--requests-per-day", type=int, help="Expected requests per day")
    exp_parser.add_argument("--output", "-o", type=Path, help="Output YAML file path")
    exp_parser.add_argument(
        "--preview", action="store_true", help="Preview scenario params without saving"
    )

    dashboard_parser = subparsers.add_parser(
        "dashboard", help="Launch the interactive web dashboard."
    )
    dashboard_parser.add_argument(
        "--host", default="127.0.0.1", help="Host to bind the dashboard server"
    )
    dashboard_parser.add_argument(
        "--port", type=int, default=8050, help="Port to run the dashboard server"
    )
    dashboard_parser.add_argument("--debug", action="store_true", help="Run in debug mode")

    carbon_parser = subparsers.add_parser(
        "list-carbon-presets", help="List available carbon intensity presets."
    )
    carbon_parser.add_argument(
        "--region", help="Show details for specific region (ISO code or cloud region)"
    )
    carbon_parser.add_argument(
        "--format", choices=["text", "json"], default="text", help="Output format"
    )

    cloud_pricing_parser = subparsers.add_parser(
        "cloud-pricing",
        help="Estimate and compare cloud/GPU-rental pricing across providers.",
    )
    cloud_subparsers = cloud_pricing_parser.add_subparsers(dest="subcommand", required=True)

    cloud_subparsers.add_parser("list-providers", help="List available cloud providers.")

    cloud_estimate_parser = cloud_subparsers.add_parser(
        "estimate", help="Estimate cloud pricing for a scenario."
    )
    cloud_estimate_parser.add_argument("--scenario", type=Path, required=True)
    cloud_estimate_parser.add_argument("--provider", help="Override deployment.platform")
    cloud_estimate_parser.add_argument("--fetch-live-pricing", action="store_true")
    cloud_estimate_parser.add_argument("--mock-pricing-api", action="store_true")

    cloud_compare_parser = cloud_subparsers.add_parser(
        "compare", help="Compare cloud pricing across providers for one scenario."
    )
    cloud_compare_parser.add_argument("--scenario", type=Path, required=True)
    cloud_compare_parser.add_argument(
        "--providers",
        required=True,
        help="Comma-separated providers (e.g., aws,azure,lambda-labs)",
    )
    cloud_compare_parser.add_argument("--fetch-live-pricing", action="store_true")
    cloud_compare_parser.add_argument("--mock-pricing-api", action="store_true")

    calibrate_parser = subparsers.add_parser(
        "calibrate",
        help="Calibrate scenario parameters against real telemetry data.",
    )
    calibrate_parser.add_argument("scenario", help="Scenario name (preset) or path to YAML file")
    calibrate_parser.add_argument(
        "telemetry", type=Path, help="Path to telemetry file (JSON, CSV, or log)"
    )
    calibrate_parser.add_argument(
        "--parser", choices=list(PARSERS.keys()), help="Telemetry parser type"
    )
    calibrate_parser.add_argument(
        "--tolerance",
        type=float,
        default=10.0,
        help="Acceptable variance tolerance percentage (default: 10)",
    )
    calibrate_parser.add_argument(
        "--format",
        choices=["markdown", "json"],
        default="markdown",
        help="Output format",
    )
    calibrate_parser.add_argument(
        "--output", "-o", type=Path, help="Output file path (default: stdout)"
    )

    # Pipeline command
    pipeline_parser = subparsers.add_parser(
        "pipeline",
        help="Run the Atropos optimization pipeline.",
    )
    pipeline_parser.add_argument("scenario", help="Scenario name (preset) or path to YAML file")
    pipeline_parser.add_argument(
        "--config", "-c", type=Path, required=True, help="Pipeline configuration YAML file"
    )
    pipeline_parser.add_argument(
        "--strategy",
        choices=sorted(STRATEGIES.keys()),
        default="structured_pruning",
        help="Optimization strategy to use",
    )
    pipeline_parser.add_argument(
        "--region", help="Region for carbon intensity (ISO code or cloud region)"
    )
    pipeline_parser.add_argument(
        "--dry-run", action="store_true", help="Simulate pipeline without actual execution"
    )
    pipeline_parser.add_argument(
        "--output", "-o", type=Path, help="Output JSON file for pipeline results"
    )

    # Pipeline config validation command
    pipeline_config_parser = subparsers.add_parser(
        "validate-pipeline-config",
        help="Validate a pipeline configuration file.",
    )
    pipeline_config_parser.add_argument(
        "config", type=Path, help="Pipeline configuration YAML file"
    )

    # Validation command (test against real models)
    validate_parser = subparsers.add_parser(
        "validate",
        help="Validate Atropos projections against real neural networks.",
    )
    validate_parser.add_argument("scenario", help="Scenario name (preset) or path to YAML file")
    validate_parser.add_argument(
        "--strategy",
        choices=sorted(STRATEGIES.keys()),
        default="structured_pruning",
        help="Optimization strategy to test",
    )
    validate_parser.add_argument(
        "--model", help="HuggingFace model name to test (auto-selected if not provided)"
    )
    validate_parser.add_argument(
        "--device", choices=["cpu", "cuda"], default="cpu", help="Device to run validation on"
    )
    validate_parser.add_argument(
        "--gpu-count",
        type=int,
        default=1,
        help="Number of GPUs to use for benchmarking (default: 1)",
    )
    validate_parser.add_argument(
        "--parallel-strategy",
        choices=["data", "layer", "model"],
        default="data",
        help="Parallelization strategy for multi-GPU (default: data)",
    )
    validate_parser.add_argument(
        "--pruning-method",
        default="magnitude",
        choices=["magnitude", "random", "structured"],
        help="Pruning method to apply",
    )
    validate_parser.add_argument(
        "--format", choices=["markdown", "json"], default="markdown", help="Output format"
    )
    validate_parser.add_argument(
        "--output", "-o", type=Path, help="Output file path (default: stdout)"
    )

    # Anomaly detection command
    anomaly_parser = subparsers.add_parser(
        "detect-anomalies",
        help="Detect statistical anomalies in optimization cost projections.",
    )
    anomaly_parser.add_argument("scenario", help="Scenario name (preset) or path to YAML file")
    anomaly_parser.add_argument(
        "--strategy",
        choices=sorted(STRATEGIES.keys()),
        default="structured_pruning",
        help="Optimization strategy to test",
    )
    anomaly_parser.add_argument(
        "--baseline",
        type=Path,
        help="Path to JSON file with historical optimization outcomes for baseline",
    )
    anomaly_parser.add_argument(
        "--threshold",
        type=float,
        default=3.0,
        help="Z-score threshold for anomaly detection (default: 3.0)",
    )
    anomaly_parser.add_argument(
        "--format", choices=["text", "json", "markdown"], default="text", help="Output format"
    )
    anomaly_parser.add_argument(
        "--output", "-o", type=Path, help="Output file path (default: stdout)"
    )

    # Multi-GPU benchmarking command
    benchmark_parser = subparsers.add_parser(
        "benchmark-multi-gpu",
        help="Analyze scaling efficiency across multiple GPUs.",
    )
    benchmark_parser.add_argument("scenario", help="Scenario name (preset) or path to YAML file")
    benchmark_parser.add_argument(
        "--strategy",
        choices=sorted(STRATEGIES.keys()),
        default="structured_pruning",
        help="Optimization strategy to test",
    )
    benchmark_parser.add_argument("--model", required=True, help="HuggingFace model name to test")
    benchmark_parser.add_argument(
        "--device", choices=["cpu", "cuda"], default="cuda", help="Device to run benchmarks on"
    )
    benchmark_parser.add_argument(
        "--gpu-count",
        type=int,
        default=1,
        help="Number of GPUs to use for benchmarking (default: 1)",
    )
    benchmark_parser.add_argument(
        "--parallel-strategy",
        choices=["data", "layer", "model"],
        default="data",
        help="Parallelization strategy for multi-GPU (default: data)",
    )
    benchmark_parser.add_argument(
        "--max-gpus",
        type=int,
        default=8,
        help="Maximum GPU count to test (default: 8)",
    )
    benchmark_parser.add_argument(
        "--gpu-counts",
        type=int,
        nargs="+",
        help="Specific GPU counts to test (e.g., '1 2 4 8')",
    )
    benchmark_parser.add_argument(
        "--format", choices=["markdown", "json"], default="markdown", help="Output format"
    )
    benchmark_parser.add_argument(
        "--output", "-o", type=Path, help="Output file path (default: stdout)"
    )

    # Test models command
    test_models_parser = subparsers.add_parser(
        "test-models",
        help="Test HuggingFace models for Atropos compatibility.",
    )
    test_models_parser.add_argument(
        "--models",
        nargs="+",
        help="Specific model IDs to test (default: recommended list)",
    )
    test_models_parser.add_argument(
        "--device",
        choices=["cpu", "cuda"],
        default="cpu",
        help="Device to test on",
    )
    test_models_parser.add_argument(
        "--max-params",
        type=float,
        default=3.0,
        help="Maximum parameter count in billions (default: 3)",
    )
    test_models_parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=Path("model-test-results.json"),
        help="Output JSON file for test results",
    )
    test_models_parser.add_argument(
        "--catalog",
        "-c",
        type=Path,
        help="Generate YAML catalog file",
    )

    # Visualization command
    visualize_parser = subparsers.add_parser(
        "visualize",
        help="Generate interactive visualizations from JSON reports.",
    )
    visualize_parser.add_argument(
        "input",
        type=Path,
        help="Path to JSON report file",
    )
    visualize_parser.add_argument(
        "output_dir",
        type=Path,
        help="Directory to save visualization files",
    )
    visualize_parser.add_argument(
        "--formats",
        nargs="+",
        choices=["html", "png"],
        default=["html"],
        help="Output formats (default: html)",
    )
    visualize_parser.add_argument(
        "--report-type",
        choices=[
            "auto",
            "framework-comparison",
            "tradeoff-analysis",
            "pruning-report",
            "validation-report",
        ],
        default="auto",
        help="Report type (default: auto-detect)",
    )

    # A/B testing command
    abtest_parser = subparsers.add_parser(
        "ab-test",
        help="Manage A/B testing experiments for model variants.",
    )
    abtest_subparsers = abtest_parser.add_subparsers(dest="subcommand", required=True)

    # create subcommand
    create_parser = abtest_subparsers.add_parser("create", help="Create a new A/B test experiment")
    create_parser.add_argument(
        "--config", "-c", type=Path, required=True, help="Path to experiment configuration YAML"
    )
    create_parser.add_argument(
        "--dry-run", action="store_true", help="Validate configuration without starting experiment"
    )

    # status subcommand
    status_parser = abtest_subparsers.add_parser("status", help="Check status of an experiment")
    status_parser.add_argument("experiment_id", help="Experiment identifier")
    status_parser.add_argument(
        "--format", choices=["text", "json", "yaml"], default="text", help="Output format"
    )

    # stop subcommand
    stop_parser = abtest_subparsers.add_parser("stop", help="Stop a running experiment")
    stop_parser.add_argument("experiment_id", help="Experiment identifier")
    stop_parser.add_argument("--reason", help="Reason for stopping")

    # analyze subcommand
    analyze_parser = abtest_subparsers.add_parser("analyze", help="Analyze experiment results")
    analyze_parser.add_argument("experiment_id", help="Experiment identifier")
    analyze_parser.add_argument(
        "--format", choices=["text", "json", "markdown"], default="text", help="Output format"
    )

    # promote subcommand
    promote_parser = abtest_subparsers.add_parser(
        "promote", help="Promote winning variant to production"
    )
    promote_parser.add_argument("experiment_id", help="Experiment identifier")
    promote_parser.add_argument(
        "--variant-id", help="Specific variant to promote (default: winner)"
    )
    promote_parser.add_argument(
        "--force", action="store_true", help="Promote even if not statistically significant"
    )

    # list subcommand
    list_parser = abtest_subparsers.add_parser("list", help="List experiments")
    list_parser.add_argument(
        "--status",
        choices=["draft", "running", "paused", "stopped", "completed"],
        help="Filter by status",
    )
    list_parser.add_argument(
        "--format", choices=["text", "json", "yaml"], default="text", help="Output format"
    )

    pruning_setup_parser = subparsers.add_parser(
        "setup-pruning",
        help="Check pruning dependencies and build isolated pruning framework containers.",
    )
    pruning_setup_parser.add_argument(
        "--fix",
        action="store_true",
        help="Force container rebuild without cache to fix dependency conflicts.",
    )

    pruning_test_parser = subparsers.add_parser(
        "test-pruning",
        help="Run pruning integration smoke test for a framework.",
    )
    pruning_test_parser.add_argument(
        "--framework",
        required=True,
        choices=["wanda", "sparsegpt", "llm-pruner"],
        help="Framework to test.",
    )

    return parser


def _add_strategy_args(parser: argparse.ArgumentParser) -> None:
    """Add common strategy-related arguments to a parser."""
    parser.add_argument(
        "--strategy", choices=sorted(STRATEGIES.keys()), default="structured_pruning"
    )
    parser.add_argument("--with-quantization", action="store_true")
    parser.add_argument("--report", choices=["text", "json", "markdown", "html"], default="text")


def _load_scenario_input(scenario_input: str) -> tuple[str, DeploymentScenario]:
    """Load scenario from preset name or YAML file path."""
    path = Path(scenario_input)
    if path.exists():
        scenario = load_scenario(path)
        return scenario.name, scenario
    if scenario_input not in SCENARIOS:
        raise KeyError(f"Unknown scenario '{scenario_input}'")
    return scenario_input, SCENARIOS[scenario_input]


def main(argv: Sequence[str] | None = None) -> int:
    """Main entry point for the CLI.

    Args:
        argv: Command-line arguments. If None, uses sys.argv.

    Returns:
        Exit code (0 for success, 1 for user errors, 2 for unexpected errors).
    """
    # Initial logging setup with defaults (before parsing args)
    setup_logging()
    try:
        parser = build_parser()
        args = parser.parse_args(argv)

        # Configure logging based on command-line flags
        setup_logging(
            verbose=args.verbose,
            debug=args.debug,
            traceback=args.traceback,
            log_file=args.log_file,
        )

        if args.command == "list-presets":
            print("Deployment Scenarios:")
            for name in sorted(SCENARIOS):
                scenario = SCENARIOS[name]
                print(f"  - {name}: {scenario.parameters_b}B params, {scenario.memory_gb:.0f}GB")
            print("\nOptimization Strategies:")
            for name in sorted(STRATEGIES):
                strategy = STRATEGIES[name]
                print(
                    f"  - {name}: {strategy.throughput_improvement_fraction * 100:.0f}% "
                    f"throughput, {strategy.memory_reduction_fraction * 100:.0f}% memory, "
                    f"risk={strategy.quality_risk}"
                )
            return 0

        if args.command == "preset":
            scenario = SCENARIOS[args.name]
            strategy = STRATEGIES[args.strategy]
            if args.with_quantization:
                strategy = combine_strategies(strategy, QUANTIZATION_BONUS)
            grid_co2e = get_carbon_intensity(args.region) if args.region else 0.35
            outcome = estimate_outcome(scenario, strategy, grid_co2e_kg_per_kwh=grid_co2e)
            print(render_report(outcome, args.report))
            return 0

        if args.command == "scenario":
            scenario = load_scenario(args.path)
            strategy = STRATEGIES[args.strategy]
            if args.with_quantization:
                strategy = combine_strategies(strategy, QUANTIZATION_BONUS)
            grid_co2e = get_carbon_intensity(args.region) if args.region else 0.35
            outcome = estimate_outcome(scenario, strategy, grid_co2e_kg_per_kwh=grid_co2e)
            print(render_report(outcome, args.report))
            return 0

        if args.command == "compare":
            scenario_name, scenario = _load_scenario_input(args.scenario)
            grid_co2e = get_carbon_intensity(args.region) if args.region else 0.35
            config = AtroposConfig(grid_co2e_factor=grid_co2e)
            calculator = ROICalculator(config=config)
            calculator.register_scenario(scenario)
            for name in args.strategies:
                calculator.register_strategy(STRATEGIES[name])
            results = calculator.compare_strategies(
                scenario_name, args.strategies, args.with_quantization
            )

            # Sort results
            outcomes = list(results.values())
            reverse = not args.ascending
            if args.sort_by == "savings":
                outcomes.sort(key=lambda o: o.annual_total_savings_usd, reverse=reverse)
            elif args.sort_by == "breakeven":
                outcomes.sort(key=lambda o: o.break_even_years or float("inf"), reverse=reverse)
            elif args.sort_by == "risk":
                risk_order = {"low": 0, "medium": 1, "high": 2}
                outcomes.sort(key=lambda o: risk_order[o.quality_risk], reverse=reverse)

            # Generate output
            if args.format == "markdown":
                content = generate_comparison_table(outcomes)
            elif args.format == "json":
                content = generate_comparison_json(outcomes)
            else:
                content = "\n\n".join(render_report(outcome, "text") for outcome in outcomes)
            if args.output:
                args.output.write_text(content)
            else:
                print(content)
            return 0

        if args.command == "batch":
            batch_report = cast(
                BatchExecutionReport,
                batch_process(
                    args.directory,
                    args.strategies,
                    args.output,
                    args.with_quantization,
                    fail_fast=args.fail_fast,
                    max_errors=args.max_errors,
                    error_log=args.error_log,
                    retry_attempts=args.retry_attempts,
                    timeout_seconds=args.scenario_timeout_seconds,
                    checkpoint_every=args.checkpoint_every,
                    resume_file=args.resume,
                    return_report=True,
                ),
            )
            print(
                "Batch summary: "
                f"total={batch_report.total_scenarios}, successful={batch_report.successful}, "
                f"failed={batch_report.failed}, partial_success={batch_report.partial_success}"
            )
            if batch_report.failures:
                print("Failures:")
                for failure in batch_report.failures:
                    print(
                        f" - {failure.scenario} [{failure.strategy}] "
                        f"{failure.category}: {failure.error}"
                    )
            return 0

        if args.command == "sensitivity":
            scenario_name, scenario = _load_scenario_input(args.scenario)
            calculator = ROICalculator()
            calculator.register_scenario(scenario)
            calculator.register_strategy(STRATEGIES[args.strategy])
            sens_results = calculator.sensitivity_analysis(
                scenario_name, args.strategy, args.param, args.variations, args.step
            )

            if args.format == "csv" and args.output:
                export_to_csv((outcome for _, outcome in sens_results), args.output)
                print(f"Saved sensitivity results to {args.output}")
            elif args.format == "json":
                import json

                data = [
                    {
                        "variation_factor": factor,
                        "annual_savings_usd": outcome.annual_total_savings_usd,
                        "break_even_months": (
                            outcome.break_even_years * 12 if outcome.break_even_years else None
                        ),
                        "optimized_memory_gb": outcome.optimized_memory_gb,
                        "optimized_throughput": outcome.optimized_throughput_toks_per_sec,
                    }
                    for factor, outcome in sens_results
                ]
                content = json.dumps(data, indent=2)
                if args.output:
                    args.output.write_text(content)
                    print(f"Saved sensitivity results to {args.output}")
                else:
                    print(content)
            else:
                for factor, outcome in sens_results:
                    months = (
                        None if outcome.break_even_years is None else outcome.break_even_years * 12
                    )
                    break_even = f"{months:.1f} mo" if months is not None else "never"
                    print(
                        f"factor={factor:.2f} savings=${outcome.annual_total_savings_usd:,.2f} "
                        f"break_even={break_even} memory={outcome.optimized_memory_gb:.2f}GB"
                    )
            return 0

        if args.command == "monte-carlo":
            scenario_name, scenario = _load_scenario_input(args.scenario)
            calculator = ROICalculator()
            calculator.register_scenario(scenario)
            calculator.register_strategy(STRATEGIES[args.strategy])

            distributions = [
                ParameterDistribution(
                    param_name=p,
                    distribution=args.distribution,
                    std_dev=args.std_dev,
                    range_fraction=args.range_fraction,
                )
                for p in args.params
            ]

            result = calculator.monte_carlo_analysis(
                scenario_name,
                args.strategy,
                distributions,
                num_simulations=args.simulations,
                seed=args.seed,
            )

            if args.format == "json":
                import json

                mc_data: dict[str, object] = {
                    "scenario": result.scenario_name,
                    "strategy": result.strategy_name,
                    "simulations": result.num_simulations,
                    "savings": {
                        "mean": result.savings_mean,
                        "std": result.savings_std,
                        "p5": result.savings_p5,
                        "p25": result.savings_p25,
                        "median": result.savings_median,
                        "p75": result.savings_p75,
                        "p95": result.savings_p95,
                    },
                    "break_even": {
                        "mean": result.break_even_mean,
                        "median": result.break_even_median,
                    },
                    "probabilities": {
                        "positive_roi": result.probability_positive_roi,
                        "break_even_within_1yr": result.probability_break_even_within_1yr,
                        "break_even_within_2yr": result.probability_break_even_within_2yr,
                    },
                    "co2e_savings_mean_kg": result.co2e_savings_mean,
                    "memory_reduction_mean": result.memory_reduction_mean,
                }
                content = json.dumps(mc_data, indent=2)
                if args.output:
                    args.output.write_text(content)
                    print(f"Saved Monte Carlo results to {args.output}")
                else:
                    print(content)
            elif args.format == "csv":
                if args.output:
                    export_to_csv(result.all_outcomes, args.output)
                    print(f"Saved {len(result.all_outcomes)} outcomes to {args.output}")
                else:
                    print("CSV format requires --output file")
                    return 1
            else:
                print("\nMonte Carlo Uncertainty Analysis")
                print("=" * 50)
                print(f"Scenario: {result.scenario_name}")
                print(f"Strategy: {result.strategy_name}")
                print(f"Simulations: {result.num_simulations}")
                print(f"\nParameters varied: {', '.join(args.params)}")
                print(f"Distribution: {args.distribution}")
                print("\nAnnual Savings Distribution:")
                print(f"  Mean:   ${result.savings_mean:,.2f}")
                print(f"  StdDev: ${result.savings_std:,.2f}")
                print(f"  P5:     ${result.savings_p5:,.2f}")
                print(f"  P25:    ${result.savings_p25:,.2f}")
                print(f"  Median: ${result.savings_median:,.2f}")
                print(f"  P75:    ${result.savings_p75:,.2f}")
                print(f"  P95:    ${result.savings_p95:,.2f}")
                print("\nBreak-even Time:")
                if result.break_even_mean:
                    print(f"  Mean:   {result.break_even_mean:.2f} years")
                    print(f"  Median: {result.break_even_median:.2f} years")
                else:
                    print("  No break-even in most simulations")
                print("\nProbabilities:")
                print(f"  Positive ROI: {result.probability_positive_roi:.1%}")
                print(f"  Break-even <= 1yr: {result.probability_break_even_within_1yr:.1%}")
                print(f"  Break-even <= 2yr: {result.probability_break_even_within_2yr:.1%}")
                print("\nOther Metrics:")
                print(f"  Mean CO2e savings: {result.co2e_savings_mean:.1f} kg/year")
                print(f"  Mean memory reduction: {result.memory_reduction_mean:.1%}")

                if args.output:
                    args.output.write_text(
                        f"# Monte Carlo Analysis: {result.scenario_name}\n\n"
                        f"**Strategy:** {result.strategy_name}  \n"
                        f"**Simulations:** {result.num_simulations}  \n\n"
                        "## Annual Savings Distribution\n\n"
                        "| Metric | Value |\n"
                        "|--------|-------|\n"
                        f"| Mean | ${result.savings_mean:,.2f} |\n"
                        f"| P5 | ${result.savings_p5:,.2f} |\n"
                        f"| Median | ${result.savings_median:,.2f} |\n"
                        f"| P95 | ${result.savings_p95:,.2f} |\n\n"
                        "## Probabilities\n\n"
                        f"- **Positive ROI:** {result.probability_positive_roi:.1%}\n"
                        "- **Break-even <= 1 year:** "
                        f"{result.probability_break_even_within_1yr:.1%}\n"
                        "- **Break-even <= 2 years:** "
                        f"{result.probability_break_even_within_2yr:.1%}\n"
                    )
                    print(f"\nSaved report to {args.output}")
            return 0

        if args.command == "tune":
            # Load scenario
            scenario_name, scenario = _load_scenario_input(args.scenario)

            # Build tuning constraints from command-line arguments
            constraints = TuningConstraints(
                max_memory_gb=args.max_memory,
                min_throughput_toks_per_sec=args.min_throughput,
                max_latency_ms_per_request=args.max_latency,
                max_power_watts=args.max_power,
                max_quality_risk=args.max_risk,
                use_quantization=args.use_quantization,
                prefer_fast_pruning=args.fast_pruning,
            )

            # Run hyperparameter tuning
            logger = logging.getLogger(__name__)
            logger.info("Starting hyperparameter tuning for scenario: %s", scenario_name)
            tuner = HyperparameterTuner(scenario=scenario, constraints=constraints)
            tuning_result: TuningResult = tuner.tune()

            # Output results
            if args.format == "json":
                import json

                output_data = {
                    "scenario": scenario_name,
                    "strategy": {
                        "name": tuning_result.strategy.name,
                        "parameter_reduction_fraction": (
                            tuning_result.strategy.parameter_reduction_fraction
                        ),
                        "memory_reduction_fraction": (
                            tuning_result.strategy.memory_reduction_fraction
                        ),
                        "throughput_improvement_fraction": (
                            tuning_result.strategy.throughput_improvement_fraction
                        ),
                        "power_reduction_fraction": tuning_result.strategy.power_reduction_fraction,
                        "quality_risk": tuning_result.strategy.quality_risk,
                    },
                    "recommended_framework": tuning_result.recommended_framework,
                    "quality_risk": tuning_result.quality_risk,
                    "expected_metrics": {
                        "memory_gb": tuning_result.expected_memory_gb,
                        "throughput_toks_per_sec": tuning_result.expected_throughput_toks_per_sec,
                        "latency_ms_per_request": tuning_result.expected_latency_ms_per_request,
                        "power_watts": tuning_result.expected_power_watts,
                    },
                    "tuning_metadata": tuning_result.tuning_metadata,
                }
                content = json.dumps(output_data, indent=2)
            elif args.format == "yaml":
                output_data = {
                    "scenario": scenario_name,
                    "strategy": {
                        "name": tuning_result.strategy.name,
                        "parameter_reduction_fraction": (
                            tuning_result.strategy.parameter_reduction_fraction
                        ),
                        "memory_reduction_fraction": (
                            tuning_result.strategy.memory_reduction_fraction
                        ),
                        "throughput_improvement_fraction": (
                            tuning_result.strategy.throughput_improvement_fraction
                        ),
                        "power_reduction_fraction": tuning_result.strategy.power_reduction_fraction,
                        "quality_risk": tuning_result.strategy.quality_risk,
                    },
                    "recommended_framework": tuning_result.recommended_framework,
                    "quality_risk": tuning_result.quality_risk,
                    "expected_metrics": {
                        "memory_gb": tuning_result.expected_memory_gb,
                        "throughput_toks_per_sec": tuning_result.expected_throughput_toks_per_sec,
                        "latency_ms_per_request": tuning_result.expected_latency_ms_per_request,
                        "power_watts": tuning_result.expected_power_watts,
                    },
                    "tuning_metadata": tuning_result.tuning_metadata,
                }
                content = yaml.dump(output_data, default_flow_style=False)
            else:  # text format
                content = (
                    f"Hyperparameter Tuning Results\n"
                    f"{'=' * 50}\n"
                    f"Scenario: {scenario_name}\n"
                    f"Strategy: {tuning_result.strategy.name}\n"
                    f"\n"
                    f"Optimization Strategy Parameters:\n"
                    f"  Parameter reduction: "
                    f"{tuning_result.strategy.parameter_reduction_fraction:.1%}\n"
                    f"  Memory reduction: {tuning_result.strategy.memory_reduction_fraction:.1%}\n"
                    f"  Throughput improvement: "
                    f"{tuning_result.strategy.throughput_improvement_fraction:.1%}\n"
                    f"  Power reduction: {tuning_result.strategy.power_reduction_fraction:.1%}\n"
                    f"  Quality risk: {tuning_result.strategy.quality_risk}\n"
                    f"\n"
                    f"Recommended Framework: {tuning_result.recommended_framework}\n"
                    f"\n"
                    f"Expected Metrics After Optimization:\n"
                    f"  Memory: {tuning_result.expected_memory_gb:.2f} GB\n"
                    f"  Throughput: "
                    f"{tuning_result.expected_throughput_toks_per_sec:.0f} tokens/sec\n"
                    f"  Latency: {tuning_result.expected_latency_ms_per_request:.1f} ms/request\n"
                    f"  Power: {tuning_result.expected_power_watts:.0f} W\n"
                    f"\n"
                    f"Model Architecture: "
                    f"{tuning_result.tuning_metadata.get('architecture', 'unknown')}\n"
                )

            if args.output:
                args.output.write_text(content)
                logger.info("Tuning results saved to %s", args.output)
            else:
                print(content)

            return 0

        if args.command == "csv-to-markdown":
            markdown = csv_to_markdown(args.input, args.output)
            if args.output:
                print(f"Saved markdown report to {args.output}")
            else:
                print(markdown)
            return 0

        if args.command == "import-telemetry":
            # Parse field mapping if provided
            field_mapping = None
            if args.mapping:
                import json

                field_mapping = json.loads(args.mapping)

            # Get parser and parse telemetry
            parser_instance = get_parser(args.format, field_mapping)
            telemetry = parser_instance.parse_file(args.input)

            # Validate telemetry
            issues = validate_telemetry(telemetry)
            import sys

            if issues:
                print("Telemetry validation issues:", file=sys.stderr)
                for issue in issues:
                    print(f"  - {issue}", file=sys.stderr)
                if any("must be" in i for i in issues):
                    return 1

            # Create scenario
            scenario = telemetry_to_scenario(
                telemetry,
                name=args.name,
                electricity_cost_per_kwh=args.electricity_cost,
                annual_hardware_cost_usd=args.hardware_cost,
                one_time_project_cost_usd=args.project_cost,
                requests_per_day=args.requests_per_day,
            )

            # Preview or save
            if args.preview or not args.output:
                print(f"\nScenario: {scenario.name}")
                print("=" * 50)
                print(f"Parameters: {scenario.parameters_b}B")
                print(f"Memory: {scenario.memory_gb:.1f} GB")
                print(f"Throughput: {scenario.throughput_toks_per_sec:.1f} tok/s")
                print(f"Power: {scenario.power_watts:.0f} W")
                print(f"Requests/day: {scenario.requests_per_day}")
                print(f"Tokens/request: {scenario.tokens_per_request}")
                print(f"Electricity cost: ${scenario.electricity_cost_per_kwh}/kWh")
                print(f"Annual hardware: ${scenario.annual_hardware_cost_usd:,.0f}")
                print(f"Project cost: ${scenario.one_time_project_cost_usd:,.0f}")
                print(f"\nSource: {telemetry.source}")
                if telemetry.raw_metrics:
                    print(f"\nRaw metrics available: {len(telemetry.raw_metrics)} fields")

            if args.output:
                # Convert dataclass to dict for YAML output
                scenario_dict = {
                    "name": scenario.name,
                    "parameters_b": scenario.parameters_b,
                    "memory_gb": scenario.memory_gb,
                    "throughput_toks_per_sec": scenario.throughput_toks_per_sec,
                    "power_watts": scenario.power_watts,
                    "requests_per_day": scenario.requests_per_day,
                    "tokens_per_request": scenario.tokens_per_request,
                    "electricity_cost_per_kwh": scenario.electricity_cost_per_kwh,
                    "annual_hardware_cost_usd": scenario.annual_hardware_cost_usd,
                    "one_time_project_cost_usd": scenario.one_time_project_cost_usd,
                }
                with open(args.output, "w", encoding="utf-8") as f:
                    yaml.dump(scenario_dict, f, default_flow_style=False, sort_keys=False)
                print(f"\nSaved scenario to {args.output}")
            return 0

        if args.command == "collect-telemetry":
            # Configure collection
            collection_config = CollectionConfig(
                collection_duration_sec=args.duration,
                sampling_interval_sec=args.interval,
                warmup_requests=args.warmup,
            )

            print(f"Collecting telemetry from {args.server_type} at {args.url}")
            print(f"Duration: {args.duration}s, Interval: {args.interval}s")

            # Collect telemetry
            collection_result = collect_and_save(
                server_type=args.server_type,
                base_url=args.url,
                output_path=args.output,
                config=collection_config,
            )

            if not collection_result.success:
                print(f"Error: {collection_result.error_message}", file=sys.stderr)
                return 1

            print("\nCollection complete!")
            if collection_result.aggregated:
                agg = collection_result.aggregated
                print(f"Throughput: {agg.throughput_toks_per_sec:.1f} tok/s")
                print(f"Latency: {agg.latency_ms_per_request:.1f} ms")
                print(f"Memory: {agg.memory_gb:.1f} GB")

            # Optionally create scenario
            if args.create_scenario and collection_result.aggregated:
                scenario = telemetry_to_scenario(
                    collection_result.aggregated,
                    name=args.name or f"{args.server_type}-collected",
                )

                scenario_path = args.output.with_suffix(".yaml")

                scenario_dict = {
                    "name": scenario.name,
                    "parameters_b": scenario.parameters_b,
                    "memory_gb": scenario.memory_gb,
                    "throughput_toks_per_sec": scenario.throughput_toks_per_sec,
                    "power_watts": scenario.power_watts,
                    "requests_per_day": scenario.requests_per_day,
                    "tokens_per_request": scenario.tokens_per_request,
                    "electricity_cost_per_kwh": scenario.electricity_cost_per_kwh,
                    "annual_hardware_cost_usd": scenario.annual_hardware_cost_usd,
                    "one_time_project_cost_usd": scenario.one_time_project_cost_usd,
                }
                with open(scenario_path, "w", encoding="utf-8") as f:
                    yaml.dump(scenario_dict, f, default_flow_style=False, sort_keys=False)
                print(f"\nScenario saved to: {scenario_path}")

            return 0

        if args.command == "import-experiment":
            # Validate args
            if not args.run_id and not args.experiment:
                print("Error: Either --run-id or --experiment must be provided", file=sys.stderr)
                return 1

            # Get tracker kwargs
            tracker_kwargs: dict[str, Any] = {}
            if args.entity:
                tracker_kwargs["entity"] = args.entity

            try:
                tracker = get_tracker(
                    args.tracker,
                    api_key=args.api_key,
                    host=args.host,
                    **tracker_kwargs,
                )

                if args.run_id:
                    run_kwargs = {}
                    if args.project:
                        run_kwargs["project"] = args.project
                    run_info = tracker.get_run(args.run_id, **run_kwargs)
                    runs = [run_info]
                else:
                    list_kwargs = {"limit": args.limit}
                    if args.entity:
                        list_kwargs["entity"] = args.entity
                    runs = tracker.list_runs(experiment=args.experiment, **list_kwargs)

                if not runs:
                    print("No runs found matching criteria")
                    return 0

                # Process runs
                for i, run in enumerate(runs):
                    name = args.name or f"{run.experiment}-{run.run_id}"
                    if len(runs) > 1:
                        name = f"{name}-{i + 1}"

                    scenario = run_to_scenario(
                        run,
                        name=name,
                        electricity_cost_per_kwh=args.electricity_cost,
                        annual_hardware_cost_usd=args.hardware_cost,
                        one_time_project_cost_usd=args.project_cost,
                        requests_per_day=args.requests_per_day,
                    )

                    # Preview
                    if args.preview or not args.output or len(runs) > 1:
                        print(f"\nScenario: {scenario.name}")
                        print("=" * 50)
                        print(f"Source: {run.tracker} / {run.experiment} / {run.run_id}")
                        if run.url:
                            print(f"URL: {run.url}")
                        print(f"Parameters: {scenario.parameters_b}B")
                        print(f"Memory: {scenario.memory_gb:.1f} GB")
                        print(f"Throughput: {scenario.throughput_toks_per_sec:.1f} tok/s")
                        print(f"Power: {scenario.power_watts:.0f} W")
                        print(f"Requests/day: {scenario.requests_per_day}")
                        print(f"Tokens/request: {scenario.tokens_per_request}")
                        if run.tags:
                            print(f"Tags: {', '.join(run.tags)}")

                    # Save if output specified and single run or last run
                    if args.output and len(runs) == 1:
                        scenario_dict = {
                            "name": scenario.name,
                            "parameters_b": scenario.parameters_b,
                            "memory_gb": scenario.memory_gb,
                            "throughput_toks_per_sec": scenario.throughput_toks_per_sec,
                            "power_watts": scenario.power_watts,
                            "requests_per_day": scenario.requests_per_day,
                            "tokens_per_request": scenario.tokens_per_request,
                            "electricity_cost_per_kwh": scenario.electricity_cost_per_kwh,
                            "annual_hardware_cost_usd": scenario.annual_hardware_cost_usd,
                            "one_time_project_cost_usd": scenario.one_time_project_cost_usd,
                        }
                        with open(args.output, "w", encoding="utf-8") as f:
                            yaml.dump(scenario_dict, f, default_flow_style=False, sort_keys=False)
                        print(f"\nSaved scenario to {args.output}")

                if len(runs) > 1:
                    print(f"\nImported {len(runs)} scenarios from {args.tracker}")

                return 0

            except RuntimeError as e:
                logging.error(f"{e}", exc_info=SHOW_TRACEBACK)
                logging.warning("Install the required package:")
                if args.tracker == "wandb":
                    logging.warning("  pip install wandb")
                elif args.tracker == "mlflow":
                    logging.warning("  pip install mlflow")
                return 1

        if args.command == "list-carbon-presets":
            if args.region:
                # Show specific region
                intensity = get_carbon_intensity(args.region)
                preset = None
                for code, p in CARBON_PRESETS.items():
                    if code == args.region.upper() or p.region_name.lower() == args.region.lower():
                        preset = p
                        break

                if args.format == "json":
                    import json

                    region_data: dict[str, Any] = {
                        "region": args.region,
                        "carbon_intensity_kg_per_kwh": intensity,
                        "preset": {
                            "region_code": preset.region_code if preset else args.region.upper(),
                            "region_name": preset.region_name if preset else "Unknown",
                            "data_year": preset.data_year if preset else 2023,
                            "source": preset.source if preset else "Global average",
                            "notes": preset.notes if preset else "",
                        }
                        if preset
                        else None,
                    }
                    print(json.dumps(region_data, indent=2))
                else:
                    print(f"\nRegion: {args.region}")
                    print("=" * 50)
                    print(f"Carbon intensity: {intensity:.3f} kg CO2e/kWh")
                    if preset:
                        print(f"Region name: {preset.region_name}")
                        print(f"Data year: {preset.data_year}")
                        print(f"Source: {preset.source}")
                        if preset.notes:
                            print(f"Notes: {preset.notes}")
            else:
                # List all regions
                regions = list_regions()

                if args.format == "json":
                    import json

                    data_list: list[dict[str, Any]] = [
                        {
                            "region_code": code,
                            "region_name": CARBON_PRESETS[code].region_name,
                            "carbon_intensity_kg_per_kwh": CARBON_PRESETS[
                                code
                            ].carbon_intensity_kg_per_kwh,
                        }
                        for code in regions
                    ]
                    print(json.dumps(data_list, indent=2))
                else:
                    print("\nAvailable carbon intensity presets:")
                    print("=" * 70)
                    print(f"{'Code':<6} {'Region':<30} {'Intensity (kg/kWh)':<20}")
                    print("-" * 70)
                    for code in regions:
                        preset = CARBON_PRESETS[code]
                        print(
                            f"{code:<6} {preset.region_name:<30} "
                            f"{preset.carbon_intensity_kg_per_kwh:<20.3f}"
                        )
                    print("\nUse --region CODE for details")
                    print("Cloud regions (e.g., us-east-1) are also supported")
            return 0

        if args.command == "cloud-pricing":
            engine = CloudPricingEngine()
            if getattr(args, "fetch_live_pricing", False):
                engine.refresh_live_pricing(use_mock_api=getattr(args, "mock_pricing_api", False))

            if args.subcommand == "list-providers":
                for provider in engine.list_providers():
                    print(provider)
                return 0

            scenario_data = yaml.safe_load(args.scenario.read_text())
            if not isinstance(scenario_data, dict):
                raise ValueError("Scenario file must contain a YAML mapping/object.")

            if args.subcommand == "estimate":
                request = request_from_scenario_yaml(
                    scenario_data=scenario_data,
                    provider_override=args.provider,
                )
                try:
                    estimate = engine.estimate(request)
                except KeyError:
                    request = dataclasses.replace(
                        request,
                        instance_type=engine.default_instance_type(request.provider),
                    )
                    estimate = engine.estimate(request)
                print(
                    f"{estimate.provider}/{estimate.instance_type} "
                    f"({estimate.purchase_option}) [{estimate.currency}]"
                )
                print(f"  Hourly:  {estimate.hourly_total_cost:.4f}")
                print(f"  Monthly: {estimate.monthly_total_cost:.2f}")
                print(f"  Annual:  {estimate.annual_total_cost:.2f}")
                print(
                    f"  Breakdown (monthly): compute={estimate.compute_monthly_cost:.2f}, "
                    f"serverless_req={estimate.serverless_request_monthly_cost:.2f}, "
                    f"serverless_duration={estimate.serverless_duration_monthly_cost:.2f}, "
                    f"storage={estimate.storage_monthly_cost:.2f}, "
                    f"network={estimate.network_monthly_cost:.2f}"
                )
                if estimate.risk_warning:
                    print(f"  Risk warning: {estimate.risk_warning}")
                if estimate.commitment_buyout_cost > 0:
                    print(
                        "  Commitment buyout: "
                        f"{estimate.commitment_buyout_cost:.2f} {estimate.currency}"
                    )
                if estimate.interruption_probability is not None:
                    print(f"  Interruption probability: {estimate.interruption_probability:.2%}")
                return 0

            if args.subcommand == "compare":
                providers = [p.strip() for p in args.providers.split(",") if p.strip()]
                if not providers:
                    raise ValueError("At least one provider must be passed in --providers.")

                estimates = []
                for provider in providers:
                    request = request_from_scenario_yaml(
                        scenario_data=scenario_data,
                        provider_override=provider,
                    )
                    try:
                        estimates.append(engine.estimate(request))
                    except KeyError:
                        fallback = dataclasses.replace(
                            request,
                            instance_type=engine.default_instance_type(provider),
                        )
                        estimates.append(engine.estimate(fallback))
                estimates.sort(key=lambda e: e.monthly_total_cost)

                print("Provider comparison (monthly total):")
                for estimate in estimates:
                    print(
                        f"  - {estimate.provider}: {estimate.monthly_total_cost:.2f} "
                        f"{estimate.currency} "
                        f"({estimate.instance_type}, {estimate.purchase_option})"
                    )
                if len(estimates) >= 2:
                    delta = estimates[-1].monthly_total_cost - estimates[0].monthly_total_cost
                    print(f"Spread best->worst: {delta:.2f} {estimates[0].currency} per month.")
                return 0

        if args.command == "dashboard":
            try:
                from .dashboard import run_dashboard

                print(f"Starting Atropos dashboard at http://{args.host}:{args.port}")
                print("Press Ctrl+C to stop")
                run_dashboard(host=args.host, port=args.port, debug=args.debug)
                return 0
            except ImportError:
                logging.error("Dashboard dependencies not installed", exc_info=SHOW_TRACEBACK)
                logging.warning("Install with: pip install dash plotly pandas")
                return 1

        if args.command == "calibrate":
            # Load scenario
            scenario_name, scenario = _load_scenario_input(args.scenario)

            # Parse telemetry
            telemetry_parser = get_parser(args.parser) if args.parser else None
            if telemetry_parser:
                telemetry = telemetry_parser.parse(args.telemetry)
            else:
                # Auto-detect parser from file extension
                from .telemetry import PARSERS

                suffix = args.telemetry.suffix.lower()
                if suffix == ".json":
                    telemetry = PARSERS["vllm"]().parse_file(args.telemetry)
                elif suffix == ".csv":
                    telemetry = PARSERS["csv"]().parse_file(args.telemetry)
                else:
                    print(
                        f"Error: Cannot auto-detect parser for {suffix} files. Use --parser.",
                        file=sys.stderr,
                    )
                    return 1

            # Validate telemetry
            issues = validate_telemetry(telemetry)
            if issues:
                print("Warning: Telemetry validation issues:", file=sys.stderr)
                for issue in issues:
                    print(f"  - {issue}", file=sys.stderr)

            # Run calibration
            calibration_result = calibrate_scenario(
                scenario, telemetry, tolerance_pct=args.tolerance
            )

            # Generate report
            calibration_report = generate_calibration_report(calibration_result, format=args.format)

            if args.output:
                args.output.write_text(calibration_report)
                print(f"Calibration report saved to {args.output}")
            else:
                print(calibration_report)

            return 0

        if args.command == "pipeline":
            # Load scenario and pipeline config
            scenario_name, scenario = _load_scenario_input(args.scenario)
            pipeline_config = PipelineConfig.from_yaml(args.config)
            strategy = STRATEGIES[args.strategy]
            grid_co2e = get_carbon_intensity(args.region) if args.region else 0.35

            print(f"Running pipeline: {pipeline_config.name}")
            print(f"Scenario: {scenario_name}")
            print(f"Strategy: {args.strategy}")
            if args.dry_run:
                print("Mode: DRY RUN (simulation only)")
            print()

            # Run pipeline
            pipeline_result = run_pipeline(
                config=pipeline_config,
                scenario=scenario,
                strategy=strategy,
                grid_co2e=grid_co2e,
                dry_run=args.dry_run,
            )

            # Print summary
            print(f"Pipeline status: {pipeline_result.final_status.name.lower()}")
            duration = pipeline_result.duration_seconds
            if duration:
                print(f"Duration: {duration:.1f}s")
            else:
                print("Duration: N/A")
            print()

            # Print stage results
            for stage in pipeline_result.stages:
                print(f"  {stage.stage.name.lower()}: {stage.status.name.lower()}")
                if stage.message:
                    print(f"    {stage.message}")

            # Save results if output specified
            if args.output:
                args.output.write_text(pipeline_result.to_json())
                print(f"\nResults saved to {args.output}")

            return_code = 0 if pipeline_result.final_status.name.lower() == "success" else 1
            return return_code

        if args.command == "validate-pipeline-config":
            try:
                pipeline_cfg = PipelineConfig.from_yaml(args.config)
                # Subconfigs are always set by __post_init__, but mypy doesn't know
                assert pipeline_cfg.thresholds is not None
                assert pipeline_cfg.pruning is not None
                assert pipeline_cfg.recovery is not None
                assert pipeline_cfg.validation is not None
                assert pipeline_cfg.deployment is not None

                print(f"Configuration valid: {pipeline_cfg.name}")
                print(f"  Auto-execute: {pipeline_cfg.auto_execute}")
                thresh = pipeline_cfg.thresholds
                print(
                    f"  Thresholds: {thresh.max_break_even_months} months break-even, "
                    f"${thresh.min_annual_savings_usd:,.0f} min savings"
                )
                fw = pipeline_cfg.pruning.framework
                sparsity = pipeline_cfg.pruning.target_sparsity
                print(f"  Pruning: {fw} at {sparsity:.0%} sparsity")
                recov = pipeline_cfg.recovery.enabled
                val_bm = pipeline_cfg.validation.quality_benchmark
                deploy = pipeline_cfg.deployment.auto_deploy
                print(f"  Recovery: {'enabled' if recov else 'disabled'}")
                print(f"  Validation: {val_bm} benchmark")
                print(f"  Deployment: {'auto' if deploy else 'manual'}")
                return 0
            except Exception as e:
                logging.error(f"Configuration error: {e}", exc_info=SHOW_TRACEBACK)
                return 1

        if args.command == "validate":
            # Load scenario and strategy
            scenario_name, scenario = _load_scenario_input(args.scenario)
            strategy = STRATEGIES[args.strategy]

            # Update scenario with multi-GPU parameters
            scenario = dataclasses.replace(scenario, gpu_count=args.gpu_count)
            scenario = dataclasses.replace(scenario, parallel_strategy=args.parallel_strategy)

            import os
            import sys

            print(f"Validating Atropos projections for: {scenario_name}")
            sys.stdout.flush()
            print(f"Strategy: {args.strategy}")
            sys.stdout.flush()
            print(f"Device: {args.device}")
            sys.stdout.flush()
            if args.model:
                print(f"Model: {args.model}")
                sys.stdout.flush()
            print()
            sys.stdout.flush()

            # Speed up torch import on Windows by disabling CUDA detection for CPU runs

            if args.device == "cpu":
                os.environ["CUDA_VISIBLE_DEVICES"] = ""

            # Run validation
            print("Loading validation module...")
            sys.stdout.flush()
            try:
                from .validation import run_validation

                validation_result = run_validation(
                    scenario=scenario,
                    strategy=strategy,
                    model_name=args.model,
                    device=args.device,
                    gpu_count=args.gpu_count,
                    pruning_method=args.pruning_method,
                )

                # Generate report
                if args.format == "json":
                    import json

                    validation_report = json.dumps(validation_result.to_dict(), indent=2)
                else:
                    validation_report = validation_result.to_markdown()

                if args.output:
                    args.output.write_text(validation_report)
                    print(f"Validation report saved to {args.output}")
                else:
                    print(validation_report)

                return 0
            except Exception as e:
                logging.error(f"Validation failed: {e}", exc_info=SHOW_TRACEBACK)
                return 1

        if args.command == "detect-anomalies":
            # Load scenario and strategy
            scenario_name, scenario = _load_scenario_input(args.scenario)
            strategy = STRATEGIES[args.strategy]

            # Compute optimization outcome
            outcome = estimate_outcome(scenario, strategy)

            # Load detector with optional baseline file
            from .validation.anomaly_detection import CostAnomalyDetector

            if args.baseline:
                try:
                    detector = CostAnomalyDetector.load_baselines_from_file(args.baseline)
                    # Override threshold if specified via CLI
                    if args.threshold != 3.0:
                        detector.threshold = args.threshold
                except Exception as e:
                    logging.warning(f"Failed to load baselines from {args.baseline}: {e}")
                    # Fall back to default detector
                    detector = CostAnomalyDetector(baseline_data=None, threshold=args.threshold)
            else:
                detector = CostAnomalyDetector(baseline_data=None, threshold=args.threshold)

            anomaly_result = detector.detect(outcome)

            # Generate report
            if args.format == "json":
                import json

                anomaly_report = json.dumps(anomaly_result.to_dict(), indent=2)
            elif args.format == "markdown":
                anomaly_report = anomaly_result.to_markdown()
            else:
                # text format
                anomaly_report = anomaly_result.to_markdown()  # markdown works for text

            if args.output:
                args.output.write_text(anomaly_report)
                print(f"Anomaly detection report saved to {args.output}")
            else:
                print(anomaly_report)

            return 0

        if args.command == "benchmark-multi-gpu":
            # Load scenario and strategy
            scenario_name, scenario = _load_scenario_input(args.scenario)
            strategy = STRATEGIES[args.strategy]

            # Update scenario with multi-GPU parameters (optional)
            scenario = dataclasses.replace(scenario, gpu_count=args.gpu_count)
            scenario = dataclasses.replace(scenario, parallel_strategy=args.parallel_strategy)

            import os
            import sys

            print(f"Running multi-GPU scaling analysis for: {scenario_name}")
            sys.stdout.flush()
            print(f"Strategy: {args.strategy}")
            sys.stdout.flush()
            print(f"Model: {args.model}")
            sys.stdout.flush()
            print(f"Device: {args.device}")
            sys.stdout.flush()
            if args.gpu_counts:
                print(f"GPU counts: {args.gpu_counts}")
            else:
                print(f"Max GPUs: {args.max_gpus}")
            sys.stdout.flush()
            print()
            sys.stdout.flush()

            # Speed up torch import on Windows by disabling CUDA detection for CPU runs
            if args.device == "cpu":
                os.environ["CUDA_VISIBLE_DEVICES"] = ""

            # Run scaling analysis
            print("Loading scaling analysis module...")
            sys.stdout.flush()
            try:
                from .validation.scaling_analyzer import analyze_scaling

                analysis_result = analyze_scaling(
                    scenario=scenario,
                    strategy=strategy,
                    model_name=args.model,
                    device=args.device,
                    max_gpus=args.max_gpus,
                    gpu_counts=args.gpu_counts,
                )

                # Generate report
                if args.format == "json":
                    import json

                    scaling_report = analysis_result.to_json()
                else:
                    scaling_report = analysis_result.to_markdown()

                if args.output:
                    args.output.write_text(scaling_report)
                    print(f"Scaling analysis report saved to {args.output}")
                else:
                    print(scaling_report)

                return 0
            except Exception as e:
                logging.error(f"Scaling analysis failed: {e}", exc_info=SHOW_TRACEBACK)
                return 1

        if args.command == "visualize":
            visualize_json(
                input_path=args.input,
                output_dir=args.output_dir,
                formats=args.formats,
                report_type=args.report_type,
            )
            print(f"Visualizations saved to {args.output_dir}")
            return 0

        if args.command == "test-models":
            # Get models to test
            models = args.models or get_recommended_test_models()

            # Run test suite
            test_results = run_test_suite(
                models=models,
                device=args.device,
                max_params_b=args.max_params,
                output_path=args.output,
            )

            # Generate catalog if requested
            if args.catalog:
                generate_catalog(test_results, args.catalog)

            # Return success if majority of tests passed
            success_rate = (
                test_results.successful / test_results.total_models
                if test_results.total_models > 0
                else 0
            )
            return 0 if success_rate >= 0.5 else 1

        if args.command == "setup-pruning":
            summary = setup_pruning_environment(fix=args.fix)
            if summary.success:
                print(summary.message)
                return 0
            print(summary.message, file=sys.stderr)
            return 1

        if args.command == "test-pruning":
            print(f"Testing pruning integration for {args.framework}...")
            pruning_result = test_pruning_framework(args.framework)
            if pruning_result.success:
                mode_text = pruning_result.mode.value
                print(f"{args.framework} integration OK (mode={mode_text})")
                if pruning_result.warning:
                    print(f"warning: {pruning_result.warning}")
                return 0
            print(
                pruning_result.error_message or "Unknown pruning integration error",
                file=sys.stderr,
            )
            return 1

        if args.command == "ab-test":
            # Handle A/B test subcommands
            if not hasattr(args, "subcommand"):
                parser.error("Missing subcommand for ab-test")
            subcommand = args.subcommand
            if subcommand == "create":
                # Load config

                with open(args.config) as f:
                    config_data = yaml.safe_load(f)
                # Convert variants
                variant_objects = []
                for v in config_data.get("variants", []):
                    variant_objects.append(Variant(**v))
                # Convert test_type string to enum
                test_type_str = config_data.get("test_type", "t-test")
                test_type = StatisticalTestType[test_type_str.upper().replace("-", "_")]
                # Create config
                ab_config = ABTestConfig(
                    experiment_id=config_data["experiment_id"],
                    name=config_data["name"],
                    variants=variant_objects,
                    primary_metric=config_data["primary_metric"],
                    secondary_metrics=config_data.get("secondary_metrics", []),
                    traffic_allocation=config_data.get("traffic_allocation", 1.0),
                    significance_level=config_data.get("significance_level", 0.05),
                    statistical_power=config_data.get("statistical_power", 0.8),
                    test_type=test_type,
                    min_sample_size_per_variant=config_data.get("min_sample_size_per_variant", 100),
                    max_duration_hours=config_data.get("max_duration_hours", 168.0),
                    auto_stop_conditions=config_data.get("auto_stop_conditions", {}),
                    deployment_platform=config_data.get("deployment_platform", "vllm"),
                    deployment_strategy=config_data.get("deployment_strategy", "immediate"),
                    health_checks=config_data.get("health_checks", {}),
                    metadata=config_data.get("metadata", {}),
                )
                # Get platform
                platform = get_platform(ab_config.deployment_platform)
                # Dry run?
                if args.dry_run:
                    print("Dry run: configuration valid")
                    print(f"Experiment ID: {ab_config.experiment_id}")
                    print(f"Variants: {len(ab_config.variants)}")
                    return 0
                # Save config to store
                store = get_default_store()
                store.save_config(ab_config)
                # Start experiment
                runner = ExperimentRunner(ab_config, platform)
                experiment_result = runner.start()
                # Update store with deployment IDs and status
                store.update_experiment(
                    experiment_id=ab_config.experiment_id,
                    status=runner.status,
                    deployment_ids=runner.deployment_ids,
                    start_time=runner.start_time,
                )
                print(f"Experiment started: {experiment_result.experiment_id}")
                print(f"Status: {experiment_result.status}")
                return 0
            elif subcommand == "status":
                import sys

                store = get_default_store()
                exp_data = store.load_experiment(args.experiment_id)
                if exp_data is None:
                    print(f"Experiment not found: {args.experiment_id}", file=sys.stderr)
                    return 1
                # Format output
                if args.format == "json":
                    import json

                    print(json.dumps(exp_data, indent=2))
                elif args.format == "yaml":
                    print(yaml.dump(exp_data, default_flow_style=False))
                else:  # text
                    print(f"Experiment ID: {exp_data.get('experiment_id', args.experiment_id)}")
                    print(f"Status: {exp_data.get('status', 'unknown')}")
                    print(f"Created: {exp_data.get('created_at', 'unknown')}")
                    print(f"Updated: {exp_data.get('updated_at', 'unknown')}")
                    if exp_data.get("start_time"):
                        print(f"Started: {exp_data['start_time']}")
                    if exp_data.get("end_time"):
                        print(f"Ended: {exp_data['end_time']}")
                    variant_dicts = exp_data.get("config", {}).get("variants", [])
                    print(f"Variants: {len(variant_dicts)}")
                    deployment_ids = exp_data.get("deployment_ids", {})
                    if deployment_ids:
                        print("Deployment IDs:")
                        for variant_id, dep_id in deployment_ids.items():
                            print(f"  {variant_id}: {dep_id}")
                return 0
            elif subcommand == "stop":
                import sys

                store = get_default_store()
                exp_data = store.load_experiment(args.experiment_id)
                if exp_data is None:
                    print(f"Experiment not found: {args.experiment_id}", file=sys.stderr)
                    return 1
                # Check current status
                current_status = exp_data.get("status", "").upper()
                if current_status in ("STOPPED", "COMPLETED", "FAILED"):
                    print(
                        f"Experiment already in terminal status: {current_status.lower()}",
                        file=sys.stderr,
                    )
                    return 1
                # Reconstruct config from stored data
                config_data = exp_data.get("config")
                if not config_data:
                    print("Invalid experiment data: missing config", file=sys.stderr)
                    return 1
                # Convert variants
                variants = []
                for v in config_data.get("variants", []):
                    variants.append(Variant(**v))
                # Convert test_type string to enum
                test_type_str = config_data.get("test_type", "t-test")
                test_type = StatisticalTestType[test_type_str.upper().replace("-", "_")]
                # Create config
                ab_config = ABTestConfig(
                    experiment_id=config_data["experiment_id"],
                    name=config_data["name"],
                    variants=variants,
                    primary_metric=config_data["primary_metric"],
                    secondary_metrics=config_data.get("secondary_metrics", []),
                    traffic_allocation=config_data.get("traffic_allocation", 1.0),
                    significance_level=config_data.get("significance_level", 0.05),
                    statistical_power=config_data.get("statistical_power", 0.8),
                    test_type=test_type,
                    min_sample_size_per_variant=config_data.get("min_sample_size_per_variant", 100),
                    max_duration_hours=config_data.get("max_duration_hours", 168.0),
                    auto_stop_conditions=config_data.get("auto_stop_conditions", {}),
                    deployment_platform=config_data.get("deployment_platform", "vllm"),
                    deployment_strategy=config_data.get("deployment_strategy", "immediate"),
                    health_checks=config_data.get("health_checks", {}),
                    metadata=config_data.get("metadata", {}),
                )
                # Get platform
                platform = get_platform(ab_config.deployment_platform)
                # Create runner and stop
                runner = ExperimentRunner(ab_config, platform)
                # Set runner's internal state from stored data
                runner._status = ExperimentStatus[current_status]
                runner._start_time = exp_data.get("start_time")
                runner._end_time = exp_data.get("end_time")
                runner._deployment_ids = exp_data.get("deployment_ids", {})
                # Stop experiment
                try:
                    stop_result = runner.stop(reason=args.reason if args.reason else "manual")
                except RuntimeError as e:
                    print(f"Cannot stop experiment: {e}", file=sys.stderr)
                    return 1
                # Update store with end_time and status
                store.update_experiment(
                    experiment_id=ab_config.experiment_id,
                    status=stop_result.status,
                    end_time=stop_result.end_time,
                )
                print(f"Experiment stopped: {ab_config.experiment_id}")
                print(f"Reason: {args.reason if args.reason else 'manual'}")
                return 0
            elif subcommand == "analyze":
                import json
                import sys

                store = get_default_store()
                exp_data = store.load_experiment(args.experiment_id)
                if exp_data is None:
                    print(f"Experiment not found: {args.experiment_id}", file=sys.stderr)
                    return 1
                # Reconstruct config from stored data
                config_data = exp_data.get("config")
                if not config_data:
                    print("Invalid experiment data: missing config", file=sys.stderr)
                    return 1
                # Convert variants
                variants = []
                for v in config_data.get("variants", []):
                    variants.append(Variant(**v))
                # Convert test_type string to enum
                test_type_str = config_data.get("test_type", "t-test")
                test_type = StatisticalTestType[test_type_str.upper().replace("-", "_")]
                # Create config
                ab_config = ABTestConfig(
                    experiment_id=config_data["experiment_id"],
                    name=config_data["name"],
                    variants=variants,
                    primary_metric=config_data["primary_metric"],
                    secondary_metrics=config_data.get("secondary_metrics", []),
                    traffic_allocation=config_data.get("traffic_allocation", 1.0),
                    significance_level=config_data.get("significance_level", 0.05),
                    statistical_power=config_data.get("statistical_power", 0.8),
                    test_type=test_type,
                    min_sample_size_per_variant=config_data.get("min_sample_size_per_variant", 100),
                    max_duration_hours=config_data.get("max_duration_hours", 168.0),
                    auto_stop_conditions=config_data.get("auto_stop_conditions", {}),
                    deployment_platform=config_data.get("deployment_platform", "vllm"),
                    deployment_strategy=config_data.get("deployment_strategy", "immediate"),
                    health_checks=config_data.get("health_checks", {}),
                    metadata=config_data.get("metadata", {}),
                )
                # Load variant metrics if available
                variant_metrics: dict[str, VariantMetrics] = {}
                stored_metrics = exp_data.get("variant_metrics", {})
                for variant_id, metrics_dict in stored_metrics.items():
                    # Convert nested dict structure to VariantMetrics
                    # metrics_dict should already be in the format from to_dict()
                    variant_metrics[variant_id] = VariantMetrics(**metrics_dict)
                # Perform analysis
                statistical_results = analyze_experiment_results(variant_metrics, ab_config)
                # Prepare output
                analysis_output: dict[str, Any] = {
                    "experiment_id": ab_config.experiment_id,
                    "status": exp_data.get("status", "unknown"),
                    "variant_metrics": {vid: vm.to_dict() for vid, vm in variant_metrics.items()},
                    "statistical_results": {
                        metric: sr.to_dict() for metric, sr in statistical_results.items()
                    },
                }
                # Format output
                if args.format == "json":
                    print(json.dumps(analysis_output, indent=2))
                elif args.format == "markdown":
                    # Generate markdown table
                    lines = []
                    lines.append(f"# Analysis for experiment: {ab_config.experiment_id}")
                    lines.append(f"**Status**: {exp_data.get('status', 'unknown')}")
                    lines.append("")
                    if statistical_results:
                        lines.append("## Statistical Results")
                        lines.append("| Metric | Test Type | p-value | Significant | Effect Size |")
                        lines.append("|--------|-----------|---------|-------------|-------------|")
                        for metric, sr in statistical_results.items():
                            lines.append(
                                f"| {metric} | {sr.test_type} | {sr.p_value:.4f} | "
                                f"{sr.is_significant} | {sr.effect_size:.3f} |"
                            )
                    else:
                        lines.append("No statistical results available (insufficient data).")
                    print("\n".join(lines))
                else:  # text
                    print(f"Experiment: {ab_config.experiment_id}")
                    print(f"Status: {exp_data.get('status', 'unknown')}")
                    print(f"Variant metrics: {len(variant_metrics)} variants")
                    for variant_id, vm in variant_metrics.items():
                        print(f"  {variant_id}: {vm.sample_count} samples")
                    if statistical_results:
                        print("Statistical results:")
                        for metric, sr in statistical_results.items():
                            print(
                                f"  {metric}: p={sr.p_value:.4f}, "
                                f"significant={sr.is_significant}, "
                                f"effect={sr.effect_size:.3f}"
                            )
                    else:
                        print("No statistical results (insufficient data)")
                return 0
            elif subcommand == "promote":
                import sys

                store = get_default_store()
                exp_data = store.load_experiment(args.experiment_id)
                if exp_data is None:
                    print(f"Experiment not found: {args.experiment_id}", file=sys.stderr)
                    return 1
                # Load config data
                config_data = exp_data.get("config")
                if not config_data:
                    print("Invalid experiment data: missing config", file=sys.stderr)
                    return 1
                # Get variant IDs from config
                variant_ids = [v["variant_id"] for v in config_data.get("variants", [])]
                # Determine which variant to promote
                winner_variant_id = args.variant_id
                if winner_variant_id is None:
                    # Try to get winner from stored statistical results
                    statistical_results = exp_data.get("statistical_results", {})
                    # Simplified: pick first variant for now
                    if variant_ids:
                        winner_variant_id = variant_ids[0]
                        print(
                            f"Warning: no winner specified, "
                            f"using first variant: {winner_variant_id}",
                            file=sys.stderr,
                        )
                    else:
                        print("No variants found in experiment", file=sys.stderr)
                        return 1
                # Validate variant exists
                if winner_variant_id not in variant_ids:
                    print(
                        f"Variant {winner_variant_id} not found in experiment. "
                        f"Available: {', '.join(variant_ids)}",
                        file=sys.stderr,
                    )
                    return 1
                # Check statistical significance if not forced
                if not args.force:
                    statistical_results = exp_data.get("statistical_results", {})
                    # Check if any result indicates significance for primary metric
                    primary_metric = config_data.get("primary_metric")
                    if primary_metric in statistical_results:
                        sr_dict = statistical_results[primary_metric]
                        # sr_dict may be dict (from stored StatisticalResult.to_dict())
                        is_significant = sr_dict.get("is_significant", False)
                        if not is_significant:
                            print(
                                f"Warning: primary metric '{primary_metric}' "
                                f"not statistically significant.",
                                file=sys.stderr,
                            )
                            print("Use --force to promote anyway.", file=sys.stderr)
                            return 1
                    else:
                        print(
                            f"Warning: no statistical results "
                            f"for primary metric '{primary_metric}'.",
                            file=sys.stderr,
                        )
                        print("Use --force to promote anyway.", file=sys.stderr)
                        return 1
                # Update store with winner
                store.update_experiment(
                    experiment_id=args.experiment_id,
                    winner_variant_id=winner_variant_id,
                )
                print(f"Variant promoted: {winner_variant_id}")
                print(f"Experiment: {args.experiment_id}")
                # Actually deploy promoted variant to production
                from datetime import datetime

                from ..deployment.models import DeploymentRequest, DeploymentStrategyType

                # Find winning variant details
                winning_variant = None
                for v in config_data.get("variants", []):
                    if v.get("variant_id") == winner_variant_id:
                        winning_variant = v
                        break
                if not winning_variant:
                    print(
                        f"Error: variant {winner_variant_id} not found in config", file=sys.stderr
                    )
                    return 1

                # Get deployment platform
                deployment_platform = config_data.get("deployment_platform", "vllm")
                health_checks = config_data.get("health_checks", {})

                # Create deployment request for production
                deployment_request = DeploymentRequest(
                    model_path=winning_variant["model_path"],
                    platform=deployment_platform,
                    strategy=DeploymentStrategyType.IMMEDIATE,
                    health_checks=health_checks,
                    metadata={
                        "promoted_from_experiment": args.experiment_id,
                        "variant_id": winner_variant_id,
                        "promoted_at": datetime.now().isoformat(),
                    },
                )

                try:
                    platform = get_platform(deployment_platform, {})
                    deployment_result = platform.deploy(deployment_request)

                    if deployment_result.status.name == "SUCCESS":
                        print(f"Deployed to production: {deployment_result.deployment_id}")
                        if deployment_result.endpoints:
                            print(f"Endpoints: {', '.join(deployment_result.endpoints)}")
                        # Optionally update store with production deployment ID
                        # store.update_experiment(...)
                    else:
                        print(f"Deployment failed: {deployment_result.message}", file=sys.stderr)
                        return 1
                except Exception as e:
                    print(f"Error deploying variant: {e}", file=sys.stderr)
                    return 1

                return 0
            elif subcommand == "list":
                import json

                store = get_default_store()
                experiments = store.list_experiments(status_filter=args.status)
                # Format output
                if args.format == "json":
                    print(json.dumps(experiments, indent=2))
                elif args.format == "yaml":
                    print(yaml.dump(experiments, default_flow_style=False))
                else:  # text
                    if not experiments:
                        print("No experiments found.")
                        return 0
                    print(f"Experiments ({len(experiments)}):")
                    for exp in experiments:
                        exp_id = exp.get("experiment_id", "unknown")
                        name = exp.get("config", {}).get("name", "unknown")
                        status = exp.get("status", "unknown")
                        created = exp.get("created_at", "unknown")
                        variant_count = len(exp.get("config", {}).get("variants", []))
                        print(f"  {exp_id}: {name} [{status}]")
                        print(f"    Created: {created}, Variants: {variant_count}")
                        if exp.get("start_time"):
                            print(f"    Started: {exp['start_time']}")
                return 0
            else:
                parser.error(f"Unknown subcommand: {subcommand}")

        parser.error(f"Unsupported command: {args.command}")
        return 2
    except FileNotFoundError as e:
        logging.error(f"File not found: {e}", exc_info=SHOW_TRACEBACK)
        return 1
    except ValueError as e:
        logging.error(f"Invalid input: {e}", exc_info=SHOW_TRACEBACK)
        return 1
    except KeyError as e:
        logging.error(f"{e}", exc_info=SHOW_TRACEBACK)
        return 1
    except AtroposError as e:
        logging.error(f"{e}", exc_info=SHOW_TRACEBACK)
        return 1
    except Exception as e:
        logging.error(f"Unexpected error: {e}", exc_info=SHOW_TRACEBACK)
        return 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
