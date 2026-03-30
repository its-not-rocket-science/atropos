"""Telemetry import from benchmark runs for scenario calibration."""

from __future__ import annotations

import csv
import json
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Mapping

from .models import DeploymentScenario


@dataclass(frozen=True)
class TelemetryData:
    """Parsed telemetry data from benchmark runs.

    Attributes:
        source: Source of the telemetry (e.g., 'vllm', 'triton').
        parameters_b: Model size in billions of parameters (optional).
        memory_gb: Peak memory usage in GB.
        throughput_toks_per_sec: Token throughput per second.
        latency_ms_per_request: Average latency per request in milliseconds.
        power_watts: Average power consumption in watts (optional).
        requests_per_day: Estimated or actual requests per day.
        tokens_per_request: Average tokens per request.
        timestamp: When the telemetry was collected.
        raw_metrics: Additional raw metrics from the source.
        experiment_id: Experiment identifier for A/B testing (optional).
        variant_id: Variant identifier for A/B testing (optional).
    """

    source: str
    memory_gb: float
    throughput_toks_per_sec: float
    latency_ms_per_request: float
    tokens_per_request: float
    parameters_b: float | None = None
    power_watts: float | None = None
    requests_per_day: int | None = None
    timestamp: str | None = None
    raw_metrics: dict[str, Any] = field(default_factory=dict)
    experiment_id: str | None = None
    variant_id: str | None = None


def validate_telemetry(data: TelemetryData) -> list[str]:
    """Validate telemetry data and return list of issues.

    Args:
        data: Telemetry data to validate.

    Returns:
        List of validation issue messages. Empty if valid.
    """
    issues: list[str] = []

    if data.memory_gb <= 0:
        issues.append(f"memory_gb must be positive, got {data.memory_gb}")
    if data.throughput_toks_per_sec <= 0:
        issues.append(
            f"throughput_toks_per_sec must be positive, got {data.throughput_toks_per_sec}"
        )
    if data.latency_ms_per_request <= 0:
        issues.append(f"latency_ms_per_request must be positive, got {data.latency_ms_per_request}")
    if data.tokens_per_request <= 0:
        issues.append(f"tokens_per_request must be positive, got {data.tokens_per_request}")

    # Sanity checks for reasonable values
    if data.memory_gb > 1000:
        issues.append(f"memory_gb seems high ({data.memory_gb} GB), verify value")
    if data.throughput_toks_per_sec > 10000:
        issues.append(
            f"throughput_toks_per_sec seems high ({data.throughput_toks_per_sec}), verify value"
        )

    return issues


def extract_scenario_params(
    telemetry: TelemetryData,
    electricity_cost_per_kwh: float = 0.15,
    annual_hardware_cost_usd: float = 24000.0,
    one_time_project_cost_usd: float = 27000.0,
    requests_per_day: int | None = None,
) -> dict[str, Any]:
    """Extract scenario parameters from telemetry data.

    Args:
        telemetry: Parsed telemetry data.
        electricity_cost_per_kwh: Electricity cost per kWh.
        annual_hardware_cost_usd: Annual hardware cost.
        one_time_project_cost_usd: One-time optimization project cost.
        requests_per_day: Override for requests per day (uses telemetry if None).

    Returns:
        Dictionary of parameters suitable for DeploymentScenario.
    """
    reqs_per_day = requests_per_day or telemetry.requests_per_day or 50000

    # Estimate power if not provided (rough heuristic: ~10W per GB of GPU memory)
    power = telemetry.power_watts or telemetry.memory_gb * 10

    return {
        "parameters_b": telemetry.parameters_b or 7.0,
        "memory_gb": telemetry.memory_gb,
        "throughput_toks_per_sec": telemetry.throughput_toks_per_sec,
        "power_watts": power,
        "requests_per_day": reqs_per_day,
        "tokens_per_request": int(telemetry.tokens_per_request),
        "electricity_cost_per_kwh": electricity_cost_per_kwh,
        "annual_hardware_cost_usd": annual_hardware_cost_usd,
        "one_time_project_cost_usd": one_time_project_cost_usd,
    }


class TelemetryParser(ABC):
    """Abstract base class for telemetry parsers."""

    @abstractmethod
    def parse(self, data: dict[str, Any] | str) -> TelemetryData:
        """Parse telemetry data from source format.

        Args:
            data: Raw data to parse (dict for JSON, str for CSV lines).

        Returns:
            Parsed TelemetryData.

        Raises:
            ValueError: If data cannot be parsed.
        """
        raise NotImplementedError

    def parse_file(self, path: Path) -> TelemetryData:
        """Parse telemetry from a file.

        Args:
            path: Path to the telemetry file.

        Returns:
            Parsed TelemetryData.
        """
        content = path.read_text(encoding="utf-8")

        if path.suffix.lower() == ".json":
            data: dict[str, Any] = json.loads(content)
            return self.parse(data)

        if path.suffix.lower() == ".csv":
            return self.parse(content)

        # Try JSON first, then treat as CSV
        try:
            data = json.loads(content)
            return self.parse(data)
        except json.JSONDecodeError:
            return self.parse(content)


class VLLMTelemetryParser(TelemetryParser):
    """Parser for vLLM benchmark output."""

    def parse(self, data: dict[str, Any] | str) -> TelemetryData:
        """Parse vLLM benchmark JSON output."""
        if isinstance(data, str):
            parsed_data: dict[str, Any] = json.loads(data)
        else:
            parsed_data = data

        # vLLM benchmark format
        metrics: dict[str, Any] = parsed_data.get("metrics", parsed_data)

        # Extract throughput - vLLM reports tok/s
        throughput = metrics.get("throughput", metrics.get("tokens_per_second", 0.0))
        if isinstance(throughput, str):
            throughput = float(throughput)

        # Extract latency
        latency = metrics.get("mean_latency_ms", metrics.get("latency_ms", 0.0))
        if isinstance(latency, str):
            latency = float(latency)

        # Extract memory - may be in different fields
        memory = metrics.get("gpu_memory_gb", metrics.get("memory_gb", 0.0))
        if memory == 0.0:
            # Try to infer from GPU usage
            gpu_usage = metrics.get("gpu_memory_usage", 0.0)
            if gpu_usage > 0:
                # Assume percentage of common GPU sizes
                memory = gpu_usage * 80.0 / 100.0  # Rough estimate for 80GB GPU

        # Extract tokens per request
        input_tokens = metrics.get("mean_input_tokens", metrics.get("input_tokens", 0))
        output_tokens = metrics.get("mean_output_tokens", metrics.get("output_tokens", 0))
        tokens_per_req = input_tokens + output_tokens or 1000

        # Extract model size if available
        model_info = parsed_data.get("model", "")
        params_b = None
        if isinstance(model_info, str):
            # Try to extract from name like "meta-llama/Llama-2-7b-chat-hf"
            match = re.search(r"-(\d+)(\.\d+)?b", model_info.lower())
            if match:
                params_b = (
                    float(match.group(1))
                    if match.group(2) is None
                    else float(match.group(1) + match.group(2))
                )

        return TelemetryData(
            source="vllm",
            parameters_b=params_b,
            memory_gb=float(memory) if memory else 0.0,
            throughput_toks_per_sec=float(throughput) if throughput else 0.0,
            latency_ms_per_request=float(latency) if latency else 0.0,
            tokens_per_request=float(tokens_per_req),
            power_watts=metrics.get("power_watts"),
            requests_per_day=metrics.get("requests_per_day"),
            timestamp=metrics.get("timestamp"),
            raw_metrics=metrics,
            experiment_id=None,
            variant_id=None,
        )


class TritonTelemetryParser(TelemetryParser):
    """Parser for Triton inference server metrics."""

    def parse(self, data: dict[str, Any] | str) -> TelemetryData:
        """Parse Triton inference server metrics."""
        if isinstance(data, str):
            parsed_data: dict[str, Any] = json.loads(data)
        else:
            parsed_data = data

        # Triton metrics format
        model_stats: list[dict[str, Any]] = parsed_data.get("model_stats", [])
        if not model_stats:
            raise ValueError("No model_stats found in Triton telemetry")

        stats = model_stats[0]
        inference_stats: dict[str, Any] = stats.get("inference_stats", {})
        batch_stats: dict[str, Any] = stats.get("batch_stats", {})

        # Extract latency (in microseconds from Triton)
        latency_us = inference_stats.get("execution_count", 0)
        compute_time = inference_stats.get("compute_infer", {}).get("ns", 0) / 1e6  # Convert to ms

        # Extract throughput
        success_count = inference_stats.get("success", {}).get("count", 0)
        time_window_sec = 60.0  # Assume 1-minute window if not specified
        throughput = success_count / time_window_sec if time_window_sec > 0 else 0.0

        # Extract batch size and estimate tokens
        batch_size = batch_stats.get("max_batch_size", 1)
        tokens_per_req = batch_size * 1000  # Rough estimate

        return TelemetryData(
            source="triton",
            memory_gb=parsed_data.get("memory_gb", 0.0),
            throughput_toks_per_sec=throughput,
            latency_ms_per_request=compute_time or latency_us / 1000.0,
            tokens_per_request=float(tokens_per_req),
            parameters_b=parsed_data.get("parameters_b"),
            power_watts=parsed_data.get("power_watts"),
            requests_per_day=parsed_data.get("requests_per_day"),
            timestamp=parsed_data.get("timestamp"),
            raw_metrics=parsed_data,
            experiment_id=None,
            variant_id=None,
        )


class CSVTelemetryParser(TelemetryParser):
    """Parser for CSV telemetry data."""

    def __init__(self, field_mapping: Mapping[str, str] | None = None):
        """Initialize with optional field mapping.

        Args:
            field_mapping: Maps standard field names to CSV column names.
                e.g., {"memory_gb": "gpu_memory", "throughput": "tok_per_sec"}
        """
        self.field_mapping = field_mapping or {}

    def _get_field(
        self, row: dict[str, str], standard_name: str, default: str | None = ""
    ) -> str | None:
        """Get field value using mapping or standard name."""
        csv_name = self.field_mapping.get(standard_name, standard_name)
        return row.get(csv_name, row.get(standard_name, default))

    def parse(self, data: dict[str, Any] | str) -> TelemetryData:
        """Parse CSV telemetry data."""
        if isinstance(data, dict):
            raise ValueError("CSV parser expects string data, not dict")

        # Parse CSV
        lines = data.strip().split("\n")
        if len(lines) < 2:
            raise ValueError("CSV must have header and at least one data row")

        reader = csv.DictReader(lines)
        rows = list(reader)
        if not rows:
            raise ValueError("No data rows found in CSV")

        # Use first row for now (could average multiple rows)
        row = rows[0]

        def get_float(field: str, default: float | None = 0.0) -> float | None:
            val = self._get_field(row, field, "")
            return float(val) if val else default

        def get_int(field: str, default: int | None = 0) -> int | None:
            val = self._get_field(row, field, "")
            return int(float(val)) if val else default

        memory = get_float("memory_gb", get_float("gpu_memory", 0.0))
        throughput = get_float(
            "throughput_toks_per_sec", get_float("throughput", get_float("tok_per_sec", 0.0))
        )
        latency = get_float(
            "latency_ms_per_request", get_float("latency_ms", get_float("latency", 0.0))
        )
        tokens_val = get_int("tokens_per_request", get_int("tokens", 1000))
        tokens = float(tokens_val if tokens_val is not None else 1000)

        if memory is None or throughput is None or latency is None:
            raise ValueError("memory_gb, throughput, and latency are required")

        return TelemetryData(
            source="csv",
            memory_gb=memory,
            throughput_toks_per_sec=throughput,
            latency_ms_per_request=latency,
            tokens_per_request=tokens,
            parameters_b=get_float("parameters_b", None),
            power_watts=get_float("power_watts", None),
            requests_per_day=get_int("requests_per_day", None),
            timestamp=self._get_field(row, "timestamp", None),
            raw_metrics=dict(row),
            experiment_id=self._get_field(row, "experiment_id", None),
            variant_id=self._get_field(row, "variant_id", None),
        )


class GenericJSONTelemetryParser(TelemetryParser):
    """Parser for generic JSON with configurable field mapping."""

    def __init__(self, field_mapping: Mapping[str, str] | None = None):
        """Initialize with field mapping.

        Args:
            field_mapping: Maps standard field names to JSON path (dot notation).
                e.g., {"memory_gb": "gpu.memory_used_gb"}
        """
        self.field_mapping = field_mapping or {}

    def _get_nested(self, data: dict[str, Any], path: str) -> Any:
        """Get nested value using dot notation."""
        parts = path.split(".")
        current: Any = data
        for part in parts:
            if isinstance(current, dict):
                current = current.get(part)
            else:
                return None
        return current

    def _get_value(self, data: dict[str, Any], standard_name: str, default: Any = None) -> Any:
        """Get value using mapping or standard name."""
        if standard_name in self.field_mapping:
            mapped_path = self.field_mapping[standard_name]
            return self._get_nested(data, mapped_path)

        # Try standard name directly
        return data.get(standard_name, default)

    def parse(self, data: dict[str, Any] | str) -> TelemetryData:
        """Parse generic JSON telemetry."""
        if isinstance(data, str):
            parsed_data: dict[str, Any] = json.loads(data)
        else:
            parsed_data = data

        def get_float(field: str, default: float | None = None) -> float | None:
            val = self._get_value(parsed_data, field)
            if val is None:
                return default
            return float(val)

        def get_int(field: str, default: int | None = None) -> int | None:
            val = self._get_value(parsed_data, field)
            if val is None:
                return default
            return int(float(val))

        memory = get_float("memory_gb")
        if memory is None:
            raise ValueError("memory_gb is required")

        throughput = get_float("throughput_toks_per_sec")
        if throughput is None:
            raise ValueError("throughput_toks_per_sec is required")

        latency = get_float("latency_ms_per_request")
        if latency is None:
            raise ValueError("latency_ms_per_request is required")

        tokens = get_float("tokens_per_request")
        if tokens is None:
            raise ValueError("tokens_per_request is required")

        return TelemetryData(
            source="generic_json",
            memory_gb=memory,
            throughput_toks_per_sec=throughput,
            latency_ms_per_request=latency,
            tokens_per_request=tokens,
            parameters_b=get_float("parameters_b"),
            power_watts=get_float("power_watts"),
            requests_per_day=get_int("requests_per_day"),
            timestamp=self._get_value(parsed_data, "timestamp"),
            raw_metrics=parsed_data,
            experiment_id=self._get_value(parsed_data, "experiment_id"),
            variant_id=self._get_value(parsed_data, "variant_id"),
        )


PARSERS: dict[str, type[TelemetryParser]] = {
    "vllm": VLLMTelemetryParser,
    "triton": TritonTelemetryParser,
    "csv": CSVTelemetryParser,
    "json": GenericJSONTelemetryParser,
}


def get_parser(format_name: str, field_mapping: Mapping[str, str] | None = None) -> TelemetryParser:
    """Get a telemetry parser by format name.

    Args:
        format_name: One of 'vllm', 'triton', 'csv', 'json'.
        field_mapping: Optional field mapping for csv/json parsers.

    Returns:
        TelemetryParser instance.

    Raises:
        ValueError: If format is not recognized.
    """
    if format_name not in PARSERS:
        raise ValueError(f"Unknown format '{format_name}'. Available: {list(PARSERS.keys())}")

    if format_name == "csv":
        return CSVTelemetryParser(field_mapping)
    if format_name == "json":
        return GenericJSONTelemetryParser(field_mapping)
    if format_name == "vllm":
        return VLLMTelemetryParser()
    return TritonTelemetryParser()


def telemetry_to_scenario(
    telemetry: TelemetryData,
    name: str,
    electricity_cost_per_kwh: float = 0.15,
    annual_hardware_cost_usd: float = 24000.0,
    one_time_project_cost_usd: float = 27000.0,
    requests_per_day: int | None = None,
) -> DeploymentScenario:
    """Convert telemetry data to a DeploymentScenario.

    Args:
        telemetry: Parsed telemetry data.
        name: Scenario name.
        electricity_cost_per_kwh: Electricity cost per kWh.
        annual_hardware_cost_usd: Annual hardware cost.
        one_time_project_cost_usd: One-time optimization project cost.
        requests_per_day: Override for requests per day.

    Returns:
        DeploymentScenario populated from telemetry.
    """
    params = extract_scenario_params(
        telemetry,
        electricity_cost_per_kwh=electricity_cost_per_kwh,
        annual_hardware_cost_usd=annual_hardware_cost_usd,
        one_time_project_cost_usd=one_time_project_cost_usd,
        requests_per_day=requests_per_day,
    )

    return DeploymentScenario(name=name, **params)
