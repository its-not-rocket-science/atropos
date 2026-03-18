"""Tests for telemetry import functionality."""

import json
from pathlib import Path

import pytest

from atropos.models import DeploymentScenario
from atropos.telemetry import (
    CSVTelemetryParser,
    GenericJSONTelemetryParser,
    TelemetryData,
    TritonTelemetryParser,
    VLLMTelemetryParser,
    extract_scenario_params,
    get_parser,
    telemetry_to_scenario,
    validate_telemetry,
)


class TestValidateTelemetry:
    """Tests for telemetry validation."""

    def test_valid_telemetry(self):
        """Test validation of valid telemetry data."""
        data = TelemetryData(
            source="test",
            memory_gb=16.0,
            throughput_toks_per_sec=40.0,
            latency_ms_per_request=25.0,
            tokens_per_request=1000,
        )
        issues = validate_telemetry(data)
        assert issues == []

    def test_invalid_memory(self):
        """Test validation catches invalid memory."""
        data = TelemetryData(
            source="test",
            memory_gb=-1.0,
            throughput_toks_per_sec=40.0,
            latency_ms_per_request=25.0,
            tokens_per_request=1000,
        )
        issues = validate_telemetry(data)
        assert any("memory_gb must be positive" in i for i in issues)

    def test_invalid_throughput(self):
        """Test validation catches invalid throughput."""
        data = TelemetryData(
            source="test",
            memory_gb=16.0,
            throughput_toks_per_sec=0.0,
            latency_ms_per_request=25.0,
            tokens_per_request=1000,
        )
        issues = validate_telemetry(data)
        assert any("throughput_toks_per_sec must be positive" in i for i in issues)

    def test_high_memory_warning(self):
        """Test validation warns about suspiciously high memory."""
        data = TelemetryData(
            source="test",
            memory_gb=2000.0,
            throughput_toks_per_sec=40.0,
            latency_ms_per_request=25.0,
            tokens_per_request=1000,
        )
        issues = validate_telemetry(data)
        assert any("memory_gb seems high" in i for i in issues)


class TestExtractScenarioParams:
    """Tests for extracting scenario parameters."""

    def test_basic_extraction(self):
        """Test basic parameter extraction."""
        telemetry = TelemetryData(
            source="test",
            parameters_b=7.0,
            memory_gb=16.0,
            throughput_toks_per_sec=40.0,
            latency_ms_per_request=25.0,
            tokens_per_request=1000,
            power_watts=320.0,
            requests_per_day=50000,
        )
        params = extract_scenario_params(telemetry)

        assert params["parameters_b"] == pytest.approx(7.0, rel=1e-9)
        assert params["memory_gb"] == pytest.approx(16.0, rel=1e-9)
        assert params["throughput_toks_per_sec"] == pytest.approx(40.0, rel=1e-9)
        assert params["power_watts"] == pytest.approx(320.0, rel=1e-9)
        assert params["requests_per_day"] == 50000
        assert params["tokens_per_request"] == 1000

    def test_power_estimation(self):
        """Test power estimation when not provided."""
        telemetry = TelemetryData(
            source="test",
            memory_gb=16.0,
            throughput_toks_per_sec=40.0,
            latency_ms_per_request=25.0,
            tokens_per_request=1000,
        )
        params = extract_scenario_params(telemetry)

        # Should estimate power as ~10W per GB
        assert params["power_watts"] == pytest.approx(160.0, rel=1e-9)

    def test_default_parameters(self):
        """Test default values for missing parameters."""
        telemetry = TelemetryData(
            source="test",
            memory_gb=16.0,
            throughput_toks_per_sec=40.0,
            latency_ms_per_request=25.0,
            tokens_per_request=1000,
        )
        params = extract_scenario_params(telemetry)

        assert params["parameters_b"] == pytest.approx(7.0, rel=1e-9)  # Default value
        assert params["requests_per_day"] == 50000  # Default value


class TestVLLMTelemetryParser:
    """Tests for vLLM telemetry parser."""

    def test_parse_vllm_json(self):
        """Test parsing vLLM benchmark JSON."""
        data = {
            "model": "meta-llama/Llama-2-7b-chat-hf",
            "metrics": {
                "throughput": 45.5,
                "mean_latency_ms": 22.0,
                "gpu_memory_gb": 14.0,
                "mean_input_tokens": 500,
                "mean_output_tokens": 500,
            },
        }
        parser = VLLMTelemetryParser()
        result = parser.parse(data)

        assert result.source == "vllm"
        assert result.parameters_b == pytest.approx(7.0, rel=1e-9)
        assert result.memory_gb == pytest.approx(14.0, rel=1e-9)
        assert result.throughput_toks_per_sec == pytest.approx(45.5, rel=1e-9)
        assert result.latency_ms_per_request == pytest.approx(22.0, rel=1e-9)
        assert result.tokens_per_request == pytest.approx(1000.0, rel=1e-9)

    def test_parse_vllm_string_json(self):
        """Test parsing vLLM JSON from string."""
        json_str = json.dumps(
            {
                "model": "codellama-34b",
                "metrics": {
                    "tokens_per_second": 30.0,
                    "latency_ms": 33.0,
                    "memory_gb": 40.0,
                },
            }
        )
        parser = VLLMTelemetryParser()
        result = parser.parse(json_str)

        assert result.parameters_b == pytest.approx(34.0, rel=1e-9)
        assert result.throughput_toks_per_sec == pytest.approx(30.0, rel=1e-9)

    def test_vllm_model_name_parsing(self):
        """Test extracting parameters from various model names."""
        parser = VLLMTelemetryParser()

        test_cases = [
            ("Llama-2-7b-chat-hf", 7.0),
            ("Llama-2-13b", 13.0),
            ("Llama-2-70b-chat", 70.0),
            ("codellama-34b-instruct", 34.0),
        ]

        for model_name, expected_params in test_cases:
            data = {
                "model": model_name,
                "metrics": {
                    "throughput": 40.0,
                    "latency_ms": 25.0,
                    "memory_gb": 16.0,
                },
            }
            result = parser.parse(data)
            assert result.parameters_b == pytest.approx(expected_params, rel=1e-9), (
                f"Failed for {model_name}"
            )

    def test_vllm_parse_file(self, tmp_path: Path):
        """Test parsing vLLM from file."""
        json_file = tmp_path / "vllm_output.json"
        json_file.write_text(
            json.dumps(
                {
                    "metrics": {
                        "throughput": 50.0,
                        "mean_latency_ms": 20.0,
                        "gpu_memory_gb": 16.0,
                    },
                }
            )
        )

        parser = VLLMTelemetryParser()
        result = parser.parse_file(json_file)

        assert result.throughput_toks_per_sec == pytest.approx(50.0, rel=1e-9)
        assert result.memory_gb == pytest.approx(16.0, rel=1e-9)


class TestTritonTelemetryParser:
    """Tests for Triton telemetry parser."""

    def test_parse_triton_json(self):
        """Test parsing Triton metrics JSON."""
        data = {
            "model_stats": [
                {
                    "inference_stats": {
                        "success": {"count": 6000},
                        "compute_infer": {"ns": 2_000_000_000},  # 2 seconds in ns
                    },
                    "batch_stats": {
                        "max_batch_size": 4,
                    },
                }
            ],
            "memory_gb": 16.0,
            "parameters_b": 7.0,
        }
        parser = TritonTelemetryParser()
        result = parser.parse(data)

        assert result.source == "triton"
        assert result.memory_gb == pytest.approx(16.0, rel=1e-9)
        assert result.parameters_b == pytest.approx(7.0, rel=1e-9)
        assert result.tokens_per_request == pytest.approx(4000.0, rel=1e-9)  # batch_size * 1000

    def test_triton_missing_model_stats(self):
        """Test error when model_stats is missing."""
        data = {"model_stats": []}
        parser = TritonTelemetryParser()

        with pytest.raises(ValueError, match="No model_stats found"):
            parser.parse(data)


class TestCSVTelemetryParser:
    """Tests for CSV telemetry parser."""

    def test_parse_csv_basic(self):
        """Test parsing basic CSV telemetry."""
        csv_data = """memory_gb,throughput_toks_per_sec,latency_ms_per_request,tokens_per_request
16.0,40.0,25.0,1000"""
        parser = CSVTelemetryParser()
        result = parser.parse(csv_data)

        assert result.source == "csv"
        assert result.memory_gb == pytest.approx(16.0, rel=1e-9)
        assert result.throughput_toks_per_sec == pytest.approx(40.0, rel=1e-9)
        assert result.latency_ms_per_request == pytest.approx(25.0, rel=1e-9)
        assert result.tokens_per_request == pytest.approx(1000.0, rel=1e-9)

    def test_parse_csv_with_mapping(self):
        """Test parsing CSV with field mapping."""
        csv_data = """gpu_mem,tok_per_sec,latency,tokens
32.0,80.0,12.5,2000"""
        mapping = {
            "memory_gb": "gpu_mem",
            "throughput_toks_per_sec": "tok_per_sec",
            "latency_ms_per_request": "latency",
            "tokens_per_request": "tokens",
        }
        parser = CSVTelemetryParser(field_mapping=mapping)
        result = parser.parse(csv_data)

        assert result.memory_gb == pytest.approx(32.0, rel=1e-9)
        assert result.throughput_toks_per_sec == pytest.approx(80.0, rel=1e-9)

    def test_csv_invalid_data(self):
        """Test error for invalid CSV data."""
        parser = CSVTelemetryParser()

        with pytest.raises(ValueError, match="CSV must have header"):
            parser.parse("invalid")

    def test_csv_requires_string(self):
        """Test that CSV parser rejects dict input."""
        parser = CSVTelemetryParser()

        with pytest.raises(ValueError, match="CSV parser expects string data"):
            parser.parse({"memory_gb": 16.0})


class TestGenericJSONTelemetryParser:
    """Tests for generic JSON telemetry parser."""

    def test_parse_basic_json(self):
        """Test parsing basic JSON telemetry."""
        data = {
            "memory_gb": 16.0,
            "throughput_toks_per_sec": 40.0,
            "latency_ms_per_request": 25.0,
            "tokens_per_request": 1000,
        }
        parser = GenericJSONTelemetryParser()
        result = parser.parse(data)

        assert result.source == "generic_json"
        assert result.memory_gb == pytest.approx(16.0, rel=1e-9)

    def test_parse_with_field_mapping(self):
        """Test parsing with field mapping."""
        data = {
            "gpu": {"memory": 16.0},
            "perf": {"tok_s": 40.0},
            "latency": 25.0,
            "tokens": 1000,
        }
        mapping = {
            "memory_gb": "gpu.memory",
            "throughput_toks_per_sec": "perf.tok_s",
            "latency_ms_per_request": "latency",
            "tokens_per_request": "tokens",
        }
        parser = GenericJSONTelemetryParser(field_mapping=mapping)
        result = parser.parse(data)

        assert result.memory_gb == pytest.approx(16.0, rel=1e-9)
        assert result.throughput_toks_per_sec == pytest.approx(40.0, rel=1e-9)

    def test_missing_required_fields(self):
        """Test error for missing required fields."""
        data = {"memory_gb": 16.0}  # Missing other fields
        parser = GenericJSONTelemetryParser()

        with pytest.raises(ValueError, match="throughput_toks_per_sec is required"):
            parser.parse(data)


class TestGetParser:
    """Tests for get_parser factory function."""

    def test_get_vllm_parser(self):
        """Test getting vLLM parser."""
        parser = get_parser("vllm")
        assert isinstance(parser, VLLMTelemetryParser)

    def test_get_triton_parser(self):
        """Test getting Triton parser."""
        parser = get_parser("triton")
        assert isinstance(parser, TritonTelemetryParser)

    def test_get_csv_parser(self):
        """Test getting CSV parser with mapping."""
        mapping = {"memory_gb": "gpu_memory"}
        parser = get_parser("csv", field_mapping=mapping)
        assert isinstance(parser, CSVTelemetryParser)

    def test_get_json_parser(self):
        """Test getting JSON parser with mapping."""
        mapping = {"throughput": "perf.throughput"}
        parser = get_parser("json", field_mapping=mapping)
        assert isinstance(parser, GenericJSONTelemetryParser)

    def test_invalid_format(self):
        """Test error for invalid format."""
        with pytest.raises(ValueError, match="Unknown format"):
            get_parser("invalid")


class TestTelemetryToScenario:
    """Tests for converting telemetry to scenario."""

    def test_basic_conversion(self):
        """Test basic telemetry to scenario conversion."""
        telemetry = TelemetryData(
            source="test",
            parameters_b=7.0,
            memory_gb=16.0,
            throughput_toks_per_sec=40.0,
            latency_ms_per_request=25.0,
            tokens_per_request=1000,
            power_watts=320.0,
            requests_per_day=50000,
        )
        scenario = telemetry_to_scenario(telemetry, name="my-deployment")

        assert isinstance(scenario, DeploymentScenario)
        assert scenario.name == "my-deployment"
        assert scenario.parameters_b == pytest.approx(7.0, rel=1e-9)
        assert scenario.memory_gb == pytest.approx(16.0, rel=1e-9)

    def test_conversion_with_overrides(self):
        """Test conversion with parameter overrides."""
        telemetry = TelemetryData(
            source="test",
            memory_gb=16.0,
            throughput_toks_per_sec=40.0,
            latency_ms_per_request=25.0,
            tokens_per_request=1000,
        )
        scenario = telemetry_to_scenario(
            telemetry,
            name="custom-deployment",
            electricity_cost_per_kwh=0.20,
            requests_per_day=100000,
        )

        assert scenario.electricity_cost_per_kwh == pytest.approx(0.20, rel=1e-9)
        assert scenario.requests_per_day == 100000
