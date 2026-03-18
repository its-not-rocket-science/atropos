"""Tests for I/O utilities."""

import tempfile
from pathlib import Path

import pytest
import yaml

from atropos.io import load_scenario


def test_load_valid_yaml() -> None:
    """Test loading a valid YAML scenario file."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        yaml.dump(
            {
                "name": "test-scenario",
                "parameters_b": 34.0,
                "memory_gb": 14.0,
                "throughput_toks_per_sec": 40.0,
                "power_watts": 320.0,
                "requests_per_day": 50000,
                "tokens_per_request": 1200,
                "electricity_cost_per_kwh": 0.15,
                "annual_hardware_cost_usd": 24000.0,
                "one_time_project_cost_usd": 27000.0,
            },
            f,
        )
        f.flush()
        scenario = load_scenario(f.name)
    assert scenario.name == "test-scenario"
    assert scenario.parameters_b == pytest.approx(34.0, rel=1e-9)
    Path(f.name).unlink()


def test_load_invalid_yaml() -> None:
    """Test that invalid YAML raises an error."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write("invalid: yaml: content: [unclosed")
        f.flush()
        with pytest.raises(yaml.YAMLError):
            load_scenario(f.name)
    Path(f.name).unlink()


def test_missing_keys() -> None:
    """Test that missing required keys raises a ValueError."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        yaml.dump({"name": "incomplete"}, f)
        f.flush()
        with pytest.raises(ValueError, match="missing required keys"):
            load_scenario(f.name)
    Path(f.name).unlink()
