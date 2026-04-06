"""Configuration management for Atropos."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import yaml  # type: ignore[import-untyped]


@dataclass
class AtroposConfig:
    """Global configuration for Atropos."""

    grid_co2e_factor: float = 0.35
    hardware_savings_correlation: float = 0.8
    default_report_format: str = "text"
    risk_rank: dict[str, int] = field(default_factory=lambda: {"low": 1, "medium": 2, "high": 3})

    def __post_init__(self) -> None:
        """Initialize reverse risk mapping."""
        self.reverse_risk = {v: k for k, v in self.risk_rank.items()}

    @classmethod
    def from_env(cls) -> AtroposConfig:
        """Load configuration from environment variables.

        Variables:
            ATROPOS_GRID_CO2E: Grid CO2e factor (default: 0.35).
            ATROPOS_HW_SAVINGS_CORR: Hardware savings correlation (default: 0.8).
            ATROPOS_REPORT_FORMAT: Default report format (default: "text").
        """
        return cls(
            grid_co2e_factor=float(os.getenv("ATROPOS_GRID_CO2E", "0.35")),
            hardware_savings_correlation=float(os.getenv("ATROPOS_HW_SAVINGS_CORR", "0.8")),
            default_report_format=os.getenv("ATROPOS_REPORT_FORMAT", "text"),
        )

    @classmethod
    def from_file(cls, path: str | Path) -> AtroposConfig:
        """Load configuration from a YAML file.

        Args:
            path: Path to the YAML configuration file.

        Returns:
            AtroposConfig instance.

        Raises:
            ValueError: If the file does not contain a valid YAML mapping.
        """
        data = yaml.safe_load(Path(path).read_text())
        if not isinstance(data, dict):
            raise ValueError("Config file must contain a YAML mapping/object.")
        return cls(**data)
