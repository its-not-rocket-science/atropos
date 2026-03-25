"""Custom exceptions for Atropos error handling."""

from __future__ import annotations


class AtroposError(Exception):
    """Base exception for all Atropos-specific errors."""

    def __init__(self, message: str, context: dict[str, object] | None = None):
        """Initialize Atropos error.

        Args:
            message: Human-readable error description.
            context: Optional context dictionary with additional debugging information.
        """
        super().__init__(message)
        self.context = context or {}
        self.message = message

    def __str__(self) -> str:
        """Format error with context if available."""
        if self.context:
            context_str = ", ".join(f"{k}={v!r}" for k, v in self.context.items())
            return f"{self.message} [{context_str}]"
        return self.message


class ValidationError(AtroposError):
    """Error raised when validation fails or invalid inputs are provided."""


class ConfigurationError(AtroposError):
    """Error raised for configuration issues (missing, invalid, or conflicting settings)."""


class CalculationError(AtroposError):
    """Error raised during ROI calculation or optimization estimation."""


class IOError(AtroposError):
    """Error raised for file I/O operations (reading/writing scenarios, reports, etc.)."""


class PipelineError(AtroposError):
    """Error raised during pipeline execution stages."""


class FrameworkError(AtroposError):
    """Error raised when pruning framework integration fails."""


class ModelError(AtroposError):
    """Error raised for model-related issues (loading, analyzing, pruning)."""


class TelemetryError(AtroposError):
    """Error raised during telemetry collection or processing."""


class CalibrationError(AtroposError):
    """Error raised during calibration of scenario parameters."""


class QualityBenchmarkError(AtroposError):
    """Error raised during quality benchmarking."""


class DeploymentError(AtroposError):
    """Error raised during deployment operations."""


class ResourceError(AtroposError):
    """Error raised for resource constraints (memory, GPU, etc.)."""


# Compatibility alias for existing ImportErrorException
class ImportErrorException(AtroposError):  # noqa: N818
    """Exception raised when required packages not installed.

    Note: Kept for backward compatibility with existing code.
    """

    pass
