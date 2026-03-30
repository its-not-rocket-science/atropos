"""Data models for deployment automation."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import Any


class DeploymentStatus(Enum):
    """Status of a deployment operation."""

    PENDING = auto()
    RUNNING = auto()
    SUCCESS = auto()
    FAILED = auto()
    ROLLED_BACK = auto()

    def __str__(self) -> str:
        return self.name.lower()


class DeploymentStrategyType(Enum):
    """Type of deployment strategy."""

    IMMEDIATE = auto()  # Full immediate deployment
    CANARY = auto()  # Gradual traffic shift
    BLUE_GREEN = auto()  # Alternate environment swap
    ROLLING = auto()  # Incremental instance updates
    A_B_TEST = auto()  # A/B testing with multiple variants

    def __str__(self) -> str:
        return self.name.lower().replace("_", "-")


@dataclass
class DeploymentRequest:
    """Request to deploy a model.

    Attributes:
        model_path: Path to the model (local or remote).
        platform: Deployment platform identifier (e.g., 'vllm', 'triton').
        strategy: Deployment strategy type.
        strategy_config: Configuration specific to the strategy.
        experiment_config: A/B testing experiment configuration (optional).
        health_checks: Health check configuration.
        metadata: Additional metadata for the deployment.
    """

    model_path: str
    platform: str = "vllm"
    strategy: DeploymentStrategyType = DeploymentStrategyType.IMMEDIATE
    strategy_config: dict[str, Any] = field(default_factory=dict)
    experiment_config: dict[str, Any] = field(default_factory=dict)
    health_checks: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class DeploymentResult:
    """Result of a deployment operation.

    Attributes:
        request: The deployment request.
        status: Final deployment status.
        message: Human-readable result message.
        start_time: Deployment start timestamp.
        end_time: Deployment end timestamp.
        deployment_id: Platform-specific deployment identifier.
        endpoints: List of serving endpoints (URLs).
        health_check_results: Results of health checks.
        metrics: Deployment metrics (latency, error rate, etc.).
    """

    request: DeploymentRequest
    status: DeploymentStatus
    message: str = ""
    start_time: str | None = None
    end_time: str | None = None
    deployment_id: str | None = None
    endpoints: list[str] = field(default_factory=list)
    health_check_results: dict[str, Any] = field(default_factory=dict)
    metrics: dict[str, Any] = field(default_factory=dict)

    @property
    def duration_seconds(self) -> float | None:
        """Calculate deployment duration in seconds."""
        if self.start_time and self.end_time:
            start = datetime.fromisoformat(self.start_time)
            end = datetime.fromisoformat(self.end_time)
            return (end - start).total_seconds()
        return None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "request": {
                "model_path": self.request.model_path,
                "platform": self.request.platform,
                "strategy": str(self.request.strategy),
                "strategy_config": self.request.strategy_config,
                "experiment_config": self.request.experiment_config,
                "health_checks": self.request.health_checks,
                "metadata": self.request.metadata,
            },
            "status": str(self.status),
            "message": self.message,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration_seconds": self.duration_seconds,
            "deployment_id": self.deployment_id,
            "endpoints": self.endpoints,
            "health_check_results": self.health_check_results,
            "metrics": self.metrics,
        }
