"""Pipeline configuration models."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from ..models import QualityRisk


@dataclass(frozen=True)
class ThresholdConfig:
    """Thresholds for pipeline gating decisions.

    Attributes:
        max_break_even_months: Maximum acceptable break-even time in months.
        min_annual_savings_usd: Minimum annual savings to proceed (USD).
        max_quality_risk: Maximum acceptable quality risk level.
    """

    max_break_even_months: int = 12
    min_annual_savings_usd: float = 10_000.0
    max_quality_risk: QualityRisk = "medium"

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "max_break_even_months": self.max_break_even_months,
            "min_annual_savings_usd": self.min_annual_savings_usd,
            "max_quality_risk": self.max_quality_risk,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ThresholdConfig:
        """Create from dictionary."""
        return cls(
            max_break_even_months=data.get("max_break_even_months", 12),
            min_annual_savings_usd=data.get("min_annual_savings_usd", 10_000.0),
            max_quality_risk=data.get("max_quality_risk", "medium"),
        )


@dataclass(frozen=True)
class PruningConfig:
    """Configuration for the pruning stage.

    Attributes:
        framework: Pruning framework to use (llm-pruner, wanda, sparsegpt, custom).
        target_sparsity: Target sparsity level (0-1).
        structured: Whether to use structured pruning.
        custom_command: Custom command for framework integration.
        output_path: Path to save pruned model.
        distributed: Whether to use distributed pruning across multiple GPUs.
        num_gpus: Number of GPUs to use for distributed pruning.
        parallel_strategy: Parallelization strategy (data, layer, or model).
        distributed_backend: Distributed backend (nccl, gloo, mpi).
    """

    framework: Literal[
        "llm-pruner", "wanda", "sparsegpt", "custom", "wanda-patched", "sparsegpt-patched"
    ] = "llm-pruner"
    target_sparsity: float = 0.30
    structured: bool = True
    custom_command: str | None = None
    output_path: str | None = None
    distributed: bool = False
    num_gpus: int = 1
    parallel_strategy: Literal["data", "layer", "model"] = "data"
    distributed_backend: str = "nccl"

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        result: dict[str, Any] = {
            "framework": self.framework,
            "target_sparsity": self.target_sparsity,
            "structured": self.structured,
            "distributed": self.distributed,
            "num_gpus": self.num_gpus,
            "parallel_strategy": self.parallel_strategy,
            "distributed_backend": self.distributed_backend,
        }
        if self.custom_command:
            result["custom_command"] = self.custom_command
        if self.output_path:
            result["output_path"] = self.output_path
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PruningConfig:
        """Create from dictionary."""
        return cls(
            framework=data.get("framework", "llm-pruner"),
            target_sparsity=data.get("target_sparsity", 0.30),
            structured=data.get("structured", True),
            custom_command=data.get("custom_command"),
            output_path=data.get("output_path"),
            distributed=data.get("distributed", False),
            num_gpus=data.get("num_gpus", 1),
            parallel_strategy=data.get("parallel_strategy", "data"),
            distributed_backend=data.get("distributed_backend", "nccl"),
        )


@dataclass(frozen=True)
class RecoveryConfig:
    """Configuration for the recovery/fine-tuning stage.

    Attributes:
        enabled: Whether to run fine-tuning recovery.
        epochs: Number of fine-tuning epochs.
        learning_rate: Learning rate for recovery.
        batch_size: Batch size for recovery training.
        custom_command: Custom command for recovery framework.
    """

    enabled: bool = True
    epochs: int = 3
    learning_rate: float = 5e-5
    batch_size: int = 32
    custom_command: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        result: dict[str, Any] = {
            "enabled": self.enabled,
            "epochs": self.epochs,
            "learning_rate": self.learning_rate,
            "batch_size": self.batch_size,
        }
        if self.custom_command:
            result["custom_command"] = self.custom_command
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RecoveryConfig:
        """Create from dictionary."""
        return cls(
            enabled=data.get("enabled", True),
            epochs=data.get("epochs", 3),
            learning_rate=data.get("learning_rate", 5e-5),
            batch_size=data.get("batch_size", 32),
            custom_command=data.get("custom_command"),
        )


@dataclass(frozen=True)
class ValidationConfig:
    """Configuration for the validation stage.

    Attributes:
        tolerance_percent: Tolerance for metric variance (percentage).
        quality_benchmark: Benchmark to use for quality validation.
        min_quality_score: Minimum acceptable quality score.
        benchmark_command: Custom command for benchmarking.
    """

    tolerance_percent: float = 10.0
    quality_benchmark: str = "humaneval"
    min_quality_score: float | None = None
    benchmark_command: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        result: dict[str, Any] = {
            "tolerance_percent": self.tolerance_percent,
            "quality_benchmark": self.quality_benchmark,
        }
        if self.min_quality_score is not None:
            result["min_quality_score"] = self.min_quality_score
        if self.benchmark_command:
            result["benchmark_command"] = self.benchmark_command
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ValidationConfig:
        """Create from dictionary."""
        return cls(
            tolerance_percent=data.get("tolerance_percent", 10.0),
            quality_benchmark=data.get("quality_benchmark", "humaneval"),
            min_quality_score=data.get("min_quality_score"),
            benchmark_command=data.get("benchmark_command"),
        )


@dataclass(frozen=True)
class DeploymentConfig:
    """Configuration for the deployment stage.

    Attributes:
        auto_deploy: Whether to auto-deploy on successful validation.
        deployment_command: Command to deploy the optimized model.
        rollback_command: Command to rollback if deployment fails.
        canary_percent: Percentage of traffic for canary deployment.
    """

    auto_deploy: bool = False
    deployment_command: str | None = None
    rollback_command: str | None = None
    canary_percent: float = 10.0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        result: dict[str, Any] = {
            "auto_deploy": self.auto_deploy,
            "canary_percent": self.canary_percent,
        }
        if self.deployment_command:
            result["deployment_command"] = self.deployment_command
        if self.rollback_command:
            result["rollback_command"] = self.rollback_command
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DeploymentConfig:
        """Create from dictionary."""
        return cls(
            auto_deploy=data.get("auto_deploy", False),
            deployment_command=data.get("deployment_command"),
            rollback_command=data.get("rollback_command"),
            canary_percent=data.get("canary_percent", 10.0),
        )


@dataclass(frozen=True)
class PipelineConfig:
    """Complete pipeline configuration.

    Attributes:
        name: Pipeline name/identifier.
        auto_execute: Whether to auto-execute without confirmation.
        thresholds: Gating thresholds for ROI assessment.
        pruning: Pruning stage configuration.
        recovery: Recovery/fine-tuning configuration.
        validation: Validation stage configuration.
        deployment: Deployment stage configuration.
    """

    name: str = "atropos-pipeline"
    auto_execute: bool = False
    thresholds: ThresholdConfig | None = None
    pruning: PruningConfig | None = None
    recovery: RecoveryConfig | None = None
    validation: ValidationConfig | None = None
    deployment: DeploymentConfig | None = None

    def __post_init__(self) -> None:
        """Set default configs if not provided."""
        # This is a workaround for frozen dataclass with mutable defaults
        object.__setattr__(self, "thresholds", self.thresholds or ThresholdConfig())
        object.__setattr__(self, "pruning", self.pruning or PruningConfig())
        object.__setattr__(self, "recovery", self.recovery or RecoveryConfig())
        object.__setattr__(self, "validation", self.validation or ValidationConfig())
        object.__setattr__(self, "deployment", self.deployment or DeploymentConfig())

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "name": self.name,
            "auto_execute": self.auto_execute,
            "thresholds": self.thresholds.to_dict() if self.thresholds else {},
            "pruning": self.pruning.to_dict() if self.pruning else {},
            "recovery": self.recovery.to_dict() if self.recovery else {},
            "validation": self.validation.to_dict() if self.validation else {},
            "deployment": self.deployment.to_dict() if self.deployment else {},
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PipelineConfig:
        """Create from dictionary (e.g., from YAML config)."""
        return cls(
            name=data.get("name", "atropos-pipeline"),
            auto_execute=data.get("auto_execute", False),
            thresholds=ThresholdConfig.from_dict(data.get("thresholds", {})),
            pruning=PruningConfig.from_dict(data.get("pruning", {})),
            recovery=RecoveryConfig.from_dict(data.get("recovery", {})),
            validation=ValidationConfig.from_dict(data.get("validation", {})),
            deployment=DeploymentConfig.from_dict(data.get("deployment", {})),
        )

    @classmethod
    def from_yaml(cls, path: Path) -> PipelineConfig:
        """Load configuration from YAML file."""
        import yaml

        with open(path) as f:
            data = yaml.safe_load(f)
        return cls.from_dict(data.get("pipeline", data))

    def to_yaml(self, path: Path) -> None:
        """Save configuration to YAML file."""
        import yaml

        with open(path, "w") as f:
            yaml.dump({"pipeline": self.to_dict()}, f, default_flow_style=False)
