"""Pipeline execution models and stage definitions."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..models import OptimizationOutcome


class PipelineStage(Enum):
    """Pipeline execution stages."""

    ASSESS = auto()
    GATE = auto()
    PRUNE = auto()
    RECOVER = auto()
    VALIDATE = auto()
    DEPLOY = auto()
    ROLLBACK = auto()

    def __str__(self) -> str:
        """Return human-readable stage name."""
        return self.name.lower()


class StageStatus(Enum):
    """Status of a pipeline stage execution."""

    PENDING = auto()
    RUNNING = auto()
    SUCCESS = auto()
    FAILED = auto()
    SKIPPED = auto()
    WARNED = auto()

    def __str__(self) -> str:
        """Return human-readable status."""
        return self.name.lower()


@dataclass
class StageResult:
    """Result of executing a single pipeline stage.

    Attributes:
        stage: The pipeline stage that was executed.
        status: Execution status (success, failed, skipped, etc.).
        message: Human-readable result message.
        start_time: Execution start timestamp (ISO format).
        end_time: Execution end timestamp (ISO format).
        metrics: Stage-specific metrics and outputs.
        artifacts: Paths to generated artifacts (models, logs, etc.).
    """

    stage: PipelineStage
    status: StageStatus
    message: str = ""
    start_time: str | None = None
    end_time: str | None = None
    metrics: dict[str, Any] = field(default_factory=dict)
    artifacts: list[str] = field(default_factory=list)

    @property
    def duration_seconds(self) -> float | None:
        """Calculate stage duration in seconds."""
        if self.start_time and self.end_time:
            from datetime import datetime

            start = datetime.fromisoformat(self.start_time)
            end = datetime.fromisoformat(self.end_time)
            return (end - start).total_seconds()
        return None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "stage": str(self.stage),
            "status": str(self.status),
            "message": self.message,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration_seconds": self.duration_seconds,
            "metrics": self.metrics,
            "artifacts": self.artifacts,
        }


@dataclass
class PipelineResult:
    """Complete pipeline execution result.

    Attributes:
        pipeline_name: Name of the pipeline configuration.
        scenario_name: Name of the scenario being optimized.
        strategy_name: Name of the optimization strategy.
        stages: Results for each executed stage.
        final_status: Overall pipeline status.
        overall_message: Summary message.
        start_time: Pipeline start timestamp.
        end_time: Pipeline end timestamp.
        roi_outcome: ROI assessment outcome (if Assess stage ran).
    """

    pipeline_name: str
    scenario_name: str
    strategy_name: str
    stages: list[StageResult] = field(default_factory=list)
    final_status: StageStatus = StageStatus.PENDING
    overall_message: str = ""
    start_time: str | None = None
    end_time: str | None = None
    roi_outcome: OptimizationOutcome | None = None

    @property
    def duration_seconds(self) -> float | None:
        """Calculate total pipeline duration in seconds."""
        if self.start_time and self.end_time:
            from datetime import datetime

            start = datetime.fromisoformat(self.start_time)
            end = datetime.fromisoformat(self.end_time)
            return (end - start).total_seconds()
        return None

    @property
    def passed_gate(self) -> bool | None:
        """Check if the pipeline passed the gate stage."""
        for stage in self.stages:
            if stage.stage == PipelineStage.GATE:
                return stage.status == StageStatus.SUCCESS
        return None

    def get_stage(self, stage: PipelineStage) -> StageResult | None:
        """Get result for a specific stage."""
        for s in self.stages:
            if s.stage == stage:
                return s
        return None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        result: dict[str, Any] = {
            "pipeline_name": self.pipeline_name,
            "scenario_name": self.scenario_name,
            "strategy_name": self.strategy_name,
            "final_status": str(self.final_status),
            "overall_message": self.overall_message,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration_seconds": self.duration_seconds,
            "passed_gate": self.passed_gate,
            "stages": [s.to_dict() for s in self.stages],
        }
        if self.roi_outcome:
            result["roi_outcome"] = {
                "annual_savings": self.roi_outcome.annual_total_savings_usd,
                "break_even_years": self.roi_outcome.break_even_years,
                "quality_risk": self.roi_outcome.quality_risk,
            }
        return result

    def to_json(self, indent: int = 2) -> str:
        """Convert to JSON string."""
        import json

        return json.dumps(self.to_dict(), indent=indent)
