"""Canonical, versioned trajectory schema for Atropos."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

SCHEMA_NAME = "atropos.trajectory"
SCHEMA_VERSION = "1.0.0"


JSONValue = dict[str, "JSONValue"] | list["JSONValue"] | str | int | float | bool | None


@dataclass(slots=True)
class RewardSignal:
    """Reward payload suitable for RLHF and offline RL pipelines."""

    total: float
    components: dict[str, float] = field(default_factory=dict)
    source: str = "unknown"


@dataclass(slots=True)
class TrajectoryStep:
    """One transition in an episode trajectory."""

    step_idx: int
    tokens_in: list[int] = field(default_factory=list)
    tokens_out: list[int] = field(default_factory=list)
    reward: RewardSignal = field(default_factory=lambda: RewardSignal(total=0.0))
    action: JSONValue = None
    observation: JSONValue = None
    next_observation: JSONValue = None
    done: bool = False
    metadata: dict[str, JSONValue] = field(default_factory=dict)


@dataclass(slots=True)
class TrajectoryRecord:
    """Top-level canonical trajectory envelope."""

    schema_name: str = SCHEMA_NAME
    schema_version: str = SCHEMA_VERSION
    trajectory_id: str = field(default_factory=lambda: f"traj_{uuid4().hex}")
    episode_id: str = field(default_factory=lambda: f"ep_{uuid4().hex}")
    created_at: str = field(
        default_factory=lambda: datetime.now(tz=timezone.utc).isoformat(timespec="seconds")
    )
    steps: list[TrajectoryStep] = field(default_factory=list)
    metadata: dict[str, JSONValue] = field(default_factory=dict)
    environment_state: JSONValue = None

    def validate(self) -> None:
        """Validate required schema invariants."""

        if self.schema_name != SCHEMA_NAME:
            raise ValueError(f"Unsupported schema name: {self.schema_name}.")
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError(f"Unsupported schema version: {self.schema_version}.")
        expected_step = 1
        for step in self.steps:
            if step.step_idx != expected_step:
                raise ValueError("Step indices must be contiguous and start at 1.")
            expected_step += 1


@dataclass(frozen=True, slots=True)
class TrajectorySchemaSpec:
    """Schema metadata for JSON and Arrow representations."""

    schema_name: str
    schema_version: str
    json_schema: dict[str, Any]
    arrow_columns: dict[str, str]


def build_schema_spec() -> TrajectorySchemaSpec:
    """Return a machine-readable schema spec."""

    return TrajectorySchemaSpec(
        schema_name=SCHEMA_NAME,
        schema_version=SCHEMA_VERSION,
        json_schema={
            "type": "object",
            "required": [
                "schema_name",
                "schema_version",
                "trajectory_id",
                "episode_id",
                "created_at",
                "steps",
                "metadata",
            ],
            "properties": {
                "schema_name": {"type": "string", "const": SCHEMA_NAME},
                "schema_version": {"type": "string", "const": SCHEMA_VERSION},
                "trajectory_id": {"type": "string"},
                "episode_id": {"type": "string"},
                "created_at": {"type": "string", "format": "date-time"},
                "steps": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "required": ["step_idx", "tokens_in", "tokens_out", "reward", "done"],
                    },
                },
                "metadata": {"type": "object"},
                "environment_state": {},
            },
        },
        arrow_columns={
            "schema_name": "string",
            "schema_version": "string",
            "trajectory_id": "string",
            "episode_id": "string",
            "created_at": "string",
            "step_idx": "int64",
            "tokens_in": "list<int64>",
            "tokens_out": "list<int64>",
            "reward_total": "float64",
            "reward_components": "struct",
            "reward_source": "string",
            "action": "string",
            "observation": "string",
            "next_observation": "string",
            "done": "bool",
            "step_metadata": "string",
            "trajectory_metadata": "string",
            "environment_state": "string",
        },
    )


def record_to_dict(record: TrajectoryRecord) -> dict[str, Any]:
    """Convert a canonical record to a JSON-serializable dict."""

    record.validate()
    return asdict(record)
