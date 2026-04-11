"""Serialization and deserialization utilities for canonical trajectories."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .schema import RewardSignal, TrajectoryRecord, TrajectoryStep, record_to_dict


@dataclass(frozen=True, slots=True)
class ArrowTableAdapter:
    """Simple adapter to avoid hard-depending on pyarrow."""

    rows: list[dict[str, Any]]



def to_json(record: TrajectoryRecord, *, indent: int = 2) -> str:
    """Serialize a trajectory record to JSON."""

    return json.dumps(record_to_dict(record), indent=indent, sort_keys=True)


def from_json(payload: str) -> TrajectoryRecord:
    """Deserialize JSON payload into a validated trajectory record."""

    data = json.loads(payload)
    steps = [
        TrajectoryStep(
            step_idx=int(step["step_idx"]),
            tokens_in=[int(token) for token in step.get("tokens_in", [])],
            tokens_out=[int(token) for token in step.get("tokens_out", [])],
            reward=RewardSignal(
                total=float(step.get("reward", {}).get("total", 0.0)),
                components={
                    str(name): float(value)
                    for name, value in step.get("reward", {}).get("components", {}).items()
                },
                source=str(step.get("reward", {}).get("source", "unknown")),
            ),
            action=step.get("action"),
            observation=step.get("observation"),
            next_observation=step.get("next_observation"),
            done=bool(step.get("done", False)),
            metadata=dict(step.get("metadata", {})),
        )
        for step in data.get("steps", [])
    ]
    record = TrajectoryRecord(
        schema_name=str(data.get("schema_name", "")),
        schema_version=str(data.get("schema_version", "")),
        trajectory_id=str(data["trajectory_id"]),
        episode_id=str(data["episode_id"]),
        created_at=str(data["created_at"]),
        steps=steps,
        metadata=dict(data.get("metadata", {})),
        environment_state=data.get("environment_state"),
    )
    record.validate()
    return record


def write_json(record: TrajectoryRecord, path: Path) -> Path:
    """Persist a canonical trajectory JSON file."""

    path.write_text(to_json(record), encoding="utf-8")
    return path


def read_json(path: Path) -> TrajectoryRecord:
    """Read and deserialize a canonical trajectory JSON file."""

    return from_json(path.read_text(encoding="utf-8"))


def to_arrow_rows(record: TrajectoryRecord) -> ArrowTableAdapter:
    """Flatten a trajectory record to Arrow-compatible row dictionaries."""

    record.validate()
    rows = []
    for step in record.steps:
        rows.append(
            {
                "schema_name": record.schema_name,
                "schema_version": record.schema_version,
                "trajectory_id": record.trajectory_id,
                "episode_id": record.episode_id,
                "created_at": record.created_at,
                "step_idx": step.step_idx,
                "tokens_in": step.tokens_in,
                "tokens_out": step.tokens_out,
                "reward_total": step.reward.total,
                "reward_components": step.reward.components,
                "reward_source": step.reward.source,
                "action": json.dumps(step.action, sort_keys=True),
                "observation": json.dumps(step.observation, sort_keys=True),
                "next_observation": json.dumps(step.next_observation, sort_keys=True),
                "done": step.done,
                "step_metadata": json.dumps(step.metadata, sort_keys=True),
                "trajectory_metadata": json.dumps(record.metadata, sort_keys=True),
                "environment_state": json.dumps(record.environment_state, sort_keys=True),
            }
        )
    return ArrowTableAdapter(rows=rows)


def from_arrow_rows(rows: list[dict[str, Any]]) -> TrajectoryRecord:
    """Reconstruct a trajectory record from Arrow-style rows."""

    if not rows:
        raise ValueError("Cannot build TrajectoryRecord from empty rows.")
    ordered_rows = sorted(rows, key=lambda row: int(row["step_idx"]))
    first = ordered_rows[0]
    steps: list[TrajectoryStep] = []
    for row in ordered_rows:
        steps.append(
            TrajectoryStep(
                step_idx=int(row["step_idx"]),
                tokens_in=[int(token) for token in row.get("tokens_in", [])],
                tokens_out=[int(token) for token in row.get("tokens_out", [])],
                reward=RewardSignal(
                    total=float(row.get("reward_total", 0.0)),
                    components={
                        str(name): float(value)
                        for name, value in row.get("reward_components", {}).items()
                    },
                    source=str(row.get("reward_source", "unknown")),
                ),
                action=json.loads(row.get("action", "null")),
                observation=json.loads(row.get("observation", "null")),
                next_observation=json.loads(row.get("next_observation", "null")),
                done=bool(row.get("done", False)),
                metadata=json.loads(row.get("step_metadata", "{}")),
            )
        )
    record = TrajectoryRecord(
        schema_name=str(first["schema_name"]),
        schema_version=str(first["schema_version"]),
        trajectory_id=str(first["trajectory_id"]),
        episode_id=str(first["episode_id"]),
        created_at=str(first["created_at"]),
        steps=steps,
        metadata=json.loads(first.get("trajectory_metadata", "{}")),
        environment_state=json.loads(first.get("environment_state", "null")),
    )
    record.validate()
    return record
