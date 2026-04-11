"""Conversion tools into the canonical trajectory schema."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

from atropos.rl_env.contracts import LineWorldStepRecord

from .schema import RewardSignal, TrajectoryRecord, TrajectoryStep


def from_line_world_history(
    history: list[LineWorldStepRecord],
    *,
    trajectory_id: str = "lineworld-history",
    episode_id: str = "lineworld-episode",
    metadata: dict[str, Any] | None = None,
) -> TrajectoryRecord:
    """Convert Atropos LineWorld history into a canonical trajectory."""

    steps: list[TrajectoryStep] = []
    for record in history:
        steps.append(
            TrajectoryStep(
                step_idx=record.step_idx,
                tokens_in=[],
                tokens_out=[],
                reward=RewardSignal(total=record.reward, source="line_world"),
                action={"delta": record.action},
                observation={"position": record.position_before},
                next_observation={"position": record.position_after},
                done=record.done,
                metadata={"introspection": asdict(record.introspection)},
            )
        )
    canonical = TrajectoryRecord(
        trajectory_id=trajectory_id,
        episode_id=episode_id,
        steps=steps,
        metadata={"source_format": "line_world_history", **(metadata or {})},
    )
    canonical.validate()
    return canonical


def from_line_world_rollout(payload: dict[str, Any]) -> TrajectoryRecord:
    """Convert legacy ``LineWorldEnv.save_rollout`` payload into canonical schema."""

    history = payload.get("history", [])
    steps: list[TrajectoryStep] = []
    for idx, item in enumerate(history, start=1):
        introspection = item.get("introspection", {})
        steps.append(
            TrajectoryStep(
                step_idx=idx,
                reward=RewardSignal(total=float(item.get("reward", 0.0)), source="line_world"),
                action={"delta": int(item["action"])},
                observation={"position": int(item.get("position_before", 0))},
                next_observation={"position": int(item.get("position_after", 0))},
                done=bool(item.get("done", False)),
                metadata={"introspection": introspection},
            )
        )
    canonical = TrajectoryRecord(
        trajectory_id="lineworld-rollout",
        episode_id="lineworld-rollout-episode",
        steps=steps,
        metadata={"source_format": "line_world_rollout", "seed": payload.get("seed")},
        environment_state=payload.get("initial_state"),
    )
    canonical.validate()
    return canonical


def from_rlhf_pairs(samples: list[dict[str, Any]]) -> TrajectoryRecord:
    """Convert common RLHF pairwise/preference samples into canonical schema."""

    steps: list[TrajectoryStep] = []
    for idx, sample in enumerate(samples, start=1):
        steps.append(
            TrajectoryStep(
                step_idx=idx,
                tokens_in=[int(token) for token in sample.get("prompt_tokens", [])],
                tokens_out=[int(token) for token in sample.get("response_tokens", [])],
                reward=RewardSignal(
                    total=float(sample.get("reward", sample.get("score", 0.0))),
                    components={"preference": float(sample.get("preference_score", 0.0))},
                    source="rlhf",
                ),
                action={"response_text": sample.get("response")},
                observation={"prompt": sample.get("prompt")},
                next_observation={"label": sample.get("label")},
                done=idx == len(samples),
                metadata={
                    "chosen": bool(sample.get("chosen", False)),
                    "policy_id": sample.get("policy_id", "unknown"),
                },
            )
        )
    canonical = TrajectoryRecord(
        trajectory_id="rlhf-trajectory",
        episode_id="rlhf-episode",
        steps=steps,
        metadata={"source_format": "rlhf_pairs"},
    )
    canonical.validate()
    return canonical


def from_offline_rl_transitions(transitions: list[dict[str, Any]]) -> TrajectoryRecord:
    """Convert offline RL transition datasets to canonical schema."""

    steps: list[TrajectoryStep] = []
    for idx, item in enumerate(transitions, start=1):
        steps.append(
            TrajectoryStep(
                step_idx=idx,
                tokens_in=[int(token) for token in item.get("obs_tokens", [])],
                tokens_out=[int(token) for token in item.get("action_tokens", [])],
                reward=RewardSignal(
                    total=float(item.get("reward", 0.0)),
                    components={
                        str(name): float(value)
                        for name, value in item.get("reward_components", {}).items()
                    },
                    source="offline_rl",
                ),
                action=item.get("action"),
                observation=item.get("observation"),
                next_observation=item.get("next_observation"),
                done=bool(item.get("done", False)),
                metadata={
                    "discount": float(item.get("discount", 1.0)),
                    "dataset": item.get("dataset", "unknown"),
                },
            )
        )
    canonical = TrajectoryRecord(
        trajectory_id="offline-rl-trajectory",
        episode_id="offline-rl-episode",
        steps=steps,
        metadata={"source_format": "offline_rl_transitions"},
    )
    canonical.validate()
    return canonical
