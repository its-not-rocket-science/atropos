"""Minimal, complete trajectory pipeline example.

This example now uses decomposed components:
1) Generator
2) Parser
3) RewardFunction
4) TrajectoryBuilder
5) EnvOrchestrator (via LineWorldEnv facade)
"""

from __future__ import annotations

import json
from pathlib import Path

from atropos.rl_env.contracts import LineWorldState
from atropos.rl_env.line_world import LineWorldEnv, as_dict


def greedy_policy(state: LineWorldState) -> int:
    return 1 if state.position < state.goal else -1


def generate_trajectory(env: LineWorldEnv) -> list[object]:
    records: list[object] = []
    state = env.reset()

    while True:
        action = greedy_policy(state)
        record = env.step(action)
        records.append(record)
        if record.done:
            break

    return records


def score_trajectory(records: list[object], goal: int) -> dict[str, float | int | bool]:
    total_reward = sum(r.reward for r in records)
    steps = len(records)
    reached_goal = any(r.position_after >= goal for r in records)
    score = total_reward / max(steps, 1)
    return {
        "total_reward": total_reward,
        "steps": steps,
        "efficiency_score": score,
        "success": reached_goal,
    }


def write_output(
    path: Path,
    trajectory: list[object],
    metrics: dict[str, float | int | bool],
    goal: int,
) -> None:
    payload = {
        "environment": {"name": "LineWorld", "goal": goal},
        "trajectory": [as_dict(step) for step in trajectory],
        "metrics": metrics,
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


if __name__ == "__main__":
    env = LineWorldEnv(goal=5, max_steps=20)
    trajectory = generate_trajectory(env)
    metrics = score_trajectory(trajectory, goal=env.initial_state.goal)

    output_path = Path("minimal_trajectory_output.json")
    write_output(output_path, trajectory, metrics, goal=env.initial_state.goal)

    print(f"Wrote {output_path}")
    print(json.dumps(metrics, indent=2))
