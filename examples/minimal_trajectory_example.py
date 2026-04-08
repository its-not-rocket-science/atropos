"""Minimal, complete trajectory pipeline example.

This file intentionally keeps everything in one place so the data flow is easy to learn:
1) define environment
2) generate trajectory
3) score trajectory
4) write output
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path


# WHY: A tiny state type makes transitions explicit and testable.
@dataclass
class State:
    position: int
    goal: int
    max_steps: int
    step_count: int = 0


# WHY: We record each transition so we can debug and audit how scores were produced.
@dataclass
class StepRecord:
    step_idx: int
    action: int
    position_before: int
    position_after: int
    reward: float
    done: bool


# WHY: Environment owns transition rules; separating this from policy avoids hidden coupling.
class LineWorldEnv:
    """One-dimensional environment.

    Agent starts at position 0 and tries to reach `goal`.
    Allowed actions: -1 (left) or +1 (right).
    """

    def __init__(self, goal: int = 5, max_steps: int = 20) -> None:
        self.initial_state = State(position=0, goal=goal, max_steps=max_steps)
        self.state = self.initial_state

    def reset(self) -> State:
        # WHY: reset returns a fresh run state, enabling deterministic repeatable experiments.
        self.state = State(
            position=self.initial_state.position,
            goal=self.initial_state.goal,
            max_steps=self.initial_state.max_steps,
            step_count=0,
        )
        return self.state

    def step(self, action: int) -> StepRecord:
        if action not in (-1, 1):
            raise ValueError("Action must be -1 or +1.")

        before = self.state.position
        self.state.position += action
        self.state.step_count += 1

        reached_goal = self.state.position >= self.state.goal
        out_of_steps = self.state.step_count >= self.state.max_steps
        done = reached_goal or out_of_steps

        # WHY: Dense reward (+ progress each step) teaches smoother behavior
        progress = self.state.position - before
        reward = progress
        if reached_goal:
            reward += 10.0  # success bonus
        if out_of_steps and not reached_goal:
            reward -= 5.0  # failure penalty

        return StepRecord(
            step_idx=self.state.step_count,
            action=action,
            position_before=before,
            position_after=self.state.position,
            reward=reward,
            done=done,
        )


# WHY: Policy is intentionally simple and explicit for teaching; no hidden ML model logic.
def greedy_policy(state: State) -> int:
    return 1 if state.position < state.goal else -1


# WHY: Generating a full trajectory object makes downstream scoring and storage straightforward.
def generate_trajectory(env: LineWorldEnv) -> list[StepRecord]:
    records: list[StepRecord] = []
    state = env.reset()

    while True:
        action = greedy_policy(state)
        record = env.step(action)
        records.append(record)
        if record.done:
            break

    return records


# WHY: Keep scoring a pure function of trajectory data, so it's easy to validate and reuse.
def score_trajectory(records: list[StepRecord], goal: int) -> dict[str, float | int | bool]:
    total_reward = sum(r.reward for r in records)
    steps = len(records)
    reached_goal = any(r.position_after >= goal for r in records)

    # A simple efficiency metric to demonstrate multiple score outputs.
    score = total_reward / max(steps, 1)

    return {
        "total_reward": total_reward,
        "steps": steps,
        "efficiency_score": score,
        "success": reached_goal,
    }


# WHY: JSON output is language-agnostic and easy to inspect in notebooks, scripts, or dashboards.
def write_output(
    path: Path,
    trajectory: list[StepRecord],
    metrics: dict[str, float | int | bool],
    goal: int,
) -> None:
    payload = {
        "environment": {"name": "LineWorld", "goal": goal},
        "trajectory": [asdict(step) for step in trajectory],
        "metrics": metrics,
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


if __name__ == "__main__":
    env = LineWorldEnv(goal=5, max_steps=20)  # 1) environment definition
    trajectory = generate_trajectory(env)  # 2) trajectory generation
    metrics = score_trajectory(trajectory, goal=env.initial_state.goal)  # 3) scoring

    output_path = Path("minimal_trajectory_output.json")
    write_output(output_path, trajectory, metrics, goal=env.initial_state.goal)  # 4) data output

    print(f"Wrote {output_path}")
    print(json.dumps(metrics, indent=2))
