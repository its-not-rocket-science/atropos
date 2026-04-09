"""Contracts for composable RL environment components.

This module intentionally defines **strict interfaces** for all core RL components.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol, runtime_checkable

LineWorldAction = Literal[-1, 1]
JSONScalar = str | int | float | bool


@dataclass(slots=True)
class LineWorldState:
    """Mutable environment state for the LineWorld domain.

    Invariants:
    - goal > 0
    - max_steps > 0
    - step_count >= 0
    - step_count <= max_steps
    """

    position: int
    goal: int
    max_steps: int
    step_count: int = 0

    def __post_init__(self) -> None:
        if self.goal <= 0:
            raise ValueError("State invariant violated: goal must be > 0.")
        if self.max_steps <= 0:
            raise ValueError("State invariant violated: max_steps must be > 0.")
        if self.step_count < 0:
            raise ValueError("State invariant violated: step_count must be >= 0.")
        if self.step_count > self.max_steps:
            raise ValueError("State invariant violated: step_count must be <= max_steps.")


@dataclass(frozen=True, slots=True)
class LineWorldParsedAction:
    """Canonical parsed action used by transition/reward layers.

    Invariants:
    - delta is one of {-1, +1}
    """

    delta: LineWorldAction

    def __post_init__(self) -> None:
        if self.delta not in (-1, 1):
            raise ValueError("ParsedAction invariant violated: delta must be -1 or +1.")


@dataclass(frozen=True, slots=True)
class LineWorldTransition:
    """Result of applying a parsed action to state.

    Invariants:
    - step_idx >= 1
    - done == (reached_goal or out_of_steps)
    - position_after - position_before == action
    """

    action: LineWorldAction
    position_before: int
    position_after: int
    step_idx: int
    reached_goal: bool
    out_of_steps: bool
    done: bool

    def __post_init__(self) -> None:
        if self.step_idx < 1:
            raise ValueError("Transition invariant violated: step_idx must be >= 1.")
        if self.done != (self.reached_goal or self.out_of_steps):
            raise ValueError(
                "Transition invariant violated: done must equal reached_goal OR out_of_steps."
            )
        if self.position_after - self.position_before != self.action:
            raise ValueError(
                "Transition invariant violated: position_after - position_before must equal action."
            )


@dataclass(frozen=True, slots=True)
class LineWorldReward:
    """Reward output with scalar and named components.

    Invariants:
    - components has exactly: progress, success_bonus, timeout_penalty
    - value == sum(components.values())
    """

    value: float
    components: dict[str, float]

    def __post_init__(self) -> None:
        required = {"progress", "success_bonus", "timeout_penalty"}
        keys = set(self.components.keys())
        if keys != required:
            raise ValueError(
                "Reward invariant violated: components must contain exactly "
                "progress, success_bonus, timeout_penalty."
            )
        total = sum(self.components.values())
        if abs(self.value - total) > 1e-9:
            raise ValueError("Reward invariant violated: value must equal sum(components.values()).")


@dataclass(frozen=True, slots=True)
class StageIntrospection:
    """Introspection payload for a single RL pipeline stage."""

    stage: str
    metadata: dict[str, JSONScalar]
    intermediate_output: dict[str, JSONScalar]
    reasoning_trace: list[str]


@dataclass(frozen=True, slots=True)
class LineWorldIntrospection:
    """Full step-level introspection spanning generation/parsing/scoring."""

    generation: StageIntrospection
    parsing: StageIntrospection
    scoring: StageIntrospection


@dataclass(frozen=True, slots=True)
class LineWorldStepRecord:
    """Trajectory artifact emitted after each step.

    Invariants:
    - step_idx >= 1
    """

    step_idx: int
    action: LineWorldAction
    position_before: int
    position_after: int
    reward: float
    done: bool
    introspection: LineWorldIntrospection

    def __post_init__(self) -> None:
        if self.step_idx < 1:
            raise ValueError("StepRecord invariant violated: step_idx must be >= 1.")


# Validation layer (runtime contract checks)
def validate_state(state: LineWorldState) -> LineWorldState:
    """Validate and return a state object for chaining.

    Raises:
        TypeError: If object is not LineWorldState.
        ValueError: If any state invariants are violated.
    """

    if not isinstance(state, LineWorldState):
        raise TypeError("Expected LineWorldState.")
    # dataclass invariants already checked during construction.
    return state


def validate_parsed_action(parsed_action: LineWorldParsedAction) -> LineWorldParsedAction:
    """Validate parsed action contract."""

    if not isinstance(parsed_action, LineWorldParsedAction):
        raise TypeError("Expected LineWorldParsedAction.")
    return parsed_action


def validate_transition(transition: LineWorldTransition) -> LineWorldTransition:
    """Validate transition contract."""

    if not isinstance(transition, LineWorldTransition):
        raise TypeError("Expected LineWorldTransition.")
    return transition


def validate_reward(reward: LineWorldReward) -> LineWorldReward:
    """Validate reward contract."""

    if not isinstance(reward, LineWorldReward):
        raise TypeError("Expected LineWorldReward.")
    return reward


def validate_step_record(record: LineWorldStepRecord) -> LineWorldStepRecord:
    """Validate step record contract."""

    if not isinstance(record, LineWorldStepRecord):
        raise TypeError("Expected LineWorldStepRecord.")
    return record


@runtime_checkable
class Generator(Protocol):
    """Generates environment transition candidates from current state.

    Inputs:
    - state: LineWorldState
    - parsed_action: LineWorldParsedAction

    Outputs:
    - LineWorldTransition
    """

    async def generate(
        self,
        state: LineWorldState,
        parsed_action: LineWorldParsedAction,
    ) -> LineWorldTransition:
        """Apply parsed action and produce a transition.

        Failure modes:
        - RuntimeError for invalid state invariants.
        - ValueError for unsupported parsed action values.
        """


@runtime_checkable
class Parser(Protocol):
    """Parses raw policy output into canonical action semantics.

    Inputs:
    - raw_action: int

    Outputs:
    - LineWorldParsedAction
    """

    async def parse(self, raw_action: int) -> LineWorldParsedAction:
        """Parse action.

        Failure modes:
        - TypeError when action type is invalid.
        - ValueError when action is outside supported action space.
        """


@runtime_checkable
class RewardFunction(Protocol):
    """Computes reward from pre/post transition context.

    Inputs:
    - transition: LineWorldTransition

    Outputs:
    - LineWorldReward
    """

    async def compute(self, transition: LineWorldTransition) -> LineWorldReward:
        """Compute reward scalar and component breakdown.

        Failure modes:
        - RuntimeError when transition metadata is inconsistent.
        """


@runtime_checkable
class TrajectoryBuilder(Protocol):
    """Builds append-only trajectory records for auditing and scoring.

    Inputs:
    - transition: LineWorldTransition
    - reward: LineWorldReward
    - introspection: LineWorldIntrospection

    Outputs:
    - LineWorldStepRecord
    """

    async def build_step(
        self,
        transition: LineWorldTransition,
        reward: LineWorldReward,
        introspection: LineWorldIntrospection,
    ) -> LineWorldStepRecord:
        """Create a trajectory step record.

        Failure modes:
        - RuntimeError for non-monotonic step indices.
        """

    def reset(self) -> None:
        """Reset internal builder state between episodes."""
