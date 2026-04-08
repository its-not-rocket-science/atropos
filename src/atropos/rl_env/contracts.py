"""Contracts for composable RL environment components."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable


@dataclass
class LineWorldState:
    """Mutable environment state for the LineWorld domain."""

    position: int
    goal: int
    max_steps: int
    step_count: int = 0


@dataclass
class LineWorldParsedAction:
    """Canonical parsed action used by transition/reward layers."""

    delta: int


@dataclass
class LineWorldTransition:
    """Result of applying a parsed action to state."""

    action: int
    position_before: int
    position_after: int
    step_idx: int
    reached_goal: bool
    out_of_steps: bool
    done: bool


@dataclass
class LineWorldReward:
    """Reward output with scalar and named components."""

    value: float
    components: dict[str, float]


@dataclass
class LineWorldStepRecord:
    """Trajectory artifact emitted after each step."""

    step_idx: int
    action: int
    position_before: int
    position_after: int
    reward: float
    done: bool


@runtime_checkable
class Generator(Protocol):
    """Generates environment transition candidates from current state."""

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
    """Parses raw policy output into canonical action semantics."""

    async def parse(self, raw_action: int) -> LineWorldParsedAction:
        """Parse action.

        Failure modes:
        - TypeError when action type is invalid.
        - ValueError when action is outside supported action space.
        """


@runtime_checkable
class RewardFunction(Protocol):
    """Computes reward from pre/post transition context."""

    async def compute(self, transition: LineWorldTransition) -> LineWorldReward:
        """Compute reward scalar and component breakdown.

        Failure modes:
        - RuntimeError when transition metadata is inconsistent.
        """


@runtime_checkable
class TrajectoryBuilder(Protocol):
    """Builds append-only trajectory records for auditing and scoring."""

    async def build_step(
        self,
        transition: LineWorldTransition,
        reward: LineWorldReward,
    ) -> LineWorldStepRecord:
        """Create a trajectory step record.

        Failure modes:
        - RuntimeError for non-monotonic step indices.
        """

    def reset(self) -> None:
        """Reset internal builder state between episodes."""
