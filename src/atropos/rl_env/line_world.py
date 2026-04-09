"""Composable LineWorld environment implementation with async-first contracts."""

from __future__ import annotations

import asyncio
from dataclasses import asdict, dataclass

from .contracts import (
    Generator,
    LineWorldIntrospection,
    LineWorldParsedAction,
    LineWorldReward,
    LineWorldState,
    StageIntrospection,
    LineWorldStepRecord,
    LineWorldTransition,
    Parser,
    RewardFunction,
    TrajectoryBuilder,
)


class LineWorldParser(Parser):
    """Parses raw integer actions into LineWorld action deltas."""

    async def parse(self, raw_action: int) -> LineWorldParsedAction:
        if not isinstance(raw_action, int):
            raise TypeError("Action must be an int.")
        if raw_action not in (-1, 1):
            raise ValueError("Action must be -1 or +1.")
        return LineWorldParsedAction(delta=raw_action)


class LineWorldGenerator(Generator):
    """Applies canonical actions to state and computes terminal signals."""

    async def generate(
        self,
        state: LineWorldState,
        parsed_action: LineWorldParsedAction,
    ) -> LineWorldTransition:
        if state.max_steps <= 0:
            raise RuntimeError("State max_steps must be positive.")

        before = state.position
        state.position += parsed_action.delta
        state.step_count += 1

        reached_goal = state.position >= state.goal
        out_of_steps = state.step_count >= state.max_steps
        done = reached_goal or out_of_steps
        return LineWorldTransition(
            action=parsed_action.delta,
            position_before=before,
            position_after=state.position,
            step_idx=state.step_count,
            reached_goal=reached_goal,
            out_of_steps=out_of_steps,
            done=done,
        )


class LineWorldRewardFunction(RewardFunction):
    """Dense reward with success bonus and timeout penalty."""

    async def compute(self, transition: LineWorldTransition) -> LineWorldReward:
        progress = float(transition.position_after - transition.position_before)
        success_bonus = 10.0 if transition.reached_goal else 0.0
        timeout_penalty = -5.0 if transition.out_of_steps and not transition.reached_goal else 0.0
        total = progress + success_bonus + timeout_penalty
        return LineWorldReward(
            value=total,
            components={
                "progress": progress,
                "success_bonus": success_bonus,
                "timeout_penalty": timeout_penalty,
            },
        )


class LineWorldTrajectoryBuilder(TrajectoryBuilder):
    """Builds trajectory records while validating step monotonicity."""

    def __init__(self) -> None:
        self._last_step_idx = 0

    async def build_step(
        self,
        transition: LineWorldTransition,
        reward: LineWorldReward,
        introspection: LineWorldIntrospection,
    ) -> LineWorldStepRecord:
        if transition.step_idx <= self._last_step_idx:
            raise RuntimeError("Step index must increase monotonically.")
        self._last_step_idx = transition.step_idx
        return LineWorldStepRecord(
            step_idx=transition.step_idx,
            action=transition.action,
            position_before=transition.position_before,
            position_after=transition.position_after,
            reward=reward.value,
            done=transition.done,
            introspection=introspection,
        )

    def reset(self) -> None:
        self._last_step_idx = 0


@dataclass
class LineWorldOrchestrator:
    """Thin environment wrapper coordinating parser/generator/reward/trajectory components."""

    initial_state: LineWorldState
    parser: Parser
    generator: Generator
    reward_function: RewardFunction
    trajectory_builder: TrajectoryBuilder

    def __post_init__(self) -> None:
        self.state = LineWorldState(
            position=self.initial_state.position,
            goal=self.initial_state.goal,
            max_steps=self.initial_state.max_steps,
            step_count=0,
        )

    async def async_reset(self) -> LineWorldState:
        self.state = LineWorldState(
            position=self.initial_state.position,
            goal=self.initial_state.goal,
            max_steps=self.initial_state.max_steps,
            step_count=0,
        )
        self.trajectory_builder.reset()
        return self.state

    async def async_step(self, raw_action: int) -> LineWorldStepRecord:
        pre_state = LineWorldState(
            position=self.state.position,
            goal=self.state.goal,
            max_steps=self.state.max_steps,
            step_count=self.state.step_count,
        )
        parsed = await self.parser.parse(raw_action)
        transition = await self.generator.generate(self.state, parsed)
        reward = await self.reward_function.compute(transition)
        introspection = self._build_introspection(
            raw_action=raw_action,
            pre_state=pre_state,
            parsed=parsed,
            transition=transition,
            reward=reward,
        )
        return await self.trajectory_builder.build_step(transition, reward, introspection)

    def _build_introspection(
        self,
        raw_action: int,
        pre_state: LineWorldState,
        parsed: LineWorldParsedAction,
        transition: LineWorldTransition,
        reward: LineWorldReward,
    ) -> LineWorldIntrospection:
        parsing = StageIntrospection(
            stage="parsing",
            metadata={
                "parser": self.parser.__class__.__name__,
                "action_space": "[-1, +1]",
            },
            intermediate_output={
                "raw_action": raw_action,
                "parsed_delta": parsed.delta,
            },
            reasoning_trace=[
                f"Received raw action {raw_action}.",
                f"Validated action against [-1, +1] and mapped to delta={parsed.delta}.",
            ],
        )
        generation = StageIntrospection(
            stage="generation",
            metadata={
                "generator": self.generator.__class__.__name__,
                "goal": pre_state.goal,
                "max_steps": pre_state.max_steps,
            },
            intermediate_output={
                "position_before": transition.position_before,
                "position_after": transition.position_after,
                "step_idx": transition.step_idx,
                "reached_goal": transition.reached_goal,
                "out_of_steps": transition.out_of_steps,
            },
            reasoning_trace=[
                f"Applied delta={parsed.delta} to position={pre_state.position}.",
                f"New position={transition.position_after}, step_idx={transition.step_idx}.",
                "Computed done flag from reached_goal OR out_of_steps.",
            ],
        )
        scoring = StageIntrospection(
            stage="scoring",
            metadata={
                "reward_model": self.reward_function.__class__.__name__,
                "components": ",".join(sorted(reward.components.keys())),
            },
            intermediate_output={
                "progress": reward.components["progress"],
                "success_bonus": reward.components["success_bonus"],
                "timeout_penalty": reward.components["timeout_penalty"],
                "total_reward": reward.value,
            },
            reasoning_trace=[
                "progress = position_after - position_before.",
                "success_bonus = 10.0 when reached_goal else 0.0.",
                "timeout_penalty = -5.0 when out_of_steps and not reached_goal.",
                "total_reward = progress + success_bonus + timeout_penalty.",
            ],
        )
        return LineWorldIntrospection(
            generation=generation,
            parsing=parsing,
            scoring=scoring,
        )


class LineWorldEnv:
    """Compatibility facade with synchronous API plus async support."""

    def __init__(self, goal: int = 5, max_steps: int = 20) -> None:
        self.initial_state = LineWorldState(position=0, goal=goal, max_steps=max_steps)
        self._orchestrator = LineWorldOrchestrator(
            initial_state=self.initial_state,
            parser=LineWorldParser(),
            generator=LineWorldGenerator(),
            reward_function=LineWorldRewardFunction(),
            trajectory_builder=LineWorldTrajectoryBuilder(),
        )
        self.state = self._orchestrator.state
        self._history: list[LineWorldStepRecord] = []

    def reset(self) -> LineWorldState:
        self.state = asyncio.run(self._orchestrator.async_reset())
        self._history = []
        return self.state

    def step(self, action: int) -> LineWorldStepRecord:
        record = asyncio.run(self._orchestrator.async_step(action))
        self.state = self._orchestrator.state
        self._history.append(record)
        return record

    async def async_reset(self) -> LineWorldState:
        self.state = await self._orchestrator.async_reset()
        self._history = []
        return self.state

    async def async_step(self, action: int) -> LineWorldStepRecord:
        record = await self._orchestrator.async_step(action)
        self.state = self._orchestrator.state
        self._history.append(record)
        return record

    def replay(self) -> list[LineWorldStepRecord]:
        """Return append-only step history for step-by-step debug replay."""

        return list(self._history)

    def explain_reward(self, step_idx: int) -> list[str]:
        """Return scoring trace for a specific step index."""

        for step in self._history:
            if step.step_idx == step_idx:
                return step.introspection.scoring.reasoning_trace
        raise ValueError(f"Step {step_idx} not found in trajectory history.")


def as_dict(record: LineWorldStepRecord) -> dict[str, object]:
    """Stable conversion helper for json serialization."""

    return asdict(record)
