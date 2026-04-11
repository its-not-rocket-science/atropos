from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from atropos.rl_env.contracts import LineWorldReward, LineWorldState, LineWorldTransition
from atropos.rl_env.line_world import LineWorldEnv


def run_sync_episode(goal: int, max_steps: int) -> tuple[float, int, bool]:
    env = LineWorldEnv(goal=goal, max_steps=max_steps)
    state = env.reset()
    total_reward = 0.0
    steps = 0
    success = False

    while True:
        action = 1 if state.position < state.goal else -1
        record = env.step(action)
        total_reward += record.reward
        steps += 1
        success = success or record.position_after >= goal
        state = env.state
        if record.done:
            break

    return total_reward, steps, success


def test_async_sync_parity() -> None:
    sync_total, sync_steps, sync_success = run_sync_episode(goal=5, max_steps=20)

    env = LineWorldEnv(goal=5, max_steps=20)
    state = asyncio.run(env.async_reset())
    total_reward = 0.0
    steps = 0
    success = False

    while True:
        action = 1 if state.position < state.goal else -1
        record = asyncio.run(env.async_step(action))
        total_reward += record.reward
        steps += 1
        success = success or record.position_after >= 5
        state = env.state
        if record.done:
            break

    assert (total_reward, steps, success) == (sync_total, sync_steps, sync_success)


def test_invalid_action_is_rejected() -> None:
    env = LineWorldEnv(goal=5, max_steps=20)
    env.reset()
    with pytest.raises(ValueError, match=r"Action must be -1 or \+1"):
        env.step(0)


def test_legacy_behavior_preserved() -> None:
    total_reward, steps, success = run_sync_episode(goal=5, max_steps=20)
    assert total_reward == 15.0
    assert steps == 5
    assert success is True


def test_step_contains_introspection_for_all_stages() -> None:
    env = LineWorldEnv(goal=3, max_steps=10)
    env.reset()
    record = env.step(1)

    assert record.introspection.parsing.stage == "parsing"
    assert record.introspection.generation.stage == "generation"
    assert record.introspection.scoring.stage == "scoring"
    assert "raw_action" in record.introspection.parsing.intermediate_output
    assert "position_after" in record.introspection.generation.intermediate_output
    assert "total_reward" in record.introspection.scoring.intermediate_output
    assert len(record.introspection.scoring.reasoning_trace) >= 1


def test_replay_and_reward_explanation() -> None:
    env = LineWorldEnv(goal=2, max_steps=10)
    env.reset()
    env.step(1)
    env.step(1)

    replay = env.replay()
    assert len(replay) == 2
    assert replay[0].step_idx == 1
    assert replay[1].step_idx == 2
    assert any("total_reward" in line for line in env.explain_reward(step_idx=2))


def test_state_invariants_are_enforced() -> None:
    with pytest.raises(ValueError, match="goal must be > 0"):
        LineWorldState(position=0, goal=0, max_steps=10)


def test_reward_invariants_are_enforced() -> None:
    with pytest.raises(ValueError, match="components must contain exactly"):
        LineWorldReward(value=1.0, components={"progress": 1.0})


def test_transition_done_invariant_is_enforced() -> None:
    with pytest.raises(ValueError, match="done must equal reached_goal OR out_of_steps"):
        LineWorldTransition(
            action=1,
            position_before=0,
            position_after=1,
            step_idx=1,
            reached_goal=False,
            out_of_steps=False,
            done=True,
        )


def test_bool_action_is_rejected() -> None:
    env = LineWorldEnv(goal=2, max_steps=5)
    env.reset()
    with pytest.raises(TypeError, match="bool is not allowed"):
        env.step(True)


def test_save_rollout_and_replay_exact(tmp_path: Path) -> None:
    env = LineWorldEnv(goal=3, max_steps=10, seed=7)
    env.reset()
    env.step(1)
    env.step(1)
    env.step(1)
    path = tmp_path / "rollout.json"
    env.save_rollout(path)

    replayed = LineWorldEnv.replay_from_rollout(path)
    assert len(replayed) == 3
    assert replayed[-1].done is True
