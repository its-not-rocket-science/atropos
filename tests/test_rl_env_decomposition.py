from __future__ import annotations

import asyncio

import pytest

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
