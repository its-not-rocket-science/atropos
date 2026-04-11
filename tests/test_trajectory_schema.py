from __future__ import annotations

from pathlib import Path

from atropos.rl_env.line_world import LineWorldEnv
from atropos.trajectory import (
    SCHEMA_NAME,
    SCHEMA_VERSION,
    build_schema_spec,
    from_arrow_rows,
    from_json,
    from_line_world_history,
    from_line_world_rollout,
    from_offline_rl_transitions,
    from_rlhf_pairs,
    read_json,
    to_arrow_rows,
    to_json,
    write_json,
)


def test_schema_spec_is_versioned() -> None:
    spec = build_schema_spec()
    assert spec.schema_name == SCHEMA_NAME
    assert spec.schema_version == SCHEMA_VERSION
    assert "trajectory_id" in spec.json_schema["required"]
    assert spec.arrow_columns["reward_total"] == "float64"


def test_json_roundtrip(tmp_path: Path) -> None:
    env = LineWorldEnv(goal=2, max_steps=5)
    env.reset()
    env.step(1)
    env.step(1)
    canonical = from_line_world_history(env.replay())

    payload = to_json(canonical)
    parsed = from_json(payload)
    assert parsed.trajectory_id == canonical.trajectory_id
    assert len(parsed.steps) == 2

    path = tmp_path / "trajectory.json"
    write_json(canonical, path)
    loaded = read_json(path)
    assert loaded.episode_id == canonical.episode_id


def test_arrow_rows_roundtrip() -> None:
    samples = [
        {
            "prompt": "What is 2+2?",
            "response": "4",
            "prompt_tokens": [101, 200],
            "response_tokens": [55],
            "reward": 1.0,
            "preference_score": 0.8,
            "chosen": True,
        }
    ]
    canonical = from_rlhf_pairs(samples)
    adapter = to_arrow_rows(canonical)
    restored = from_arrow_rows(adapter.rows)
    assert restored.steps[0].tokens_in == [101, 200]
    assert restored.steps[0].reward.source == "rlhf"


def test_offline_rl_converter() -> None:
    dataset = [
        {
            "observation": {"state": 0},
            "action": {"delta": 1},
            "next_observation": {"state": 1},
            "reward": 0.5,
            "done": False,
            "reward_components": {"task": 0.5},
            "obs_tokens": [1],
            "action_tokens": [2],
            "dataset": "toy-buffer",
        },
        {
            "observation": {"state": 1},
            "action": {"delta": 1},
            "next_observation": {"state": 2},
            "reward": 1.0,
            "done": True,
            "reward_components": {"task": 1.0},
            "obs_tokens": [3],
            "action_tokens": [4],
        },
    ]
    canonical = from_offline_rl_transitions(dataset)
    assert canonical.steps[0].reward.source == "offline_rl"
    assert canonical.steps[1].done is True


def test_legacy_rollout_payload_converter() -> None:
    env = LineWorldEnv(goal=2, max_steps=10, seed=11)
    env.reset()
    env.step(1)
    env.step(1)
    rollout_path = Path("/tmp/lineworld_rollout.json")
    env.save_rollout(rollout_path)

    import json

    payload = json.loads(rollout_path.read_text(encoding="utf-8"))
    canonical = from_line_world_rollout(payload)
    assert canonical.environment_state is not None
    assert canonical.metadata["source_format"] == "line_world_rollout"
