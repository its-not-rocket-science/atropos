from atroposlib.envs.base import BaseEnv


def test_step_delegates_to_runtime_transport_logs_and_checkpoint() -> None:
    env = BaseEnv()
    result = env.step({"task": "x"}, worker_count=2)

    assert result["ok"] is True
    assert result["payload"]["worker_count"] == 2
    assert len(env.logger.events) == 2
    assert env.checkpoint_manager.snapshots[-1] == result


def test_merge_yaml_and_cli_cli_takes_precedence() -> None:
    env = BaseEnv()
    merged = env.merge_yaml_and_cli(
        {"model": "baseline", "workers": 1},
        {"workers": 4},
    )
    assert merged == {"model": "baseline", "workers": 4}


def test_build_cli_args_is_deterministic() -> None:
    env = BaseEnv()
    args = env.build_cli_args({"workers": 2, "model_name": "gpt"})
    assert args == ["--model-name", "gpt", "--workers", "2"]
