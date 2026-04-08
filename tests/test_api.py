from __future__ import annotations

from types import SimpleNamespace

import pytest

from atropos.api import RunExperimentConfig, SimpleVariantConfig, run_experiment


def test_run_experiment_requires_two_variants() -> None:
    config = RunExperimentConfig(variants=[SimpleVariantConfig(model_path="model-a")])

    with pytest.raises(ValueError, match="at least two variants"):
        run_experiment(config)


def test_run_experiment_maps_to_existing_architecture(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    def fake_get_platform(name: str, config: dict | None = None) -> object:
        captured["platform_name"] = name
        captured["platform_config"] = config
        return object()

    def fake_run_ab_test(ab_config, platform):
        captured["ab_config"] = ab_config
        captured["platform"] = platform
        return SimpleNamespace(status="completed", experiment_id=ab_config.experiment_id)

    monkeypatch.setattr("atropos.api.get_platform", fake_get_platform)
    monkeypatch.setattr("atropos.api.run_ab_test", fake_run_ab_test)

    result = run_experiment(
        {
            "variants": [
                {"model_path": "control-model", "name": "control"},
                {"model_path": "treatment-model", "name": "treatment"},
            ],
            "primary_metric": "throughput_toks_per_sec",
            "deployment_platform": "vllm",
            "server_config": {"port": 9000},
            "rollout": {"confidence_threshold": 0.95},
            "tokenizer_alignment": {"pad_token": "<eos>"},
        }
    )

    assert result.status == "completed"
    assert captured["platform_name"] == "vllm"
    assert captured["platform_config"] == {"port": 9000}

    ab_config = captured["ab_config"]
    assert ab_config.primary_metric == "throughput_toks_per_sec"
    assert ab_config.auto_stop_conditions["confidence_threshold"] == 0.95
    assert ab_config.auto_stop_conditions["monitoring_interval_seconds"] == 30.0
    assert ab_config.variants[0].deployment_config["tokenizer_alignment"] == {"pad_token": "<eos>"}
