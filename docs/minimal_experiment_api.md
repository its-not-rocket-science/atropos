# Minimal Experiment API (`run_experiment`)

This API layer provides a single high-level entrypoint for running an experiment in ~10 lines while mapping directly onto the existing architecture:

- `ABTestConfig` for experiment semantics
- `DeploymentPlatform` (`get_platform`) for server setup
- `ExperimentRunner` (`run_ab_test`) for lifecycle + rollout handling

## API spec

### Function

```python
run_experiment(config: RunExperimentConfig | Mapping[str, Any]) -> ExperimentResult
```

### Minimal config schema

```python
RunExperimentConfig(
    variants: list[SimpleVariantConfig],  # required, at least 2
    primary_metric: str = "latency_ms_per_request",
    deployment_platform: str = "vllm",
    experiment_id: str | None = None,
    name: str | None = None,
)
```

### Defaults (sane)

- `secondary_metrics=["throughput_toks_per_sec", "error_rate"]`
- `traffic_allocation=1.0`
- `significance_level=0.05`
- `statistical_power=0.8`
- `test_type="t-test"`
- `min_sample_size_per_variant=100`
- `max_duration_hours=24.0`
- rollout polling via `monitoring_interval_seconds=30.0`

### Hidden internals handled by the API

1. **Server setup**
   - `get_platform(deployment_platform, config=server_config)`
2. **Rollout handling**
   - maps `rollout` dict to `ABTestConfig.auto_stop_conditions`
3. **Tokenizer alignment**
   - attaches `tokenizer_alignment` policy to each variant deployment config

### No feature loss guarantee

Advanced behavior is still available via passthrough fields on `RunExperimentConfig`:

- `server_config`
- `rollout`
- `tokenizer_alignment`
- `health_checks`
- `metadata`

These map cleanly to existing lower-level `ABTestConfig` and platform interfaces.

## One-file usage (10 lines)

```python
from atropos.api import run_experiment

result = run_experiment({
    "variants": [
        {"model_path": "meta-llama/Llama-3.1-8B", "name": "control"},
        {"model_path": "meta-llama/Llama-3.1-8B-Instruct", "name": "treatment"},
    ],
    "primary_metric": "latency_ms_per_request",
    "deployment_platform": "vllm",
})
print(result.status, result.experiment_id)
```
