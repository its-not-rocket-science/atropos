# Python API

Atropos exposes an ROI-estimation-and-optimization-first Python API for scenario/strategy modeling. Secondary modules provide pipeline, validation, telemetry, and A/B testing workflows around that core.

> Use `atropos` for Python imports and `atropos-llm` for the CLI.

For CLI workflows, use `atropos-llm --help` and the subcommands defined in `src/atropos/cli.py`.

## Core Models

### `DeploymentScenario`

Represents a deployment configuration.

```python
from atropos import DeploymentScenario

scenario = DeploymentScenario(
    name="my-deployment",
    parameters_b=34.0,               # Model size in billions
    memory_gb=14.0,                  # Memory usage
    throughput_toks_per_sec=40.0,    # Current throughput
    power_watts=320.0,               # Power draw
    requests_per_day=50000,          # Daily requests
    tokens_per_request=1200,         # Tokens per request
    electricity_cost_per_kwh=0.15,   # Electricity cost
    annual_hardware_cost_usd=24000.0,  # Optional legacy hardware cost field
    one_time_project_cost_usd=27000.0,
)
```

### `OptimizationStrategy`

Defines optimization parameters.

```python
from atropos import OptimizationStrategy

strategy = OptimizationStrategy(
    name="my-strategy",
    parameter_reduction_fraction=0.30,
    memory_reduction_fraction=0.22,
    throughput_improvement_fraction=0.20,
    power_reduction_fraction=0.14,
    quality_risk="medium",
)
```

## Calculations

### `estimate_outcome`

Calculate the outcome of applying a strategy to a scenario.

```python
from atropos import estimate_outcome

outcome = estimate_outcome(scenario, strategy)

print(f"Annual savings: ${outcome.annual_total_savings_usd}")
print(f"Break-even: {outcome.break_even_years} years")
```

### `combine_strategies`

Combine two strategies (for example, pruning + quantization bonus).

```python
from atropos import combine_strategies, estimate_outcome
from atropos.presets import QUANTIZATION_BONUS

combined = combine_strategies(strategy, QUANTIZATION_BONUS)
outcome = estimate_outcome(scenario, combined)
```

## Batch Processing

```python
from atropos.batch import batch_process

results = batch_process(
    "scenarios/",
    ["mild_pruning", "structured_pruning"],
    output_file="results.csv",
)
```
