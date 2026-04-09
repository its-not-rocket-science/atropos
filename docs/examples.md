# Examples

Atropos examples focus on ROI estimation + optimization first; pipeline/validation/telemetry/A/B-testing modules are secondary layers for operational workflows.

> Use `atropos` for Python imports and `atropos-llm` for the CLI.

## Basic Analysis

```python
from atropos import estimate_outcome
from atropos.presets import SCENARIOS, STRATEGIES

scenario = SCENARIOS["medium-coder"]
strategy = STRATEGIES["structured_pruning"]

outcome = estimate_outcome(scenario, strategy)
print(f"Savings: ${outcome.annual_total_savings_usd:,.0f}")
```

## Custom Scenario

```python
from atropos import DeploymentScenario, estimate_outcome
from atropos.presets import STRATEGIES

my_scenario = DeploymentScenario(
    name="production-api",
    parameters_b=70.0,
    memory_gb=48.0,
    throughput_toks_per_sec=35.0,
    power_watts=600.0,
    requests_per_day=100000,
    tokens_per_request=1500,
    electricity_cost_per_kwh=0.12,
    annual_hardware_cost_usd=50000.0,
    one_time_project_cost_usd=45000.0,
)

strategy = STRATEGIES["structured_pruning"]
outcome = estimate_outcome(my_scenario, strategy)
```

## Sensitivity Analysis

```python
from atropos.core.calculator import ROICalculator
from atropos.presets import SCENARIOS, STRATEGIES

calc = ROICalculator()
calc.register_scenario(SCENARIOS["medium-coder"])
calc.register_strategy(STRATEGIES["structured_pruning"])

results = calc.sensitivity_analysis(
    "medium-coder",
    "structured_pruning",
    "memory_reduction_fraction",
    variations=5,
    step=0.1,
)

for factor, outcome in results:
    print(f"Factor {factor:.2f}: ${outcome.annual_total_savings_usd:,.0f}")
```

## Comparing Strategies

```python
from atropos import estimate_outcome
from atropos.presets import SCENARIOS, STRATEGIES

scenario = SCENARIOS["medium-coder"]

for name, strategy in STRATEGIES.items():
    outcome = estimate_outcome(scenario, strategy)
    be_months = outcome.break_even_years * 12 if outcome.break_even_years else None
    print(f"{name}: ${outcome.annual_total_savings_usd:,.0f}/year, break-even: {be_months:.0f}mo")
```


## CLI Companion Example

```bash
atropos-llm scenario my_scenario.yaml --strategy structured_pruning --report markdown
```

This mirrors the `scenario` argparse interface in `src/atropos/cli.py` and is useful when you want the same ROI outputs from the command line.
