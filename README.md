# Atropos

**Atropos** estimates whether pruning and related optimizations for a coding LLM are worth the engineering effort.

📚 [Documentation](https://its-not-rocket-science.github.io/atropos/)
![PyPI version](https://img.shields.io/pypi/v/atropos-llm) ![Python versions](https://img.shields.io/pypi/pyversions/atropos-llm)

Named after the Fate who cuts the thread, Atropos is built for practical deployment questions:

- How much memory, throughput, energy, and cost improvement is realistic?
- When does a pruning project break even?
- How do pruning-only and pruning-plus-quantization compare?
- Which deployment scenarios justify optimization work?

**Atropos Pipeline** automates the full workflow—assess, prune, fine-tune, validate, and deploy—only when the math shows it's worth doing.

## What's included

- Python package and CLI
- built-in scenario presets
- pruning and quantization strategy composition
- validation and error handling
- comparison, batch, and sensitivity-analysis commands
- Monte Carlo uncertainty analysis
- CSV-to-markdown report generation
- markdown / HTML / JSON reporting
- **web dashboard** for interactive exploration
- **telemetry collection** from vLLM/TGI/Triton inference servers
- **model testing** suite for HuggingFace Hub compatibility
- **pruning framework integrations** (LLM-Pruner, Wanda, SparseGPT)
- **calibration** against real performance metrics
- **Atropos Pipeline** for automated optimization
- tests, CI workflows, pre-commit config, and Makefile

## Installation

### PyPI (stable release)
```bash
pip install atropos-llm
```

### Development installation
```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .[dev]
```

See [CHANGELOG.md](CHANGELOG.md) for version history and release notes.

## Pruning Framework Setup

Atropos includes integrations with pruning frameworks (LLM-Pruner, Wanda, SparseGPT) via the `external/` submodule.

### Wanda Pruning

The `external/wanda` submodule requires specific dependency versions that may conflict with Atropos' main dependencies. For pruning experiments, you can install these dependencies using the setup script:

```bash
python scripts/setup_wanda.py
```

Alternatively, use the Makefile target: `make setup-wanda`.

Or install manually following `external/wanda/INSTALL.md`. Consider using a separate environment.

### Other Frameworks

LLM-Pruner and SparseGPT dependencies are included in the main `pyproject.toml`.

## Quick start

Run a preset:

```bash
atropos-llm preset medium-coder --strategy structured_pruning --report text
```

Add quantization:

```bash
atropos-llm preset medium-coder --strategy structured_pruning --with-quantization --report text
```

Use a YAML scenario:

```bash
atropos-llm scenario examples/medium_coder.yaml --report json
```

Compare strategies:

```bash
atropos-llm compare medium-coder --strategies mild_pruning structured_pruning --format markdown
```

Sort comparison by break-even time:

```bash
atropos-llm compare medium-coder --strategies mild_pruning structured_pruning --sort-by breakeven --ascending
```

Batch process a directory of scenarios:

```bash
atropos-llm batch examples --strategies mild_pruning structured_pruning --output results.csv
```

Hyperparameter tuning for optimal pruning targets:

```bash
atropos-llm tune medium-coder --max-memory 10.0 --min-throughput 30.0
```

Run sensitivity analysis and export to JSON:

```bash
atropos-llm sensitivity medium-coder --strategy structured_pruning --param memory_reduction_fraction --format json --output sensitivity.json
```

Verbose and debug logging:

```bash
atropos-llm --verbose preset medium-coder
atropos-llm --debug tune medium-coder
```

Convert CSV results to markdown report:

```bash
atropos-llm csv-to-markdown results.csv --output report.md
```

Launch the web dashboard:

```bash
atropos-llm dashboard --port 8050
```

Collect telemetry from a running inference server:

```bash
atropos-llm collect-telemetry --server-type vllm --url http://localhost:8000 \
    --duration 60 --output telemetry.json --create-scenario
```

Test HuggingFace models for compatibility:

```bash
atropos-llm test-models --device cuda --max-params 3.0 --catalog models.yaml
```

Validate projections against a real model:

```bash
atropos-llm validate medium-coder --model gpt2 --device cuda
```

Run the automated optimization pipeline:

```bash
atropos-llm pipeline medium-coder --config pipeline.yaml --strategy structured_pruning
```

## Strategy model

Built-in strategies intentionally stay conservative:

- `mild_pruning`
- `structured_pruning`
- `hardware_aware_pruning`

Quantization is modeled as a composable bonus via `--with-quantization` rather than a hardcoded combined preset.

## YAML scenario format

```yaml
name: medium-coder
parameters_b: 34
memory_gb: 14
throughput_toks_per_sec: 40
power_watts: 320
requests_per_day: 50000
tokens_per_request: 1200
electricity_cost_per_kwh: 0.15
annual_hardware_cost_usd: 24000
one_time_project_cost_usd: 27000
```

## Development

```bash
make install
make lint
make typecheck
make test
```

## Notes on the model

Atropos is a planning tool, not a training or pruning framework. It uses transparent formulas to estimate:

- memory footprint changes
- throughput and latency changes
- power and energy per request
- annual cost and CO2e savings
- break-even time for one-time project cost

The default hardware cost model assumes only part of memory reduction translates into hardware savings. That factor is configurable in `AtroposConfig`.

## Atropos Pipeline

The Pipeline automates the full optimization workflow:

1. **Assessment** — Run Atropos analysis on your deployment scenario
2. **Decision gate** — Automatically proceed only if projected savings exceed threshold
3. **Pruning execution** — Trigger structured pruning via integration with pruning frameworks (LLM-Pruner, Wanda, SparseGPT)
4. **Fine-tuning** — Run recovery fine-tuning to restore quality
5. **Validation** — Benchmark the optimized model and verify metrics match projections
6. **Deployment** — Deploy if validation passes; rollback if not

This closes the loop from estimation to automated execution, only applying optimizations when Atropos predicts they are worthwhile.

### Pipeline configuration

```yaml
pipeline:
  name: my-optimization-pipeline
  auto_execute: true
  thresholds:
    max_break_even_months: 12
    min_annual_savings_usd: 10000
    max_quality_risk: medium
  pruning:
    framework: llm-pruner
    target_sparsity: 0.30
  validation:
    tolerance_percent: 10
    quality_benchmark: humaneval
```

Run the pipeline:

```bash
atropos-llm pipeline medium-coder --config pipeline.yaml
```

## Documentation

- [CLI Reference](docs/cli-reference.md) - All commands and options
- [Dashboard Guide](docs/dashboard-guide.md) - Web dashboard usage
- [Telemetry Collection](docs/telemetry-collection-guide.md) - Measuring real inference performance
- [Model Recommendations](docs/model-recommendations.md) - Selecting models for your scenario
- [Model Testing Guide](docs/model-testing-guide.md) - Testing models for compatibility

See `ROADMAP.md` for upcoming features and active experiments.
