# Atropos

**Atropos** estimates whether pruning and related optimizations for a coding LLM are worth the engineering effort.

📚 [Documentation](https://its-not-rocket-science.github.io/atropos/)
![PyPI version](https://img.shields.io/pypi/v/atropos-llm) ![Python versions](https://img.shields.io/pypi/pyversions/atropos-llm) [![Documentation](https://img.shields.io/badge/docs-latest-blue)](https://its-not-rocket-science.github.io/atropos/)

Named after the Fate who cuts the thread, Atropos is built for practical deployment questions:

- How much memory, throughput, energy, and cost improvement is realistic?
- When does a pruning project break even?
- How do pruning-only and pruning-plus-quantization compare?
- Which deployment scenarios justify optimization work?

**Atropos Pipeline** automates the full workflow—assess, prune, fine-tune, validate, and deploy—only when the math shows it's worth doing.

## Features

Atropos provides a comprehensive suite of tools for analyzing and automating LLM optimization:

| Feature | Description | Category |
|---------|-------------|----------|
| Python package & CLI | Install via PyPI (`pip install atropos-llm`) with comprehensive command-line interface | Core Analysis |
| Built-in scenario presets | Pre-configured deployment scenarios (edge-coder, medium-coder, large-coder) | Core Analysis |
| Pruning & quantization strategy composition | Combine optimization strategies multiplicatively with `--with-quantization` flag | Core Analysis |
| Validation & error handling | Structured logging, debug modes, comprehensive error context | Core Analysis |
| Comparison commands | Compare multiple strategies side-by-side with sorting by savings, break-even, risk | Advanced Analysis |
| Batch processing | Process multiple scenario files with `batch` command | Advanced Analysis |
| Sensitivity analysis | Analyze impact of parameter variations with `sensitivity` command | Advanced Analysis |
| Monte Carlo uncertainty analysis | Quantify uncertainty in projections with probabilistic modeling | Advanced Analysis |
| CSV-to-markdown report generation | Convert batch results to formatted markdown reports | Reporting |
| Multiple report formats | Generate reports in text, JSON, markdown, or HTML formats | Reporting |
| Web dashboard | Interactive browser-based dashboard for exploration (run with `dashboard` command, requires optional `dashboard` dependencies) | Interactive Tools |
| Telemetry collection | Collect real performance metrics from vLLM/TGI/Triton inference servers | Integration |
| Model testing suite | Test HuggingFace Hub models for compatibility with `test-models` command | Integration |
| Pruning framework integrations | Integrated support for LLM-Pruner, Wanda, and SparseGPT frameworks | Integration |
| Distributed pruning support | Multi-GPU distributed pruning for large models with data/layer/model parallelism | Integration |
| Calibration against real metrics | Validate projections against actual performance telemetry | Integration |
| A/B testing framework | Statistical comparison of model variants with automatic traffic routing, significance testing, and promotion | Integration |
| Atropos Pipeline | Automated optimization workflow: assess, prune, fine-tune, validate, deploy | Automation |
| Development tooling | Comprehensive test suite, CI workflows, pre-commit config, Makefile | Development |

## Installation

### PyPI (stable release)
```bash
pip install atropos-llm
```

### Optional dependencies
The web dashboard requires additional dependencies. Install with:
```bash
pip install atropos-llm[dashboard]
```

For hyperparameter tuning functionality:
```bash
pip install atropos-llm[tuning]
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

Run A/B test experiments:

```bash
# Create and start an experiment from YAML config
atropos-llm ab-test create experiment.yaml --start

# Check experiment status
atropos-llm ab-test status experiment-id

# Analyze results with statistical tests
atropos-llm ab-test analyze experiment-id --format markdown

# Promote winning variant
atropos-llm ab-test promote experiment-id --variant-id variant-a

# List experiments filtered by status
atropos-llm ab-test list --status running
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
