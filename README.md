# Atropos

**Atropos** estimates whether pruning and related optimizations for a coding LLM are worth the engineering effort.

Named after the Fate who cuts the thread, Atropos is built for practical deployment questions:

- How much memory, throughput, energy, and cost improvement is realistic?
- When does a pruning project break even?
- How do pruning-only and pruning-plus-quantization compare?
- Which deployment scenarios justify optimization work?

Future: Atropos Pipeline will automate the full workflow—assess, prune, fine-tune, validate, and deploy—only when the math shows it's worth doing.

## What's included

- Python package and CLI
- built-in scenario presets
- pruning and quantization strategy composition
- validation and error handling
- comparison, batch, and sensitivity-analysis commands
- CSV-to-markdown report generation
- markdown / HTML / JSON reporting
- tests, CI workflows, pre-commit config, and Makefile

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .[dev]
```

## Quick start

Run a preset:

```bash
atropos preset medium-coder --strategy structured_pruning --report text
```

Add quantization:

```bash
atropos preset medium-coder --strategy structured_pruning --with-quantization --report text
```

Use a YAML scenario:

```bash
atropos scenario examples/medium_coder.yaml --report json
```

Compare strategies:

```bash
atropos compare medium-coder --strategies mild_pruning structured_pruning --format markdown
```

Sort comparison by break-even time:

```bash
atropos compare medium-coder --strategies mild_pruning structured_pruning --sort-by breakeven --ascending
```

Batch process a directory of scenarios:

```bash
atropos batch examples --strategies mild_pruning structured_pruning --output results.csv
```

Run sensitivity analysis and export to JSON:

```bash
atropos sensitivity medium-coder --strategy structured_pruning --param memory_reduction_fraction --format json --output sensitivity.json
```

Convert CSV results to markdown report:

```bash
atropos csv-to-markdown results.csv --output report.md
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

## Pipeline extension (planned)

Atropos Pipeline will automate the optimization workflow:

1. **Assessment** — Run Atropos analysis on your deployment scenario
2. **Decision gate** — Automatically proceed only if projected savings exceed threshold
3. **Pruning execution** — Trigger structured pruning via integration with pruning frameworks
4. **Fine-tuning** — Run recovery fine-tuning to restore quality
5. **Validation** — Benchmark the optimized model and verify metrics match projections
6. **Deployment** — Deploy if validation passes; rollback if not

This closes the loop from estimation to automated execution, only applying optimizations when Atropos predicts they are worthwhile.

## Roadmap highlights

- Monte Carlo uncertainty modeling
- experiment-tracking integrations
- notebook examples and dashboarding
- region-specific carbon intensity presets
- **Atropos Pipeline for automated pruning and tuning**

See `ROADMAP.md` for details.
