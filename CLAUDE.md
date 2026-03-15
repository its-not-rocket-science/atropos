# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Atropos is a Python CLI tool for estimating ROI (performance, energy, cost, break-even) of pruning and quantization optimizations for coding LLM deployments. It uses transparent formulas to estimate memory, throughput, power, and cost changes.

## Non-negotiable rules
- Do not commit code unless asked to, but suggest a commit message when a task is considered done
- Keep the README and ROADMAP up-to-date
- make sure tests and linting pass before considering any task done
- make sure ruff check passes (no missing imports, all imports top level, no unused variables and arguments etc)
- make sure mypy src passes (no lines too long etc)
- Do not modify the type ignore comment on line 133 of src/atropos/validation/runner.py
- Run ruff format --check before marking code as done

## Development Commands

Install dependencies:
```bash
make install
# or: pip install -e .[dev]
```

Run tests:
```bash
make test
# or: pytest tests/ -v --cov=atropos --cov-report=term-missing
```

Run a single test:
```bash
pytest tests/test_calculations.py::test_estimate_outcome_reduces_energy_for_positive_optimization -v
```

Lint and format:
```bash
make lint        # ruff check + format check
make format      # ruff check --fix + ruff format
```

Type check:
```bash
make typecheck   # mypy src
```

Build:
```bash
make build       # python -m build
```

Run CLI locally:
```bash
atropos preset medium-coder --strategy structured_pruning --report text
atropos scenario examples/medium_coder.yaml --report json
atropos compare medium-coder --strategies mild_pruning structured_pruning --format markdown
```

## Architecture

### Core Models (`src/atropos/models.py`)

Three frozen dataclasses form the domain model:
- `DeploymentScenario`: Hardware/deployment parameters (memory, power, request volume, costs)
- `OptimizationStrategy`: Pruning/quantization effects (reduction fractions, throughput improvement, quality risk)
- `OptimizationOutcome`: Calculated results (baseline vs optimized metrics, savings, break-even)

### Calculation Engine (`src/atropos/calculations.py`)

- `estimate_outcome()`: Main function applying optimization formulas to compute energy, cost, CO2e savings
- `combine_strategies()`: Composes strategies multiplicatively (e.g., pruning + quantization)
- Validation functions enforce reasonable ranges (fractions in [0,1), positive values)

### CLI (`src/atropos/cli.py`)

Commands: `preset`, `scenario`, `compare`, `batch`, `sensitivity`, `list-presets`
- Uses argparse with subparsers
- `--with-quantization` flag composes strategy with `QUANTIZATION_BONUS`
- Report formats: `text`, `json`, `markdown`, `html`

### Strategy Pattern

Built-in strategies in `src/atropos/presets.py`:
- `mild_pruning`, `structured_pruning`, `hardware_aware_pruning`
- `QUANTIZATION_BONUS`: Composable add-on for quantization effects

Strategies are intentionally conservative estimates. Combine via `combine_strategies()` for multiplicative effects on memory, throughput, and power.

### Configuration (`src/atropos/config.py`)

`AtroposConfig` holds global defaults:
- `grid_co2e_factor`: kg CO2e per kWh (default 0.35)
- `hardware_savings_correlation`: How much memory reduction translates to hardware savings (default 0.8)
- Loadable from env vars (`ATROPOS_*`) or YAML file

### I/O and Reporting (`src/atropos/io.py`, `src/atropos/reporting.py`)

- `load_scenario()`: Loads YAML scenario files with required key validation
- `render_report()`: Dispatches to format-specific reporters
- `export_to_csv()`: Batch results export

### Testing

- pytest with coverage (configured in pyproject.toml)
- Tests in `tests/` directory
- conftest.py adds `src/` to path for imports
