# Documentation & Packaging Consistency Audit

This file is the single source of truth for packaging metadata, runtime config defaults, and CLI entry points derived from code.

## Inconsistency report

### 1) Missing target docs

| Target | Status | Finding |
|---|---|---|
| `CONFIG.md` | Missing (created) | No central source-of-truth file previously existed. |
| `example_trainer/README.md` | Missing (created) | `example_trainer` docs target did not exist in repository tree. |

### 2) README mismatches with actual CLI

| Location | Previous doc text | Actual code | Impact |
|---|---|---|---|
| `README.md` A/B test create step | `atropos-llm ab-test create experiment.yaml --start` | `ab-test create --config <path>` and no `--start` flag | Command fails. |
| `README.md` install section | Only `pip install -e .` was shown | Extras exist (`dev`, `tuning`, `dashboard`) | Missing install options for common workflows. |

### 3) Packaging/docs consistency findings

| Area | Finding | Source of truth |
|---|---|---|
| Distribution vs import name | Distribution/CLI name is `atropos-llm`; Python module is `atropos` | `pyproject.toml`, `src/atropos/__init__.py` |
| CLI script entry point | Only one script entry point: `atropos-llm = atropos.cli:main` | `pyproject.toml` |
| Versioning | Version is `0.5.0` in both packaging and module metadata | `pyproject.toml`, `src/atropos/__init__.py` |

## Single source-of-truth tables

### Package extras

| Extra | Packages |
|---|---|
| `dev` | `pytest`, `pytest-cov`, `ruff`, `mypy`, `build`, `twine`, `types-PyYAML` |
| `tuning` | `scikit-optimize` |
| `dashboard` | `dash`, `plotly`, `pandas` |

### Config defaults (`AtroposConfig`)

| Field | Default | Env override |
|---|---:|---|
| `grid_co2e_factor` | `0.35` | `ATROPOS_GRID_CO2E` |
| `hardware_savings_correlation` | `0.8` | `ATROPOS_HW_SAVINGS_CORR` |
| `default_report_format` | `text` | `ATROPOS_REPORT_FORMAT` |
| `risk_rank` | `{low:1, medium:2, high:3}` | none |

### CLI entry points

#### Packaging script entry point

| Command | Callable |
|---|---|
| `atropos-llm` | `atropos.cli:main` |

#### Top-level CLI commands

`list-presets`, `preset`, `scenario`, `compare`, `batch`, `sensitivity`, `monte-carlo`, `tune`,
`csv-to-markdown`, `import-telemetry`, `collect-telemetry`, `import-experiment`, `dashboard`,
`list-carbon-presets`, `cloud-pricing`, `calibrate`, `pipeline`, `validate-pipeline-config`, `validate`,
`detect-anomalies`, `benchmark-multi-gpu`, `test-models`, `visualize`, `ab-test`, `setup-pruning`, `test-pruning`.

### Versioning

| Artifact | Version |
|---|---|
| `pyproject.toml` (`[project].version`) | `0.5.0` |
| `src/atropos/__init__.py` (`__version__`) | `0.5.0` |
| `CHANGELOG.md` latest tagged release | `0.5.0` on `2026-03-27` |

## Exact patch suggestions applied

1. **README CLI fix**
   - Replaced invalid command:
     - from: `atropos-llm ab-test create experiment.yaml --start`
     - to: `atropos-llm ab-test create --config experiment.yaml`
2. **README install guidance update**
   - Added optional extras install commands for `dev`, `tuning`, `dashboard`.
3. **README naming clarity**
   - Added explicit note: import module is `atropos`, CLI/distribution name is `atropos-llm`.
4. **Created this file (`CONFIG.md`)**
   - Added source-of-truth tables and audit findings.
5. **Created `example_trainer/README.md`**
   - Added compatibility note pointing to `examples/README.md` as the maintained examples doc.
