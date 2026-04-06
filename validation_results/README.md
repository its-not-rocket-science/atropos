# Validation Results Artifacts

Each run of `scripts/validate_on_models.py` writes:
- Per-model-per-strategy JSON: `<model_alias>__<strategy>.json`
- Aggregate suite summary: `suite_summary.json`

## April 2026 case study artifacts

The April 2026 publication package is stored under:

- `validation_results/validation_2026_04/`
  - Per-run JSON artifacts for 9 model/strategy combinations
  - `suite_summary.json`
  - `validation_runs.csv` (flattened table for analysis)
  - `validation_runs.json` (JSON export of the same table)
  - `hardware_preflight.json`

These artifacts are designed for transparent publication, including failures.
