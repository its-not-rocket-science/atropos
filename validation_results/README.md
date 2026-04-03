# Validation Results Artifacts

Each run of `scripts/validate_on_models.py` writes:
- Per-model-per-strategy JSON: `<model_alias>__<strategy>.json`
- Aggregate suite summary: `suite_summary.json`

These artifacts are designed for transparent publication, including failures.
