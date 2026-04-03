# Validation Suite: Multi-Scale ROI Accuracy

## Purpose
This suite validates whether Atropos ROI predictions remain accurate when moving beyond GPT-2 (124M) to practical open models at 1B/7B/13B/34B scale.

## Files
- `scripts/validate_on_models.py`: full validation driver.
- `configs/models.yaml`: model catalog (two open models per size bucket).
- `configs/validation_suite.yaml`: hardware, pruning, and metric settings.
- `docs/validation_report_template.md`: report format for publication and executive review.
- `validation_results/*.json`: per-model strategy outputs and suite summary.

## Hardware Requirements
Recommended baseline machine:
- 1x NVIDIA A100 40GB (minimum for reliable 7B pruning runs).
- 128GB host RAM.
- CUDA 12.x + recent NVIDIA driver.
- Linux with `nvidia-smi` available.

Expected per-model runtime on A100 40GB:
- 1B: ~1 hour / strategy
- 7B: ~4 hours / strategy
- 13B: ~7 hours / strategy
- 34B: ~14 hours / strategy

## Setup
```bash
conda env create -f validation_environment.yaml
conda activate atropos-validation
pip install -e .
```

## Run
Dry-run planning (no model downloads/pruning):
```bash
python scripts/validate_on_models.py --dry-run
```

Run full suite:
```bash
python scripts/validate_on_models.py
```

Run a subset (recommended for first end-to-end check):
```bash
python scripts/validate_on_models.py --models mistral_7b
```

## Reproducibility controls
- Seed pinned in `configs/validation_suite.yaml`.
- Hardware snapshot (GPU model, CUDA runtime, driver) is captured per run.
- All run outputs are JSON artifacts suitable for public sharing.

## Failure handling behavior
If pruning fails for a model/strategy (OOM, framework error, unsupported architecture), the suite:
1. Writes a result JSON with status `pruning_failed` or `error`.
2. Stores error message and continues to next run unless `fail_fast: true`.

## Public sharing guidance
To support academic rigor and reputation transparency:
- Publish all `validation_results/*.json`, including failed runs.
- Report negative results in `docs/validation_report_template.md`.
- Include comparison against naive 20% savings guess from suite summary.
