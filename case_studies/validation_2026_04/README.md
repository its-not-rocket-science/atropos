# Replication Package: Atropos Validation Study (April 2026)

This folder contains the exact configuration and reproduction workflow used for the April 2026 validation run.

## Contents

- `configs/validation_2026_04/models.yaml`: model list (Llama-2-7B, Mistral-7B, CodeLlama-13B).
- `configs/validation_2026_04/validation_suite.yaml`: pruning strategies (20%, 40%, 60%), metrics, and commercial assumptions.
- `validation_results/validation_2026_04/*.json`: per-run raw outputs and suite summary.
- `validation_results/validation_2026_04/validation_runs.csv`: flattened raw table.
- `validation_results/validation_2026_04/validation_runs.json`: same raw table as JSON.
- `validation_results/validation_2026_04/hardware_preflight.json`: hardware availability declaration.
- `scripts/generate_validation_visualizations.py`: chart generator.
- `case_studies/validation_2026_04/validation_2026_04.tex`: arXiv-style manuscript source.
- `case_studies/validation_2026_04/references.bib`: bibliography for manuscript citations.

## Reproduction steps

1. Ensure Python dependencies for Atropos are installed.
2. Run prediction + execution suite:

```bash
python scripts/validate_on_models.py --config configs/validation_2026_04/validation_suite.yaml
```

3. Optionally regenerate prediction-only baseline:

```bash
python scripts/validate_on_models.py --config configs/validation_2026_04/validation_suite.yaml --dry-run
```

4. Build publication charts:

```bash
python scripts/generate_validation_visualizations.py \
  --input validation_results/validation_2026_04/validation_runs.csv \
  --output-dir docs/case_studies/assets/validation_2026_04
```

5. Build LaTeX manuscript PDF (optional, for academic packaging):

```bash
cd case_studies/validation_2026_04
pdflatex validation_2026_04.tex
bibtex validation_2026_04
pdflatex validation_2026_04.tex
pdflatex validation_2026_04.tex
```

## Docker environment

Use the repository's existing pruning framework containers:

- `docker/pruning_frameworks/wanda/Dockerfile`
- `docker/pruning_frameworks/sparsegpt/Dockerfile`
- `docker/pruning_frameworks/llm-pruner/Dockerfile`

Example:

```bash
docker build -t atropos-wanda -f docker/pruning_frameworks/wanda/Dockerfile docker/pruning_frameworks/wanda
```

## Notes on this specific run

The execution environment used for this published run had no accessible Hugging Face model artifacts for the three target models and no visible A100/T4 GPUs. As a result, all 9 end-to-end pruning runs failed before metric collection, but all failure artifacts are retained for transparency.
