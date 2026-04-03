# Atropos Multi-Scale Validation Report

## Executive Summary (Commercial)
- **Date:** {{DATE}}
- **Hardware:** {{GPU_TYPE}} ({{GPU_COUNT}}x), CUDA {{CUDA_VERSION}}, Driver {{DRIVER_VERSION}}
- **Models validated:** {{MODEL_COUNT}}
- **Strategies:** {{STRATEGIES}}
- **Bottom-line accuracy:**
  - MAPE (memory): **{{MAPE_MEMORY}}%**
  - MAPE (throughput): **{{MAPE_THROUGHPUT}}%**
  - MAPE (cost savings): **{{MAPE_COST}}%**
  - MAPE (break-even months): **{{MAPE_BREAKEVEN}}%**
- **Atropos vs naive baseline (20% savings guess):** {{ATROPOS_VS_NAIVE}}

## Methodology (Academic)
### Study Design
- Model buckets: 1B, 7B, 13B, 34B (2 models each, cross-family).
- Pruning frameworks: Wanda and SparseGPT via native Atropos integrations.
- Hardware control: fixed A100 40GB environment and pinned software environment.
- Seeds: global seed {{SEED}} for model loading and stochastic components.

### Data and Metrics
- **System metrics:** memory (GB), throughput (tok/s), power (W).
- **Quality metrics:** Wikitext-2 perplexity, coding proxy / HumanEval pass@1.
- **Prediction validation metrics:** MAPE, Pearson correlation, CI coverage (90%).
- **Break-even validation:** predicted vs actual break-even in months.

### Statistical Testing
- Null hypothesis examples:
  - H0: mean relative error is equal across model buckets.
  - H0: Atropos does not outperform naive 20% savings guess.
- Report effect sizes and p-values; include non-significant and negative findings.

## Results

### Aggregate Accuracy
| Metric | Value |
|---|---:|
| MAPE Memory (%) | {{MAPE_MEMORY}} |
| MAPE Throughput (%) | {{MAPE_THROUGHPUT}} |
| MAPE Cost Savings (%) | {{MAPE_COST}} |
| MAPE Break-even (months, %) | {{MAPE_BREAKEVEN}} |
| Corr(Pred, Actual) Memory | {{CORR_MEMORY}} |
| Corr(Pred, Actual) Throughput | {{CORR_THROUGHPUT}} |
| Corr(Pred, Actual) Savings | {{CORR_SAVINGS}} |
| 90% CI Coverage (Memory) | {{COV_MEMORY}} |
| 90% CI Coverage (Throughput) | {{COV_THROUGHPUT}} |
| 90% CI Coverage (Savings) | {{COV_SAVINGS}} |

### Per-Model Results
| Model | Family | Size | Strategy | Pred Savings ($/yr) | Actual Savings ($/yr) | Error % | Notes |
|---|---|---|---|---:|---:|---:|---|
| {{MODEL_1}} | {{FAMILY_1}} | {{SIZE_1}} | {{STRAT_1}} | {{PRED_1}} | {{ACTUAL_1}} | {{ERROR_1}} | {{NOTE_1}} |

### Failure Analysis (Academic + Reputation)
- Document all failures, including pruning crashes, OOM events, and unstable metrics.
- Include representative logs and root-cause hypothesis.
- Explicitly list models/strategies skipped and why.

## Limitations
- Some quality metrics may use proxy scoring if full benchmark harness unavailable.
- Hardware variability (temperature, background jobs) can affect throughput/power.
- 34B-class runs may require tensor/model parallel and longer stabilization windows.

## Negative Results and Learnings (Reputation)
- **What did not work:** {{NEGATIVE_RESULTS}}
- **What changed after failures:** {{ITERATIONS}}
- **Open questions for community validation:** {{OPEN_QUESTIONS}}

## Reproducibility Appendix
- Conda env file: `validation_environment.yaml`
- Suite config: `configs/validation_suite.yaml`
- Model catalog: `configs/models.yaml`
- Script: `scripts/validate_on_models.py`
- Raw outputs: `validation_results/*.json`
- Expected runtime (A100 40GB):
  - 1B model: ~1 hour/model/strategy
  - 7B model: ~4 hours/model/strategy
  - 13B model: ~7 hours/model/strategy
  - 34B model: ~14 hours/model/strategy
