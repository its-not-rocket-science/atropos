# Atropos roadmap

## Near term

- ✅ add notebook examples
- ✅ add CSV-to-markdown report generation
- ✅ expand comparison and sensitivity CLI output
- ✅ publish documentation pages

## Medium term

- ✅ Monte Carlo uncertainty analysis
- ✅ telemetry import from benchmark runs
- experiment-tracker integration
- region-specific grid carbon presets

## Long term

- lightweight web dashboard
- richer cost models for GPU tiers and batching
- scenario calibration from real serving traces

## Atropos Pipeline (new)

A pipeline extension that automates pruning and tuning when assessments show positive ROI:

### Pipeline stages

1. **Assess** — Run Atropos analysis on deployment scenario
2. **Gate** — Proceed only if projected savings exceed configurable threshold (e.g., break-even < 12 months, annual savings > $10k)
3. **Prune** — Execute structured pruning via framework integration (LLM-Pruner, Wanda, or custom)
4. **Recover** — Run fine-tuning to restore model quality
5. **Validate** — Benchmark optimized model, verify actual metrics match Atropos projections within tolerance
6. **Deploy/Rollback** — Deploy if validation passes; auto-rollback if quality or performance degrades

### Configuration

```yaml
pipeline:
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

### Integration points

- Pruning frameworks (LLM-Pruner, Wanda, SparseGPT)
- Training orchestration (Weights & Biases, MLflow)
- Deployment platforms (vLLM, Triton, custom)
- CI/CD pipelines (GitHub Actions, GitLab CI)
