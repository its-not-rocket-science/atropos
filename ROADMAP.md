# Atropos roadmap

## Near term

- ✅ add notebook examples
- ✅ add CSV-to-markdown report generation
- ✅ expand comparison and sensitivity CLI output
- ✅ publish documentation pages

## Medium term

- ✅ Monte Carlo uncertainty analysis
- ✅ telemetry import from benchmark runs
- ✅ experiment-tracker integration
- ✅ region-specific grid carbon presets

## Long term

- ✅ lightweight web dashboard
- ✅ richer cost models for GPU tiers and batching
- ✅ scenario calibration from real serving traces

## Atropos Pipeline

✅ **Implemented** — A pipeline extension that automates pruning and tuning when assessments show positive ROI:

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

---

## Active Experiments

Practical exercises to validate Atropos projections and demonstrate real-world value.

### 1. Launching Metrics Analysis

**Goal:** Collect real performance telemetry and compare against Atropos projections.

**Tasks:**
- [ ] Set up telemetry collection from vLLM/TGI inference servers
- [ ] Capture memory, throughput, latency, power consumption for baseline models
- [ ] Import telemetry into Atropos scenarios
- [ ] Run calibration to validate projection accuracy
- [ ] Document variance findings and update models if needed

**Deliverables:**
- Calibration report comparing projected vs actual metrics
- Updated scenario presets based on real measurements
- Documentation on telemetry best practices

### 2. LLM Crawl and Analysis

**Goal:** Discover and catalog available models for Atropos testing.

**Tasks:**
- [ ] Run model discovery crawler across HuggingFace Hub
- [ ] Identify models by size tier (edge <1B, medium 1-7B, large >7B)
- [ ] Test model loading on available hardware
- [ ] Generate compatibility matrix
- [ ] Create Atropos scenario files for working models
- [ ] Document recommended test models per use case

**Deliverables:**
- `models-catalog.yaml` with tested, loadable models
- Automated test suite for model validation
- Model recommendation guide by deployment scenario

### 3. Pruning Exercise

**Goal:** Execute actual pruning on real models and validate ROI projections.

**Tasks:**
- [ ] Select 3-5 candidate models (small to medium size)
- [ ] Run Atropos analysis to project savings
- [ ] Execute pruning using integrated frameworks (LLM-Pruner/Wanda/SparseGPT)
- [ ] Measure actual performance of pruned models
- [ ] Compare achieved sparsity vs target
- [ ] Run quality benchmarks (HumanEval, etc.) to validate model quality
- [ ] Document break-even analysis with real data

**Deliverables:**
- Pruned models hosted on HuggingFace
- Before/after performance comparison report
- Quality benchmark results
- Updated Atropos strategies based on real pruning outcomes
- Case study write-up demonstrating ROI validation
