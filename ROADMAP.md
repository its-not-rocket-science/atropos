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

### 1. Launching Metrics Analysis ✅

**Goal:** Collect real performance telemetry and compare against Atropos projections.

**Tasks:**
- [x] Set up telemetry collection from vLLM/TGI inference servers
- [x] Capture memory, throughput, latency, power consumption for baseline models
- [x] Import telemetry into Atropos scenarios
- [x] Run calibration to validate projection accuracy
- [x] Document variance findings and update models if needed

**Deliverables:**
- ✅ `src/atropos/telemetry_collector.py` — Active collectors for vLLM, TGI, Triton
- ✅ `atropos collect-telemetry` CLI command
- ✅ `docs/telemetry-collection-guide.md` — Complete usage documentation
- ✅ Calibration integration with existing `atropos calibrate` command

### 2. LLM Crawl and Analysis ✅

**Goal:** Discover and catalog available models for Atropos testing.

**Tasks:**
- [x] Run model discovery crawler across HuggingFace Hub
- [x] Identify models by size tier (edge <1B, medium 1-7B, large >7B)
- [x] Test model loading on available hardware
- [x] Generate compatibility matrix
- [x] Create Atropos scenario files for working models
- [x] Document recommended test models per use case

**Deliverables:**
- ✅ `src/atropos/model_tester.py` — Automated test suite
- ✅ `atropos test-models` CLI command
- ✅ `docs/model-recommendations.md` — Comprehensive model guide
- ✅ `scripts/model-discovery-crawler.py` — Discovery tool
- ✅ Curated model lists by size tier and use case

### 3. Pruning Exercise

**Goal:** Execute actual pruning on real models and validate ROI projections.

**Tasks:**
- [x] Select 3-5 candidate models (small to medium size)
- [x] Download and cache candidate models
- [x] Run Atropos analysis to project savings
- [x] Execute pruning using integrated frameworks (LLM-Pruner/Wanda/SparseGPT)
- [x] Measure actual performance of pruned models
- [x] Compare achieved sparsity vs target
- [x] Run quality benchmarks to validate model quality
- [ ] Document break-even analysis with real data

**Deliverables:**
- ✅ `scripts/download_test_models.py` — Download script for 5 candidate models
- ✅ `scripts/project_savings.py` — Generate baseline projections
- ✅ `scripts/prune_models.py` — PyTorch-based pruning implementation
- ✅ `scripts/compare_projections.py` — Compare projected vs actual results
- ✅ `scripts/benchmark_quality.py` — Quality benchmarking script
- ✅ `scripts/test_pruning_candidates.py` — Validation test runner
- ✅ `test_data/` — Local cache of 36+ GB models
- ✅ `test_data/projections.json/md` — Baseline projections
- ✅ `test_data/pruned_models/` — 8 pruned model variants
- ✅ `test_data/pruning_report.json/md` — Pruning operation results
- ✅ `test_data/comparison_report.json/md` — Projected vs actual comparison
- ✅ `test_data/benchmark_report.json/md` — Quality benchmark results
- [ ] Pruned models hosted on HuggingFace
- [ ] Updated Atropos strategies based on real pruning outcomes
- [ ] Case study write-up demonstrating ROI validation
