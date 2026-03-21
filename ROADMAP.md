# Atropos roadmap

## Near term

- ✅ add notebook examples
- ✅ add CSV-to-markdown report generation
- ✅ expand comparison and sensitivity CLI output
- ✅ publish documentation pages
- [x] stabilize core pruning integrations (LLM-Pruner, Wanda, SparseGPT) - compatibility extended to GPT2, BLOOM, GPT-J, OPT, Pythia via patched_prune.py
  - Comprehensive test coverage across diverse model architectures
  - Clear documentation on limitations, dependencies, and failure modes
  - Validation suite to confirm model/environment compatibility
- [x] improve CI stability and reliability
- [ ] optimize validation script for GPU acceleration
- [x] add pruning result visualizations (charts, graphs)

## Medium term

- ✅ Monte Carlo uncertainty analysis
- ✅ telemetry import from benchmark runs
- ✅ experiment-tracker integration
- ✅ region-specific grid carbon presets
- [ ] multi-GPU benchmarking support
- [ ] distributed pruning experiments
- [x] quantization + pruning combination analysis
- [ ] automated hyperparameter tuning for pruning targets
- [ ] enhance error handling and debugging capabilities

## Long term

- ✅ lightweight web dashboard
  - Future enhancements: comparative views, what-if sliders, pipeline visualization
- ✅ richer cost models for GPU tiers and batching
- ✅ scenario calibration from real serving traces
- [ ] production deployment automation
- [ ] A/B testing framework for model variants
- [ ] continuous optimization pipeline
- [x] PyPI package release

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
- ✅ `atropos-llm collect-telemetry` CLI command
- ✅ `docs/telemetry-collection-guide.md` — Complete usage documentation
- ✅ Calibration integration with existing `atropos-llm calibrate` command

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
- ✅ `atropos-llm test-models` CLI command
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
- [x] Document break-even analysis with real data

**Deliverables:**
- ✅ `scripts/download_test_models.py` — Download script for 5 candidate models
- ✅ `scripts/project_savings.py` — Generate baseline projections
- ✅ `scripts/prune_models.py` — PyTorch-based pruning implementation
- ✅ `scripts/compare_projections.py` — Compare projected vs actual results
- ✅ `scripts/benchmark_quality.py` — Quality benchmarking script
- ✅ `scripts/generate_case_study.py` — Generate comprehensive case study report
- ✅ `scripts/upload_to_huggingface.py` — Upload pruned models to HF Hub
- ✅ `scripts/test_pruning_candidates.py` — Validation test runner
- ✅ `test_data/` — Local cache of 36+ GB models
- ✅ `test_data/projections.json/md` — Baseline projections
- ✅ `test_data/pruned_models/` — 8 pruned model variants
- ✅ `test_data/pruning_report.json/md` — Pruning operation results
- ✅ `test_data/comparison_report.json/md` — Projected vs actual comparison
- ✅ `test_data/benchmark_report.json/md` — Quality benchmark results
- ✅ `test_data/case_study.json/md` — Complete case study with break-even analysis
- ✅ Pruned models hosted on HuggingFace (4 validated models uploaded to https://huggingface.co/arsegarp)
- [x] Updated Atropos strategies based on real pruning outcomes
- ✅ `docs/case-study.md` — Comprehensive case study write-up

### 4. Pruned Model Validation ✅

**Goal:** Validate that pruned models maintain performance within acceptable tolerance compared to original models before HuggingFace upload.

**Tasks:**
- [x] Create side-by-side comparison script (original vs pruned)
- [x] Measure perplexity on validation dataset for both
- [x] Run identical generation tasks and compare outputs
- [x] Verify quality metrics are within tolerance (e.g., <20% degradation)
- [x] Document which models pass/fail validation
- [x] Gate HuggingFace upload on passing validation

**Deliverables:**
- ✅ `scripts/validate_pruned_models.py` — Compare original vs pruned performance
- [x] `test_data/validation_report.json/md` — Pass/fail results with metrics (requires models)
- ✅ Updated upload script with validation gate (`--force` to bypass)
- ✅ Validation criteria documented in script

**Note:** Validation requires both original and pruned models in `test_data/`. Run `download_test_models.py` and `prune_models.py` first if models are missing.

### 5. Advanced Pruning Frameworks ✅

**Goal:** Integrate and compare state-of-the-art pruning frameworks (Wanda, SparseGPT) against our PyTorch magnitude-based approach.

**Tasks:**
- [x] Integrate Wanda (Pruning by Weights AND Activations) - scripts created, compatibility with non-LLaMA models resolved via patched_prune.py
- [x] Integrate SparseGPT (GPT-specific pruning) - scripts created, compatibility with non-LLaMA models resolved via patched_prune.py
- [x] Run comparison: magnitude vs Wanda vs SparseGPT - comprehensive comparison script created
- [x] Measure quality/speed trade-offs
- [x] Update Atropos strategies with framework-specific recommendations

**Deliverables:**
- [x] `scripts/wanda_pruning.py` — Wanda integration (created, compatibility resolved via patched_prune.py)
- [x] `scripts/sparsegpt_pruning.py` — SparseGPT integration (created, compatibility resolved via patched_prune.py)
- [x] `scripts/compare_pruning_frameworks.py` — Comprehensive comparison of magnitude, wanda-patched, sparsegpt-patched
- [x] `docs/framework-comparison.md` — Comparison report (created)
- [x] Updated presets with framework-specific values

---

### 6. Quantization + Pruning Study ✅

**Goal:** Evaluate combined quantization and pruning optimizations.

**Tasks:**
- [x] Implement INT8 quantization pipeline
- [x] Test pruning + quantization combinations
- [x] Measure cumulative savings and quality impact
- [x] Document best practices for combined optimization

**Deliverables:**
- [x] `scripts/quantize_models.py` — Quantization pipeline
- [x] `scripts/combined_optimization.py` — Pruning + quantization
- [x] Comparison report showing combined ROI
