# Atropos roadmap

## Near term

- ✅ add notebook examples
- ✅ add CSV-to-markdown report generation
- ✅ expand comparison and sensitivity CLI output
- ✅ publish documentation pages
- ✅ stabilize core pruning integrations (LLM-Pruner, Wanda, SparseGPT) - compatibility extended to GPT2, BLOOM, GPT-J, OPT, Pythia via patched_prune.py
  - Comprehensive test coverage across diverse model architectures
  - Clear documentation on limitations, dependencies, and failure modes
  - Validation suite to confirm model/environment compatibility
- ✅ improve CI stability and reliability
- ✅ optimize validation script for GPU acceleration
- ✅ add pruning result visualizations (charts, graphs)
- ✅ standardize CLI examples to consistently use atropos-llm in all documentation
- ✅ add feature overview table at top of README for scannability
- ✅ create CONTRIBUTING.md with PR workflow, coding standards, and testing requirements
- ✅ add documentation badge linking to hosted docs (GitHub Pages)
- ✅ add note in validation docs clarifying GPT-2 is for pipeline testing only, not representative of 7B+ models
- ✅ make dashboard dependencies optional (pip install atropos-llm[dashboard]) to keep core lightweight
- ✅ add make dev-full target that sets up dedicated environment with all pruning frameworks

## Medium term

- ✅ Monte Carlo uncertainty analysis
- ✅ telemetry import from benchmark runs
- ✅ experiment-tracker integration
- ✅ region-specific grid carbon presets
- [x] multi-GPU benchmarking support
- [x] distributed pruning experiments
- ✅ quantization + pruning combination analysis
- ✅ automated hyperparameter tuning for pruning targets
- ✅ enhance error handling and debugging capabilities (structured logging, debug/verbose modes, traceback flag, error context)

## Long term

- ✅ lightweight web dashboard
  - Future enhancements: comparative views, what-if sliders, pipeline visualization
- ✅ richer cost models for GPU tiers and batching
- ✅ scenario calibration from real serving traces
- [ ] production deployment automation
- [ ] A/B testing framework for model variants
- [ ] continuous optimization pipeline
- ✅ PyPI package release

## Version Planning

*Three-phase progression from current alpha to autonomous operations*

**Phase 1: Core Stabilization & Foundation (v0.6.0 - v1.0.0)**
- **v0.5.0** (current)
  - Multi-GPU benchmarking support
  - Distributed pruning experiments
  - Optimize validation script for GPU acceleration
- **v0.6.0**
  - Production deployment automation
  - A/B testing framework for model variants
  - *Bridge to vision:* Basic cost anomaly detection
- **v0.7.0**
  - Continuous optimization pipeline
  - *Bridge to vision:* Pipeline lineage tracking
- **v0.8.0** (AI-Native Foundation)
  - LLM-powered optimization recommendations
  - Hyperparameter discovery prototype
  - Natural language alerts for optimization thresholds
- **v0.9.0** (Enhanced Observability)
  - Real-time dashboard improvements
  - SLA prediction based on historical data
  - Quality risk monitoring
- **v1.0.0** (Stable API & Enterprise Foundations)
  - Stable public API
  - Basic team collaboration (shared libraries, comment threads)
  - Audit trail for optimization decisions
  - Comprehensive documentation

**Phase 2: Intelligence & Collaboration (v1.1.0 - v1.5.0)**
- **v1.1.0** (Advanced AI Intelligence)
  - Self-healing pipeline components
  - Automated regression detection
  - Intelligent fallback strategies
- **v1.2.0** (Team Collaboration)
  - Approval workflows
  - Slack/Teams integration for notifications
  - Role-based access control
- **v1.3.0** (Multi-Model Orchestration)
  - Fleet optimization across multiple models
  - Dependency-aware scheduling
  - Budget allocation across projects
- **v1.4.0** (Regulatory Compliance)
  - Model card generation
  - GDPR/SOC2 compliance automation
  - Ethics review workflows
- **v1.5.0** (Edge & Embedded)
  - Mobile deployment ROI estimation
  - Raspberry Pi/edge device optimization profiles
  - Battery consumption estimation

**Phase 3: Scale & Autonomy (v2.0.0 - v3.0.0)**
- **v2.0.0** (Multi-Model Platform)
  - Cross-model dependency management
  - Portfolio-level optimization
  - Advanced budget orchestration
- **v2.1.0** (Advanced Edge)
  - WebAssembly/ONNX runtime support
  - Heterogeneous hardware optimization
  - Cross-platform deployment automation
- **v3.0.0** (Autonomous Operations)
  - Fully autonomous optimization loops
  - Predictive maintenance
  - Continuous adaptation to changing conditions

*See [Strategic Themes](#strategic-themes-2026-2027-vision) for detailed feature mapping.*

## Future Exploration

- Model compression beyond pruning (knowledge distillation, low-rank adaptation)
- Hardware-specific optimization profiles (NVIDIA vs AMD vs AWS Inferentia)
- Integration with more deployment platforms (TensorRT-LLM, ONNX Runtime, Apple MLX)
- Real-time optimization suggestions based on live telemetry
- Automated optimization across multiple model variants (Pareto frontier analysis)

## Strategic Themes (2026-2027 Vision)

*Based on community feedback envisioning v1.3.0-v3.0.0 capabilities, mapped to realistic phased progression from current v0.5.0 alpha state.*

### AI Intelligence
- LLM-powered optimization recommendations (v0.8.0)
- Hyperparameter discovery with Bayesian optimization (v0.8.0)
- Self-healing pipeline with automatic rollback (v1.1.0)
- Natural language alerts and reports (v0.8.0)
- Predictive scaling: forecast when pruning becomes cost-effective based on traffic growth patterns
- Automated regression detection between pruned and baseline model performance
- Cost-aware pruning: optimize for electricity price fluctuations and regional cost variations
- Quality-aware scheduling: run fine-tuning during periods of lower quality tolerance
- Intelligent fallback strategies for queries where pruning affects quality

### Deep Observability
- Cost anomaly detection with statistical baselines (v0.6.0)
- End-to-end pipeline lineage tracking (v0.7.0)
- Real-time dashboard with WebSocket updates (v0.9.0)
- SLA breach prediction based on historical patterns (v0.9.0)
- Circuit breakers for pruning framework failures with graceful degradation
- Retry logic with exponential backoff for telemetry collection and framework operations
- Pipeline pause/resume capability for long-running optimization jobs
- Encrypted secrets management for cloud credentials and API keys
- Production deployment guide with load testing results and best practices
- SLI/SLO tracking for pipeline execution reliability

### Team Collaboration & Compliance
- Optimization approval workflows with multi-level gating (v1.2.0)
- Comment threads on pipeline runs (v1.0.0)
- Shared scenario libraries with versioning (v1.0.0)
- Slack/Teams integration for notifications and approvals (v1.2.0)
- Audit trail for all optimization decisions with export capabilities
- Compliance automation reports (GDPR, SOC2, etc.)
- Model card generation for pruned models (v1.4.0)
- Export to regulatory formats (CSV/JSON for auditors) (v1.4.0)
- Data lineage tracking for pruning decisions (v1.4.0)
- Ethics review workflow for fairness impact (v1.4.0)
- License compliance checker for model licenses (v1.4.0)

### Multi-Model Scale
- Fleet optimization across hundreds of models (v1.3.0)
- Dependency-aware scheduling for model variants (v1.3.0)
- Budget allocation optimizer across portfolio (v1.3.0)
- Cross-model transfer learning of pruning patterns (v2.0.0)
- Fleet-wide anomaly correlation detection (v2.0.0)
- Automated rollback at fleet level for systemic issues (v2.0.0)
- Multi-region deployment support for distributed optimization pipelines
- High availability mode for pipeline controller with failover
- Customer success portal with usage analytics and optimization insights

### Edge & Deployment
- Mobile deployment profiles for iOS/Android (v1.5.0)
- Raspberry Pi optimization targets (v1.5.0)
- WebAssembly export for browser deployment (v2.1.0)
- ONNX runtime integration for cross-platform deployment (v2.1.0)
- Battery life estimation for edge devices (v1.5.0)
- Quantization-aware pruning for NPUs (v2.1.0)
- Plugin system for custom pruning frameworks and optimization algorithms
- VS Code extension with inline ROI estimates during model development
- GitHub Copilot integration: suggest pruning when code changes affect deployment costs
- Terraform provider for infrastructure-as-code deployment of Atropos resources
- OpenTelemetry support for distributed tracing of optimization pipelines
- Community templates repository for sharing optimization strategies

### Advanced Optimization Techniques
- Speculative decoding optimization estimation and ROI analysis
- Continuous batching efficiency modeling for throughput optimization
- Prefix caching ROI calculation for repeated prompt patterns
- FlashAttention memory impact projections and optimization trade-offs
- Multi-GPU tensor parallelism optimization strategies
- LoRA adapter pruning strategies for fine-tuned models
- Quantization-aware fine-tuning calibration and quality preservation

### Autonomous Operations
- Fully autonomous optimization with no human-in-loop (v3.0.0)
- Continuous optimization loop adapting to traffic patterns (v3.0.0)
- Self-tuning thresholds based on historical success (v3.0.0)
- Predictive maintenance for pruned models (v3.0.0)
- Automatic A/B test termination when statistical significance reached (v3.0.0)
- Budget-aware autonomous mode within defined cost constraints (v3.0.0)

### Nice-to-Have Backlog
- JupyterLab extension with interactive widgets
- PostgreSQL/MySQL backend for large deployments
- GraphQL API for flexible queries
- Webhook actions for external system integration
- Data export to Snowflake/BigQuery for enterprise analytics
- Custom Python operators for injecting custom logic in pipelines
- Mobile app (iOS/Android) for monitoring approvals
- Desktop app (Electron-based) for local use
- Chrome extension for ROI estimates on HuggingFace model pages
- Airflow integration as Airflow operator
- Kubeflow integration as native pipeline component
- Ray integration for distributed pruning across cluster
- Unified metrics API for any quality metric
- Benchmark suite for continuous performance regression testing
- Community forum for optimization strategy discussion

## Long-term Vision

*Note: The items previously in this section have been integrated into the [Strategic Themes](#strategic-themes-2026-2027-vision) section with version planning and thematic organization.*

**Production Readiness** → **Deep Observability** theme
**Advanced Intelligence** → **AI Intelligence** theme
**Enterprise Features** → **Team Collaboration & Compliance** theme
**Ecosystem Expansion** → **Edge & Deployment** theme
**Specialized Optimizations** → **Advanced Optimization Techniques** theme

*Refer to the Strategic Themes section for detailed feature mapping across versions v0.6.0 through v3.0.0.*

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
