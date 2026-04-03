# Atropos Roadmap (2026 Decision Window)

<!--
This roadmap is intentionally structured to keep three outcomes viable:
1) Academic publication,
2) Commercial product/investment narrative,
3) Reputation-building open-source project.
Sections below explicitly map work to one or more of these paths so we do not overcommit too early.
-->

## Current Status & Known Gaps

<!--
Why this section exists (all paths):
- Academic path needs threats-to-validity transparency.
- Commercial path needs due-diligence honesty.
- Reputation path needs trust via candid status reporting.
-->

Atropos currently provides useful ROI estimation scaffolding, but it is **not yet proven** as a production-grade predictor for large-model pruning decisions.

### What is working today
- ROI estimation workflows, CLI/reporting, and baseline experimentation loops are available.
- GPT-2 level integration and testing has been the primary validation surface.

### Critical deficiencies (must-fix)
- **No validation data for models > 1B parameters**; current evidence is concentrated around GPT-2-scale testing.
- **Pruning framework integrations are fragile** (Wanda, SparseGPT, LLM-Pruner), with recurring dependency and environment conflicts.
- **No quality degradation model** yet (e.g., perplexity/task-accuracy impact as pruning ratio changes).
- **Missing cloud/hosting cost models** for AWS, Azure, and GCP (plus practical GPU hosts).
- **No published accuracy metrics or formal validation studies** for prediction quality.
- **External submodule dependencies create maintenance risk** and can break reproducibility.

### Delivery risk signal
- Current pace (~108 commits in 25 days) shows high execution velocity, but also indicates potential instability unless stabilization work is prioritized.

---

## Strategic Tracks (Run in Parallel, Re-balance at Day 120)

<!--
Why tracks instead of one linear roadmap:
- Academic, Commercial, and Reputation goals have overlapping foundations but different proof requirements.
- Track structure allows shared work while preserving optionality for the final direction.
-->

## Track A — Scientific Validation *(Primary: Academic, Secondary: Reputation)*

<!--
Supports Academic directly (publishable rigor), and Reputation indirectly (credible open benchmarks).
-->

### Goals
- Build evidence that Atropos predictions are statistically meaningful across modern LLM families.

### Priorities
1. **Benchmark coverage expansion**
   - Validate on multiple model families: **Llama, Mistral, Qwen** (with at least one 1B+, one ~7B, one ~13B representative checkpoint).
2. **Peer-review-ready methodology paper draft**
   - Explicit assumptions, dataset definitions, protocol, and reproducibility checklist.
3. **Statistical accuracy analysis**
   - Error distribution, confidence intervals, calibration plots, and outlier characterization.
4. **Ablation studies**
   - Quantify contribution of each ROI formula component (cost, throughput, quality proxy, infra assumptions).

### Success indicators
- Reproducible benchmark suite runs end-to-end.
- Draft manuscript and replication package skeleton are publicly reviewable.

---

## Track B — Production Readiness *(Primary: Commercial, Secondary: Reputation)*

<!--
Supports Commercial directly (buyer/investor confidence), and Reputation via practical reliability.
-->

### Goals
- Make Atropos robust enough for real deployment planning and external pilot usage.

### Priorities
1. **Cloud provider cost ingestion**
   - Integrate pricing/cost models for **AWS, Azure, GCP, Lambda Labs, RunPod**.
2. **Harden pruning integrations**
   - Support robust operation with containerized fallbacks for unstable dependency stacks.
3. **Quality degradation prediction v1**
   - Add probabilistic quality impact modeling for pruning strategies.
4. **SLA-worthy validation pipeline**
   - Deterministic runs, version-pinned environments, artifact retention, and regression gating.

### Success indicators
- Repeatable environment setup across clean machines.
- End-to-end validation pipeline can be run for external due diligence.

---

## Track C — Developer Experience *(Primary: Reputation, Secondary: Commercial)*

<!--
Supports Reputation directly (adoption and contribution quality), and Commercial indirectly (lower pilot friction).
-->

### Goals
- Make Atropos easy to understand, verify, and extend.

### Priorities
1. **Documentation overhaul with worked examples**
   - Clear quickstart, architecture notes, and interpretation guidance for ROI outputs.
2. **Interactive tutorial notebooks**
   - Step-by-step notebooks from raw inputs to decision-ready reports.
3. **Real-world case studies (including failures)**
   - Publish successes and misses to improve model trust and calibration.
4. **Codebase simplification**
   - Minimal dependency profile, stronger tests, and clearer module boundaries.

### Success indicators
- New contributor can run a complete example in one sitting.
- Case studies demonstrate honest learning, not only positive outcomes.

---

## Validation Milestones (Definition of “Proven”)

<!--
This milestone ladder aligns all paths on shared evidence thresholds:
- Bronze/Silver establish technical credibility,
- Gold establishes market credibility,
- Platinum establishes academic credibility.
-->

- **Bronze**: Works on **3 model sizes** (1B, 7B, 13B) with publicly posted results.
- **Silver**: Prediction error is within **20%** against real deployment telemetry (defined evaluation protocol).
- **Gold**: Used in production decision-making by **3+ external organizations**.
- **Platinum**: Peer-reviewed publication accepted, with a complete replication package released.

---

## Time-Boxed Plan (Aggressive but Honest)

<!--
Dates are relative to the current acceleration phase and explicitly account for instability risk from recent commit velocity.
-->

### Next 30 days (Stabilization Sprint)
- Stabilize current integrations and dependency matrix.
- Complete first serious validation pass on **7B-class models**.
- Publish a transparent “known breakages” compatibility table.

### By 60 days
- Deliver **cloud cost models v1** (AWS/Azure/GCP + Lambda Labs/RunPod baselines).
- Ship **quality degradation model v1** with uncertainty bounds.

### By 90 days
- Publish first **external user case study** (acceptable even if outcome is negative or mixed).
- Convert lessons into calibration and documentation updates.

### By 120 days (Decision Gate)
Choose primary direction for the next major cycle while preserving secondary paths:
- **Option 1:** Academic paper submission package.
- **Option 2:** Commercial MVP + pilot collateral.
- **Option 3:** Reputation-focused open-source release with high contributor UX.

---

## What We Won’t Do (Scope Controls)

<!--
Why explicit non-goals matter:
- Academic: preserves methodological focus.
- Commercial: prevents roadmap dilution before PMF.
- Reputation: avoids overpromising to contributors.
-->

- We **won’t support every pruning framework**; we will prioritize the top 2 most stable/high-impact options.
- We **won’t build custom pruning algorithms** in this phase; we will integrate and evaluate existing methods.
- We **won’t provide deep on-prem hardware cost modeling** beyond basic, transparent formulas.
- We **won’t guarantee exact accuracy predictions**; outputs are probabilistic decision support, not certainty.

---

## Decision Log

<!--
Decision log keeps strategic intent explicit so future contributors understand why tradeoffs were made.
-->

1. **Adopted a 3-track roadmap** to keep Academic, Commercial, and Reputation paths simultaneously viable until the Day-120 gate.
2. **Elevated transparency** by adding explicit known gaps, including >1B validation absence and fragile integrations.
3. **Defined milestone ladder (Bronze→Platinum)** to align discussions around evidence, not feature count.
4. **Time-boxed next 120 days** around stabilization, validation, and one external proof point before strategic commitment.
5. **Set explicit non-goals** to avoid premature expansion into custom algorithms and broad framework sprawl.
