# Atropos
> Terminology follows the canonical glossary: `/docs/canonical-glossary.md`.

Atropos is an **ROI estimation + optimization toolkit** for coding-LLM deployments. It helps teams forecast cost/energy/performance impact and decide which optimization work is worth shipping. Beyond ROI estimation, Atropos also includes practical modules for **pipeline orchestration, validation, telemetry ingestion, and A/B testing**.

> Use `atropos` for Python imports and `atropos-llm` for the CLI.


## Stability tiers and product maturity

Atropos is production-oriented for **platform core runtime surfaces**, while other parts of the repository are intentionally at different maturity levels.

- **Tier 1 (platform core):** API server, store layer, runtime loop, and transport components in `atroposlib` are the most stable surfaces for production dependency.
- **Tier 2 (supported research features):** validation, A/B testing, pipeline, and related research modules are maintained and supported but may evolve faster than Tier 1.
- **Tier 3 (experimental/community environments):** environment/plugin integrations and community extensions are intentionally experimental.

Compatibility expectations and module mapping are defined in `docs/stability-tiers.md`.

## Quickstart (under 10 minutes)

Use the minimal first-run path:

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -e .
python examples/minimal/toy_env.py
python examples/minimal/toy_trainer_walkthrough.py
```

See `docs/onboarding.md` for the focused quickstart + onboarding checklist.

## Install

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -e .
```

Optional extras:

- `pip install -e .[dev]`
- `pip install -e .[tuning]`
- `pip install -e .[dashboard]`

CLI entry point:

```bash
atropos-llm --help
```

Python import package name is `atropos` (distribution/CLI name is `atropos-llm`).

## One-command demo (<5 minutes to first success)

Use the demo command to run a full local workflow end to end:

```bash
atropos-llm demo
```

What this does automatically:

1. Starts a local Atropos API server.
2. Starts a simple `LineWorld` environment.
3. Simulates a trainer loop.
4. Streams live metrics in the terminal (`reward`, `position`, `queue_depth`).

The default demo settings live in `configs/demo.yaml`. You can override them:

```bash
atropos-llm demo --config configs/demo.yaml
```

## 1) What problem this solves

Atropos' **primary identity** is ROI estimation and optimization planning for coding-LLM deployments. It gives you a reproducible way to estimate optimization ROI before committing engineering time. You can model how pruning and related changes alter memory, throughput, power, and annual cost, then compare savings to one-time project investment. This makes go/no-go decisions explicit instead of relying on disconnected benchmark snapshots.

When you need end-to-end operational support around that ROI workflow, Atropos also provides **secondary modules**:

- **Pipeline execution** for repeatable optimization/validation flows.
- **Validation tooling** for scenario and model checks.
- **Telemetry ingestion** to calibrate scenarios from observed runtime data.
- **A/B testing primitives** to compare variants and support rollout decisions.

## 2) Core concepts

Canonical definitions for **environment**, **trajectory**, **group**, **rollout**, and **server** live in `/docs/canonical-glossary.md`.

Use those definitions as the single mental model across this repository.

## 3) System architecture

Atropos is organized as an ROI-first toolkit with modular execution components.

1. A run starts from a deployment scenario and optimization strategy (or preset pair).
2. The calculator produces baseline vs optimized metrics and ROI outputs.
3. Pipeline and validation modules can execute reproducible checks around those scenarios.
4. Telemetry modules can ingest runtime signals and map them into scenario inputs.
5. A/B testing modules can aggregate control/treatment outcomes for rollout decisions.

A compact interaction view:

```text
Scenario + Strategy
  -> ROI calculation (cost/perf/energy + break-even)
  -> Validation / pipeline execution
  -> Telemetry calibration (optional)
  -> A/B comparison + rollout gate (optional)
```

This separation keeps core ROI calculation deterministic while supporting production-oriented modules around it.

## 4) Minimal working example (end-to-end)

This example creates one control/treatment experiment, starts it, and analyzes the result.

### Step A: define an experiment with two groups

Create `experiment.yaml`:

```yaml
name: latency-throughput-check
hypothesis: treatment improves throughput without unacceptable latency increase
control_variant:
  id: base
  endpoint: http://localhost:8000
  traffic_percent: 50

treatment_variant:
  id: pruned
  endpoint: http://localhost:8001
  traffic_percent: 50

metrics:
  - throughput
  - latency_p95
  - error_rate

stopping_criteria:
  min_samples_per_variant: 200
  max_duration_minutes: 30
```

### Step B: create the experiment

```bash
atropos-llm ab-test create --config experiment.yaml
```

### Step C: inspect status

```bash
atropos-llm ab-test list --status running
atropos-llm ab-test status <experiment-id>
```

### Step D: analyze and export

```bash
atropos-llm ab-test analyze <experiment-id> --format markdown
atropos-llm ab-test analyze <experiment-id> --format json > result.json
```

For complete packaging/config/CLI source-of-truth tables, see `CONFIG.md`.

## 5) Common pitfalls

1. **Mixing synthetic and real metrics without labeling source**  
   If collection falls back to synthetic data, mark it explicitly before using results for promotion.

2. **Treating aggregate statistics as raw observations**  
   Reconstructed samples can be useful for rough inference, but they are not equivalent to raw per-request traces.

3. **Inconsistent status/schema handling across storage boundaries**  
   Keep enum/string conventions and schema versions consistent when writing and reading experiment records.

4. **Letting mutable runner state hide history**  
   Prefer append-only step records so you can replay how the trajectory evolved instead of only seeing the latest snapshot.

5. **Coupling concurrency logic to scoring logic**  
   Timeouts/retries belong in runtime transport code; statistical computation should stay pure and deterministic.

6. **Skipping parity tests during refactors**  
   When extracting environment components, keep golden trajectory tests to ensure behavior does not drift.
