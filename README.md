# Atropos
> Terminology follows the canonical glossary: `/docs/canonical-glossary.md`.

Atropos is an **ROI estimation + optimization toolkit** for coding-LLM deployments, with a hardened runtime/API core for teams that need to run controlled experiments and ingestion workflows in production.

> Use `atropos` for Python imports and `atropos-llm` for the CLI.

## Operating modes (what Atropos is today)

Atropos supports three practical modes. Picking the right mode is the fastest way to set correct expectations.

### 1) Research mode

Use this mode when the goal is model/strategy exploration speed.

- Typical stack: notebooks, scripts, examples, validation modules, ad hoc pipeline runs.
- Storage/runtime posture: in-memory state and best-effort reliability are acceptable.
- Change tolerance: high; interfaces may evolve quickly.

### 2) Local dev mode

Use this mode when building features or integrations before production rollout.

- Typical stack: local API + worker, Docker Compose, contract/integration tests.
- Storage/runtime posture: reproducible local runs; Redis-backed runs recommended for parity checks.
- Change tolerance: moderate; stable platform contracts + evolving surrounding features.

### 3) Production mode

Use this mode for persistent runtime/API workloads with operational requirements.

- Typical stack: Runtime API + worker + Redis, health/readiness probes, structured logs, metrics dashboards.
- Storage/runtime posture: durable store, authenticated writes, readiness/dependency gating.
- Change tolerance: low on platform contracts; upgrades should follow documented compatibility guidance.

Deployment details for production mode are in [`docs/deployment.md`](docs/deployment.md).

## Maturity and stability

Atropos is intentionally mixed-maturity. Not every module should be treated as production-grade.

### Platform-grade today (recommended production dependency)

- Runtime API surface and health endpoints (`src/atroposlib/api/*`)
- Store abstraction + Redis-backed durability path (`src/atroposlib/api/storage.py`)
- Runtime controller and transport interfaces (`src/atroposlib/envs/runtime_controller.py`, `src/atroposlib/envs/transport_client.py`)
- Runtime observability primitives (`src/atroposlib/observability.py`)

### Supported but faster-evolving

- Pipeline orchestration (`src/atropos/pipeline/*`)
- A/B testing and quality workflows (`src/atropos/abtesting/*`, `src/atropos/quality/*`)
- Telemetry ingestion/calibration flows (`src/atropos/telemetry*`)

### Experimental / research-first

- Validation experiments and exploratory utilities (`src/atropos/validation/*`)
- Community/plugin and custom environment integrations (`src/environments/*`, `src/atroposlib/plugins/*`)
- Examples/scripts intended for iteration and learning (`examples/*`, `scripts/*`)

Canonical policy and compatibility expectations: [`docs/stability-tiers.md`](docs/stability-tiers.md).

## When to use Atropos / when not to

### Use Atropos when

- You need **ROI-first optimization planning** (cost, throughput, energy, break-even) with reproducible assumptions.
- You want to combine ROI estimation with **runtime ingestion + experiment operations** in one codebase.
- You need a pragmatic local-to-production path for API/worker runtime services.

### Do not choose Atropos when

- You need a fully managed, turnkey hosted platform with no self-operation footprint.
- Your requirement is strict long-term API stability across the entire repository (only platform-core surfaces are held to that bar).
- You only need a lightweight one-off benchmark script and do not need runtime services, storage contracts, or operational controls.

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

Atropos' primary identity is ROI estimation and optimization planning for coding-LLM deployments. It gives teams a reproducible way to estimate optimization ROI before committing engineering time. You can model how pruning and related changes alter memory, throughput, power, and annual cost, then compare savings to one-time project investment.

When you need operational support around that workflow, Atropos also provides:

- **Pipeline execution** for repeatable optimization/validation flows.
- **Validation tooling** for scenario and model checks.
- **Telemetry ingestion** to calibrate scenarios from observed runtime data.
- **A/B testing primitives** to compare variants and support rollout decisions.

## 2) Core concepts

Canonical definitions for **environment**, **trajectory**, **group**, **rollout**, and **server** live in `/docs/canonical-glossary.md`.

Use those definitions as the single mental model across this repository.

## 3) Architecture overview (current state)

Atropos currently has a layered architecture with different maturity levels:

1. **ROI and analysis layer (`src/atropos/*`)**
   - Scenario/strategy modeling and ROI calculation.
   - Validation, A/B testing, and pipeline orchestration modules (maturity varies by submodule).
2. **Platform runtime layer (`src/atroposlib/*`)**
   - API server, store contract, runtime controller, worker runtime, transport behavior.
   - Health/readiness/dependency endpoints and observability hooks.
3. **Integration and experimentation layer (`examples/*`, `scripts/*`, plugins/envs)**
   - Prototypes, experiments, adapters, and exploratory workflows.

Compact interaction view:

```text
Scenario + Strategy
  -> ROI calculation and decision support
  -> Optional runtime execution (API + worker + store)
  -> Validation / A-B comparison / telemetry calibration
```

This architecture supports both research iteration and platform operation, but only the platform runtime layer should be treated as production-grade by default.

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
