# Atropos
> Terminology follows the canonical glossary: `/docs/canonical-glossary.md`.



## Quickstart (under 10 minutes)

Use the golden onboarding path:

```bash
make quickstart
make run-golden
```

See `docs/onboarding.md` for the full onboarding workflow and troubleshooting.


## 1) What problem this solves

Atropos gives you a way to run optimization experiments for LLM inference and keep the decision process reproducible. It helps you move from one-off benchmark numbers to a step-by-step record of what was tried, what changed, and what the outcome was. In practice, teams struggle because model behavior, infrastructure behavior, and cost behavior are measured in different places and rarely tied together. Atropos organizes those measurements into one execution flow so analysis, validation, and rollout decisions use the same data model. The result is a system you can inspect later to answer: what input was used, what action was taken, what reward or metric changed, and whether the change should be promoted.

## 2) Core concepts

Canonical definitions for **environment**, **trajectory**, **group**, **rollout**, and **server** live in `/docs/canonical-glossary.md`.

Use those definitions as the single mental model across this repository.

## 3) System architecture

Atropos is organized as a pipeline around the concepts above.

1. A run starts in an **environment** that owns episode lifecycle (`reset`, `step`, terminal conditions).
2. The environment calls a model/inference **server** through a client layer that handles transport concerns (timeouts, retries, async behavior).
3. Returned outputs are parsed into typed actions, validated, and applied to transition state.
4. Reward/statistics are computed from the transition outcome.
5. Each step is appended to a **trajectory** store with telemetry and identifiers.
6. For comparative experiments, results are aggregated by **group** (control/treatment) and passed to statistical analysis.
7. A **rollout** decision uses explicit gates to promote, hold, or reject the treatment in production.

A compact interaction view:

```text
Environment
  -> Server client (collect/generate)
  -> Action parse + validate
  -> State transition
  -> Reward/statistics
  -> Trajectory append
  -> Group-level comparison (if A/B test)
  -> Rollout gate decision
```

This separation keeps transport, state logic, scoring, and persistence independently testable.

## 4) Minimal working example (end-to-end)

This example runs one control/treatment experiment against a local server, analyzes it, and exports the result.

### Step A: install and prepare

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

### Step B: define an experiment with two groups

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

### Step C: start the run

```bash
atropos-llm ab-test create experiment.yaml --start
```

This creates groups, starts collection from each server endpoint, and begins building trajectories for the run.

### Step D: inspect status

```bash
atropos-llm ab-test list --status running
atropos-llm ab-test status <experiment-id>
```

### Step E: analyze and export

```bash
atropos-llm ab-test analyze <experiment-id> --format markdown
atropos-llm ab-test analyze <experiment-id> --format json > result.json
```

At this point you have a complete chain: environment execution, server measurements, group comparison, and trajectory-backed output suitable for review or automation.

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
