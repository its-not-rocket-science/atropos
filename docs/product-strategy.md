# Product Strategy: Making Atropos the Default Platform
> Terminology follows the canonical glossary: `/docs/canonical-glossary.md`.

This strategy defines the target users, the workflow each user needs to succeed, the friction blocking adoption today, and the platform features required for Atropos to become the default choice.

## 1) Target users

Atropos should optimize for three primary user archetypes:

1. **Researcher** (applied ML, model optimization, evaluation)
2. **Infrastructure Engineer** (platform/SRE/MLOps responsible for reliability and cost)
3. **Hobbyist** (indie developer or small-team builder shipping practical apps)

These archetypes map directly to distinct success criteria:

- Researchers optimize **experiment velocity + scientific confidence**.
- Infrastructure engineers optimize **safe production rollout + operational efficiency**.
- Hobbyists optimize **time-to-value + simplicity**.

## 2) Ideal workflow and pain points by user

### A) Researcher

#### Ideal workflow

1. Start from a reproducible preset and a documented baseline.
2. Run multiple optimization strategies (pruning, quantization, combined) on the same task set.
3. Track quality, latency, throughput, and cost in a single trajectory-backed artifact.
4. Compare runs statistically with confidence intervals and significance guidance.
5. Promote the best candidate into an integration-ready package with full experiment provenance.

#### Pain points today

- Fragmented tooling across notebooks, scripts, and ad-hoc evaluators.
- Poor reproducibility (seed/config/data drift across runs).
- Slow, manual comparison loops when testing many configurations.
- Difficulty converting offline benchmark wins into production-ready recommendations.

### B) Infrastructure Engineer

#### Ideal workflow

1. Register model endpoints and deployment constraints once.
2. Run controlled control-vs-treatment experiments in staging with explicit rollout gates.
3. Monitor SLO-sensitive metrics (p95 latency, error rate, capacity headroom, cost/hour).
4. Automatically receive pass/hold/reject recommendations with rollback plans.
5. Export audit-ready reports for operations and leadership review.

#### Pain points today

- Lack of a single source of truth for optimization decisions.
- High risk of regressions during deployment handoffs from research.
- Manual rollouts with weak guardrails and inconsistent decision criteria.
- Poor observability tying model changes to infra incidents and spend shifts.

### C) Hobbyist

#### Ideal workflow

1. Install Atropos quickly and run a guided setup in under 10 minutes.
2. Select a goal (cheaper inference, faster responses, smaller memory footprint).
3. Execute a one-command optimization + validation path.
4. Get a clear "ship/don't ship" recommendation with plain-language tradeoffs.
5. Deploy with generated configs and minimal custom ops work.

#### Pain points today

- Steep learning curve around optimization methods and metrics.
- Too many knobs for users with limited infra or ML depth.
- Confusing output that does not clearly answer "is this worth it?"
- Weak defaults for local, low-budget, and GPU-constrained environments.

## 3) Features required to become the default choice

Atropos should win by being the fastest path from idea to confident deployment.

### A) Product pillars

1. **Reproducibility by default**
   - Immutable run manifests (config, model hash, dataset snapshot, seed).
   - One-click rerun of any prior experiment.
   - Built-in lineage graph from baseline to promoted candidate.

2. **Decision automation, not just measurement**
   - Policy engine for promotion gates (quality floor, latency ceiling, cost threshold).
   - Auto-generated recommendations: promote, hold, reject.
   - Risk summaries with confidence annotations and failure-mode flags.

3. **Opinionated golden paths per user type**
   - Research path: batch experiment matrix + significance-first comparison.
   - Infra path: staged rollout templates + rollback automation.
   - Hobbyist path: minimal wizard + safe defaults + plain-language reports.

4. **Ecosystem interoperability**
   - First-party integrations for common inference stacks and observability backends.
   - Exporters to markdown/JSON + CI-native outputs for PR checks.
   - Stable SDK and API contracts to prevent integration churn.

5. **Trust and governance**
   - Auditable artifacts for compliance and postmortems.
   - Signed run metadata and reproducibility attestations.
   - Team-level policy packs (startup, enterprise, regulated modes).

### B) Concrete feature roadmap (priority order)

#### Priority 0: Adoption accelerators (must-have)

- `atropos quickstart --persona researcher|infra|hobbyist`
- Benchmark-to-decision report with explicit recommendation and rationale.
- Preset libraries by model family and deployment target.
- CI action that fails PRs when optimization violates guardrail policies.

#### Priority 1: Platform stickiness (should-have)

- Experiment registry UI with lineage and run diffing.
- Scenario simulator for projected savings vs quality risk.
- Staged rollout orchestrator with canary + automatic rollback.
- Team workspaces with shared baselines and policy inheritance.

#### Priority 2: Strategic moat (differentiators)

- Cross-org anonymized benchmark network (opt-in) to compare against peers.
- Auto-tuning agent proposing candidate optimizations under user constraints.
- Domain packs (coding assistants, chatbots, RAG workloads, edge inference).

## 4) What should be removed to accelerate adoption

To become default, Atropos should remove friction aggressively.

1. **Remove duplicate pathways that produce overlapping outputs**
   - Consolidate scattered scripts into a small set of supported CLI workflows.

2. **Remove low-signal configuration surface area from default mode**
   - Hide advanced knobs unless `--advanced` is enabled.

3. **Remove ambiguous metrics and unclear labels in reports**
   - Standardize to a canonical core metric set with definitions.

4. **Remove non-opinionated onboarding**
   - Replace open-ended setup with persona-based guided flows.

5. **Remove undocumented or unstable interfaces from public promises**
   - Mark experimental APIs clearly and keep stable APIs narrow and versioned.

6. **Remove manual promotion decisions where policy can automate**
   - Shift from human-only interpretation to policy-backed recommendations.

## 5) Success criteria (12-month horizon)

- **Activation**: New users complete first meaningful run in < 15 minutes.
- **Time-to-decision**: 50% reduction from first benchmark to rollout recommendation.
- **Reproducibility**: > 95% of promoted decisions are rerunnable from stored artifacts.
- **Operational confidence**: Measurable decrease in rollout regressions.
- **Retention**: Teams run repeated optimization cycles rather than one-off evaluations.

## Strategy summary

Atropos becomes the default choice by delivering:

- The **researcher's fastest reproducible experiment loop**,
- The **infrastructure engineer's safest rollout system**, and
- The **hobbyist's simplest path to concrete value**.

The platform should prioritize opinionated workflows, automation-backed decisions, and ruthless removal of complexity that does not directly improve decision quality.
