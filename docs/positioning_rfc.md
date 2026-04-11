# RFC: Atropos Positioning and Scope Boundaries
> Terminology follows the canonical glossary: `/docs/canonical-glossary.md`.

- **Status:** Proposed
- **Authors:** Atropos maintainers
- **Last updated:** 2026-04-11
- **Audience:** Maintainers, contributors, adopters

## 1. Summary

This RFC clarifies Atropos positioning to reduce architectural drift and scope creep.

**Decision:** Atropos is **primarily a production RL platform for decision support and optimization governance**, with:

- a **limited research mode** for fast offline iteration, and
- a **supporting (not primary) data pipeline layer** for trajectory capture and reproducibility.

In short:

1. **Research playground?** **Partially** (constrained, reproducibility-first, not open-ended).
2. **Production RL platform?** **Yes (primary identity).**
3. **Data pipeline layer?** **Only as an enabling subsystem, not the product center.**

## 2. Problem Statement

Atropos has evolved across experimentation, benchmarking, telemetry, and runtime concerns. Without a clear center, the codebase risks:

- accumulating overlapping abstractions,
- broadening public APIs without long-term ownership,
- delaying production-grade hardening while adding experimental surface area.

This RFC defines a narrow strategic center and explicit boundaries for what gets first-class support.

## 3. Positioning Decision

### 3.1 Primary Positioning

Atropos is a **production RL platform** that helps teams:

- run controlled optimization/evaluation loops,
- capture decision-grade evidence (quality, latency, cost, risk),
- enforce promotion/rollback policies from reproducible artifacts.

### 3.2 Secondary Positioning

Atropos includes:

- a **research-facing workflow** for local or staging experimentation,
- a **pipeline substrate** for ingesting/storing trajectory and telemetry artifacts.

These are productized only to the extent they strengthen the production decision loop.

### 3.3 Positioning Test (for new proposals)

A new feature is in-scope only if it materially improves at least one of:

1. **Production decision confidence** (better promotion/reject signals),
2. **Operational safety** (fewer regressions, better rollback/readiness),
3. **Reproducibility and auditability** (deterministic reruns and lineage).

If a proposal mainly adds exploratory flexibility without improving these, it is out-of-scope or experimental.

## 4. Core Abstractions

The following abstractions are the long-lived core:

1. **Environment Contract**
   - Stable interface for tasks, rollouts, rewards/signals, and step lifecycle.
   - Purpose: isolate scenario logic from orchestration/runtime concerns.

2. **Policy/Strategy Unit**
   - Encapsulates an optimization or control policy (e.g., baseline vs candidate).
   - Purpose: make comparisons explicit and reproducible.

3. **Run Specification (RunSpec)**
   - Immutable, versioned experiment/run definition: configs, seeds, dataset snapshot IDs, model identity.
   - Purpose: deterministic reruns and cross-team consistency.

4. **Trajectory Record**
   - Canonical event schema for steps, outcomes, resource usage, and quality signals.
   - Purpose: single source of truth for both offline analysis and production evidence.

5. **Evaluation Gate**
   - Policy-driven pass/hold/reject criteria spanning quality, latency, cost, reliability.
   - Purpose: convert metrics into operational decisions.

6. **Promotion Artifact**
   - Signed/traceable bundle of evidence, decision rationale, and deployment metadata.
   - Purpose: enforce auditability and safe handoff into runtime systems.

## 5. Non-Goals

The following are intentionally not first-class Atropos goals:

1. **General-purpose data engineering platform**
   - No ambition to replace warehouse ETL frameworks, stream processors, or generic lakehouse stacks.

2. **Open-ended research sandbox with unconstrained APIs**
   - Atropos supports experimentation, but not at the cost of stable production contracts.

3. **Full model training framework replacement**
   - Atropos may integrate with training tooling; it does not aim to become a broad training stack.

4. **Observability-suite replacement**
   - Atropos emits decision-centric telemetry and integrates with existing observability systems.

5. **One-off benchmarking utilities without decision path**
   - Metrics without policy/gating relevance should remain peripheral.

## 6. Stability Guarantees

Stability is tiered to prevent accidental contract expansion.

### 6.1 Stable (SemVer-protected)

- Public CLI commands documented in `docs/cli.md`.
- Public Python APIs documented in `docs/api.md` and explicitly marked stable.
- Canonical trajectory schema and required fields for compatibility.
- RunSpec serialization format and version negotiation behavior.

**Guarantee:** Breaking changes only in major releases, with migration guidance.

### 6.2 Beta (Explicitly evolving)

- Feature-flagged orchestration helpers.
- Advanced optimization plugins with evolving extension points.
- Experimental rollout automation adapters.

**Guarantee:** Best-effort backward compatibility; deprecations announced before removal when feasible.

### 6.3 Experimental (No compatibility promise)

- Prototype APIs hidden behind explicit experimental namespaces/flags.
- Internal-only data fields not included in canonical schema contracts.

**Guarantee:** May change or be removed without deprecation windows.

## 7. Roadmap Aligned With Positioning

### Phase 1 (0-3 months): Contract Consolidation

- Freeze and document the core abstractions (Environment Contract, RunSpec, Trajectory Record, Evaluation Gate).
- Remove or quarantine duplicate APIs that bypass canonical run and trajectory paths.
- Add compatibility tests for schema/version behavior.

**Exit criteria:** New features cannot ship without mapping to a core abstraction and stability tier.

### Phase 2 (3-6 months): Production Safety and Decisioning

- Harden promotion gating with policy packs (quality/latency/cost/reliability).
- Standardize promotion artifacts and audit reports.
- Improve rollback/readiness integration points for runtime workflows.

**Exit criteria:** At least one end-to-end “evaluate → gate → promote/hold” path is deterministic and documented.

### Phase 3 (6-12 months): Ecosystem and Scale

- Expand adapters and plugins only where they preserve core contracts.
- Provide migration tooling for older trajectory and run formats.
- Add governance-level UX/reporting for multi-team environments.

**Exit criteria:** Upgrades preserve decision evidence continuity across versions.

## 8. Governance Rules to Prevent Scope Creep

For every architecture proposal and major PR, require a brief “Positioning Check”:

1. Which core abstraction does this change strengthen?
2. Which stability tier is affected?
3. What explicit non-goal does this avoid?
4. How does it improve production decision confidence/safety/reproducibility?

Any proposal that cannot answer these in concrete terms should not be accepted as core roadmap work.

## 9. Risks and Mitigations

- **Risk:** Research users feel constrained.
  - **Mitigation:** Keep a clearly labeled experimental lane with strict boundary to stable interfaces.

- **Risk:** Existing modules rely on overlapping legacy paths.
  - **Mitigation:** Provide deprecation maps and migration helpers tied to RunSpec/Trajectory canonicalization.

- **Risk:** Short-term velocity drop during consolidation.
  - **Mitigation:** Timebox Phase 1 and enforce compatibility tests to avoid repeated redesign cycles.

## 10. Final Positioning Statement

Atropos is a **production RL decision platform** with research and pipeline capabilities intentionally scoped to support that mission. Every durable investment should improve **decision quality, operational safety, or reproducibility**; everything else remains optional or experimental.
