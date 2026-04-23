# Atropos Documentation
> Terminology follows the canonical glossary: `/docs/canonical-glossary.md`.

**Atropos** combines an ROI estimation toolkit with a production-capable runtime/API core.

> Use `atropos` for Python imports and `atropos-llm` for the CLI.

## Operating modes at a glance

- **Research mode:** fast iteration using examples/scripts/validation with minimal operational guarantees.
- **Local dev mode:** reproducible developer workflow with local API/worker and parity-oriented testing.
- **Production mode:** hardened runtime operation (API + worker + Redis) with auth, readiness/dependency checks, and observability.

See [Deployment Guide](deployment.md) for production-mode artifacts and [Stability Tiers](stability-tiers.md) for compatibility expectations.

## Maturity map

- **Platform-grade now:** `src/atroposlib/api/*`, storage contract/backends, runtime controller/transport, observability hooks.
- **Supported but evolving:** pipeline, A/B testing, telemetry calibration, quality tooling in `src/atropos/*`.
- **Experimental:** validation experiments, plugin/environment integrations, many examples/scripts.

## Quick Links

- [Canonical Glossary](canonical-glossary.md)
- [Stability Tiers](stability-tiers.md)
- [Deployment Guide](deployment.md)
- [Environment Variables](environment_variables.md)
- [Platform Architecture RFC](platform_architecture_rfc.md)
- [Adoption Assessment (2026-04)](adoption_assessment_2026_04.md)
- [Production Readiness Audit (2026-04)](production_readiness_audit_2026_04.md)

- [Installation](installation.md)
- [CLI Usage](cli.md)
- [Python API](api.md)
- [Examples](examples.md)
- [Environment API Stability](environment_api_stability.md)
- [Environment Transport Failure Behavior](environment_transport_failure_behavior.md)

## Canonical terminology

For core concepts, use the canonical glossary definitions and avoid redefining terms in-place:
`/docs/canonical-glossary.md`.

## What Atropos does

Atropos is built for practical deployment decisions with an ROI-estimation-first identity:

- Estimate memory, throughput, energy, and cost outcomes from optimization choices.
- Evaluate break-even timelines for pruning and related strategy mixes.
- Compare candidate strategies with consistent assumptions.
- Move from estimation to execution with pipeline/validation/telemetry/A-B-test modules.

## Example

```bash
atropos-llm preset medium-coder --strategy structured_pruning --report text
```
