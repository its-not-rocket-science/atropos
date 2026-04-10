# Atropos Documentation
> Terminology follows the canonical glossary: `/docs/canonical-glossary.md`.

**Atropos** is an **ROI estimation + optimization toolkit** for coding-LLM deployments.

Secondary modules support production workflows around the ROI core:
- pipeline orchestration
- validation
- telemetry ingestion
- A/B testing

> Use `atropos` for Python imports and `atropos-llm` for the CLI.

## Quick Links

- [Canonical Glossary](canonical-glossary.md)
- [Documentation Rewrite Plan](doc-rewrite-plan.md)
- [Product Strategy](product-strategy.md)
- [Async RL Reproducibility Architecture](async_rl_reproducibility_architecture.md)

- [Installation](installation.md)
- [CLI Usage](cli.md)
- [Python API](api.md)
- [Examples](examples.md)
- [Environment API Stability](environment_api_stability.md)

## Canonical terminology

For core concepts, use the canonical glossary definitions and avoid redefining terms in-place:
`/docs/canonical-glossary.md`.

## What Atropos Does

Named after the Fate who cuts the thread, Atropos is built for practical deployment decisions with an ROI-estimation-first identity:

- Estimate memory, throughput, energy, and cost outcomes from optimization choices.
- Evaluate break-even timelines for pruning and related strategy mixes.
- Compare candidate strategies with consistent assumptions.
- Move from estimation to execution with pipeline/validation/telemetry/A-B-test modules.

## Example

```bash
atropos-llm preset medium-coder --strategy structured_pruning --report text
```

Output:
```
Scenario: medium-coder
Strategy: structured_pruning

Model memory:      14.00 GB -> 10.92 GB
Throughput:        40.00 tok/s -> 48.00 tok/s
Annual savings:    $12,500.50
Break-even:        21.60 months
Quality risk:      medium
```
