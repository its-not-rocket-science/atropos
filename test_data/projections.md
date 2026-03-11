# Atropos Pruning Projections Report

Generated: 2026-03-10T22:49:19.746086
Duration: 90.7s

## Summary

- **Total projections:** 10
- **Successful:** 10
- **Failed:** 0

## Projections by Model

| Model | Params | Strategy | Memory Red. | Throughput | Annual Savings | Break-even |
|-------|--------|----------|-------------|------------|----------------|------------|
| gpt2 | 0.124B | mild_pruning | 10.0% | +8.0% | $414 | 348.1mo |
| gpt2 | 0.124B | structured_pruning | 22.0% | +20.0% | $910 | 158.3mo |
| gpt2-medium | 0.355B | mild_pruning | 10.0% | +8.0% | $414 | 348.1mo |
| gpt2-medium | 0.355B | structured_pruning | 22.0% | +20.0% | $910 | 158.3mo |
| gpt2-xl | 1.5B | mild_pruning | 10.0% | +8.0% | $1,707 | 189.8mo |
| gpt2-xl | 1.5B | structured_pruning | 22.0% | +20.0% | $3,752 | 86.4mo |
| facebook/opt-1.3b | 1.3B | mild_pruning | 10.0% | +8.0% | $1,707 | 189.8mo |
| facebook/opt-1.3b | 1.3B | structured_pruning | 22.0% | +20.0% | $3,752 | 86.4mo |
| EleutherAI/pythia-2.8b | 2.8B | mild_pruning | 10.0% | +8.0% | $1,707 | 189.8mo |
| EleutherAI/pythia-2.8b | 2.8B | structured_pruning | 22.0% | +20.0% | $3,752 | 86.4mo |

## Notes

These are *projected* savings before actual pruning is applied.
Actual results will be compared against these projections.
