# Combined Pruning + Quantization Study

Generated: 2026-03-19T14:34:37.798487
Duration: 96.0s

## Summary

- **Total experiments:** 3
- **Successful:** 3
- **Failed:** 0

## Results

| Model | Combination | Strategy | Memory Reduction | Speedup | Perplexity Change | Score Change | Status |
|-------|-------------|----------|------------------|---------|-------------------|--------------|--------|
| gpt2 | pruned_only | mild_pruning | 0.0% | -8.5% | +0.7% | +33.3% | OK |
| gpt2 | quantized_only | N/A | 0.0% | -124.6% | +7306.9% | -42.9% | OK |
| gpt2 | combined | mild_pruning | 0.0% | -625.6% | +7315.7% | -66.7% | OK |

## Cumulative Effects Analysis

Comparing individual vs combined optimizations:

| Model | Optimization | Memory Reduction | Speedup | Perplexity Change |
|-------|--------------|------------------|---------|-------------------|
| gpt2 | Pruning only (mild_pruning) | 0.0% | -8.5% | +0.7% |
| gpt2 | Quantization only | 0.0% | -124.6% | +7306.9% |
| gpt2 | Combined (mild_pruning+quantization) | 0.0% | -625.6% | +7315.7% |

## Insights

1. **Cumulative savings**: Combined optimization should provide multiplicative memory reduction and additive speed improvements.
2. **Quality impact**: Check if combined optimization amplifies quality degradation beyond individual optimizations.
3. **ROI implications**: Combined optimizations may accelerate break-even timelines in Atropos projections.
