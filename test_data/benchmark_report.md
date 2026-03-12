# Model Quality Benchmark Results

Generated: 2026-03-11T10:38:01.499821
Duration: 512.6s

## Summary

- **Total models:** 9
- **Successful:** 9
- **Failed:** 0

## Results

| Model | Strategy | Avg Perplexity | Completion Score | Inference (ms) | Status |
|-------|----------|----------------|------------------|----------------|--------|
| gpt2 | baseline | 105.6 | 0.42 | 1920.0 | OK |
| gpt2 | mild_pruning | 105.2 | 0.68 | 1936.8 | OK |
| gpt2 | structured_pruning | 105.7 | 0.80 | 1898.0 | OK |
| gpt2-medium | baseline | 95.1 | 0.50 | 4978.3 | OK |
| gpt2-medium | mild_pruning | 93.5 | 0.25 | 4815.7 | OK |
| gpt2-medium | structured_pruning | 96.4 | 0.07 | 4883.3 | OK |
| facebook/opt-1.3b | baseline | 76.7 | N/A | 17246.3 | OK |
| facebook/opt-1.3b | mild_pruning | 83.4 | 0.35 | 18541.3 | OK |
| facebook/opt-1.3b | structured_pruning | 99.2 | 0.05 | 15631.4 | OK |

## Quality Degradation Analysis

Comparison of pruned models vs baseline:

| Model | Strategy | Perplexity Change | Score Change |
|-------|----------|-------------------|--------------|
| gpt2 | mild_pruning | -0.4% | +26.7% |
| gpt2 | structured_pruning | +0.0% | +38.3% |
| gpt2-medium | mild_pruning | -1.7% | -25.0% |
| gpt2-medium | structured_pruning | +1.3% | -43.3% |
| facebook/opt-1.3b | mild_pruning | +8.7% | +0.0% |
| facebook/opt-1.3b | structured_pruning | +29.2% | +0.0% |

## Notes

- Lower perplexity is better (model is more confident)
- Completion score measures keyword presence in generated code
- < 10% perplexity increase and < 5% score drop considered acceptable
