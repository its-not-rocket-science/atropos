# Quality/Speed Trade-off Analysis

Generated: 2026-03-18T15:43:58.969907
Duration: 85.6s

## Summary

- **Total tests:** 3
- **Successful:** 3
- **Failed:** 0

## Configuration

- **models:** ['gpt2']
- **frameworks:** ['magnitude', 'wanda-patched', 'sparsegpt-patched']
- **pruned_dir:** test_data\framework_comparison
- **cache_dir:** test_data\models
- **device:** cpu

## Results

| Model | Framework | Sparsity | Perplexity Change% | Score Change% | Speedup% | Q/S Ratio | Status |
|-------|-----------|----------|-------------------|--------------|----------|-----------|--------|
| gpt2 | magnitude | 0.0% | -0.4% | -20.0% | -2.8% | N/A | OK |
| gpt2 | wanda-patched | 9.9% | -0.9% | -16.7% | +2.0% | -0.45 | OK |
| gpt2 | sparsegpt-patched | 10.0% | +4.0% | +0.0% | +2.1% | +1.96 | OK |

## Trade-off Analysis

### Key Metrics

- **Perplexity Change%**: Positive = degradation, Negative = improvement
- **Score Change%**: Positive = improvement, Negative = degradation
- **Speedup%**: Positive = faster inference, Negative = slower
- **Q/S Ratio**: Quality degradation per unit speed improvement
  (lower/negative is better)

### Recommendations

Based on Q/S Ratio:
- **Q/S Ratio < -1**: Good trade-off
  (significant speed improvement with minimal quality loss)
- **-1 < Q/S Ratio < 0**: Acceptable trade-off
- **Q/S Ratio > 0**: Poor trade-off (quality degrades faster than speed improves)

## Notes

- Quality metrics measured on code completion prompts
- Speed measured as inference time for 50-token generation
- Results may vary with different prompts and generation parameters
- Always validate with task-specific benchmarks before production deployment