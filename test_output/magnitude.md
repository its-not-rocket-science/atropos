# Pruning Framework Comparison

Generated: 2026-03-17T18:44:31.038093
Duration: 11.2s

## Configuration

- **models**: gpt2
- **frameworks**: magnitude
- **target_sparsity**: 0.05
- **nsamples**: 1
- **seed**: 0
- **device**: cpu

## Summary

- **Total tests**: 1
- **Successful**: 1
- **Failed**: 0

## Results

| Model | Framework | Target Sparsity | Achieved Sparsity | Original Params | Pruned Params | Time (s) | Status |
|-------|-----------|-----------------|-------------------|-----------------|---------------|----------|--------|
| gpt2 | magnitude | 5.0% | 1.6% | 124,439,808 | 122,509,939 | 5.3 | ✅ |

## Framework Comparison

Average metrics across all successful tests:

| Framework | Avg Sparsity Achieved | Avg Time (s) | Success Rate |
|-----------|----------------------|--------------|--------------|
| magnitude | 1.55% | 5.3 | 100.0% |

## Analysis

### Key Findings

1. **Sparsity Accuracy**: How close each framework gets to the target sparsity.
2. **Pruning Time**: Computational cost of each pruning method.
3. **Success Rate**: Reliability across different model architectures.

### Recommendations

- **Magnitude pruning**: Fastest but may not achieve exact target sparsity due to global pruning.
- **Wanda**: Good balance of accuracy and speed, uses first-order gradient information.
- **SparseGPT**: Most accurate but computationally intensive, uses Hessian approximation.

### Notes

- All frameworks use unstructured pruning (weights set to zero).
- Memory savings require sparse tensor formats or structured pruning.
- Results may vary with different models, sparsity levels, and calibration data.