# Pruning Framework Comparison

Generated: 2026-03-18T01:29:36.293714
Duration: 529.1s

## Configuration

- **models**: gpt2
- **frameworks**: wanda-patched, sparsegpt-patched
- **target_sparsity**: 0.1
- **nsamples**: 1
- **seed**: 0
- **device**: cpu

## Summary

- **Total tests**: 2
- **Successful**: 2
- **Failed**: 0

## Results

| Model | Framework | Target Sparsity | Achieved Sparsity | Original Params | Pruned Params | Time (s) | Status |
|-------|-----------|-----------------|-------------------|-----------------|---------------|----------|--------|
| gpt2 | wanda-patched | 10.0% | 9.9% | 124,439,808 | 124,439,808 | 152.6 | ✅ |
| gpt2 | sparsegpt-patched | 10.0% | 10.0% | 124,439,808 | 124,439,808 | 376.0 | ✅ |

## Framework Comparison

Average metrics across all successful tests:

| Framework | Avg Sparsity Achieved | Avg Time (s) | Success Rate |
|-----------|----------------------|--------------|--------------|
| wanda-patched | 9.93% | 152.6 | 100.0% |
| sparsegpt-patched | 10.00% | 376.0 | 100.0% |

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