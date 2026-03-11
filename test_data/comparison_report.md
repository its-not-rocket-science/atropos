# Atropos Projections vs Actual Pruning Results

Generated: 2026-03-11T01:56:39.263909

## Summary

- **total_comparisons**: 8
- **successful_pruning**: 8
- **avg_projected_memory_reduction_pct**: 16.0
- **avg_achieved_memory_reduction_pct**: 3.01
- **avg_variance_pct**: -81.21
- **accuracy_assessment**: poor

## Detailed Comparison

| Model | Strategy | Projected Mem Red. | Achieved Sparsity | Est. Mem Red. | Variance | Status |
|-------|----------|-------------------|-------------------|---------------|----------|--------|
| gpt2 | mild_pruning | 10.0% | 3.1% | 1.6% | -84.5% | WARN |
| gpt2 | structured_pruning | 22.0% | 6.8% | 3.4% | -84.5% | WARN |
| gpt2-medium | mild_pruning | 10.0% | 1.5% | 0.7% | -92.7% | WARN |
| gpt2-medium | structured_pruning | 22.0% | 3.2% | 1.6% | -92.7% | WARN |
| gpt2-xl | mild_pruning | 10.0% | 0.5% | 0.3% | -97.4% | WARN |
| gpt2-xl | structured_pruning | 22.0% | 1.1% | 0.6% | -97.4% | WARN |
| facebook/opt-1.3b | mild_pruning | 10.0% | 10.0% | 5.0% | -50.2% | WARN |
| facebook/opt-1.3b | structured_pruning | 22.0% | 21.9% | 11.0% | -50.2% | WARN |

## Analysis

### Key Findings

1. **Sparsity vs Memory Reduction**: Unstructured pruning achieves sparsity but
   does not reduce memory footprint unless sparse tensor formats are used.
   Atropos assumes structured pruning which removes entire channels/heads.

2. **Model Architecture Matters**: OPT models achieved target sparsity better
   than GPT models, likely due to different layer structures.

3. **Projection Accuracy**: Memory variance indicates Atropos projections
   assume structured pruning with actual parameter removal.

## Recommendations

1. Update Atropos strategies to distinguish between:
   - Structured pruning (actual memory savings)
   - Unstructured pruning (sparsity only, needs sparse inference)

2. For actual memory savings, use structured pruning frameworks like LLM-Pruner
   or magnitude pruning with channel removal.
