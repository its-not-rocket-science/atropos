# Pruning Exercise Results

Generated: 2026-03-13T00:10:41.473179
Duration: 1772.1s

## Summary

- **Total models pruned:** 10
- **Successful:** 8
- **Failed:** 2

## Pruned Models

| Model | Strategy | Target Sparsity | Achieved Sparsity | Original Params | Pruned Params | Time |
|-------|----------|-----------------|-------------------|-----------------|---------------|------|
| gpt2 | mild_pruning | 10% | 3.1% | 124,439,808 | 120,580,070 | 5.4s |
| gpt2 | structured_pruning | 22% | 6.8% | 124,439,808 | 115,948,385 | 8.3s |
| gpt2-medium | mild_pruning | 10% | 1.5% | 354,823,168 | 349,676,851 | 6.6s |
| gpt2-medium | structured_pruning | 22% | 3.2% | 354,823,168 | 343,501,271 | 10.4s |
| gpt2-xl | mild_pruning | 10% | 0.5% | 1,557,611,200 | 1,549,570,080 | 11.0s |
| gpt2-xl | structured_pruning | 22% | 1.1% | 1,557,611,200 | 1,539,920,736 | 16.3s |
| facebook/opt-1.3b | mild_pruning | 10% | 10.0% | 1,315,758,080 | 1,184,664,356 | 487.4s |
| facebook/opt-1.3b | structured_pruning | 22% | 21.9% | 1,315,758,080 | 1,027,354,363 | 346.6s |

## Comparison with Projections

Compare these actual results with `test_data/projections.md`
to validate Atropos projection accuracy.
