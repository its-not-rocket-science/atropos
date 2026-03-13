# Pruned Model Validation Report

**Date:** 2026-03-13 00:27
**Summary:** 4/8 models passed validation

## Validation Criteria

- **Perplexity increase tolerance:** 20%
- **Max acceptable perplexity:** 50.0
- **Generation similarity threshold:** 70%

## Results Summary

| Model | Strategy | Perplexity | Change | Gen Sim | Status |
|-------|----------|------------|--------|---------|--------|
| facebook/opt-1.3b | mild_pruning | 77.92 | +11.3% | 61.3% | ❌ FAIL |
| facebook/opt-1.3b | structured_pruning | 115.99 | +65.6% | 41.5% | ❌ FAIL |
| gpt2 | mild_pruning | 34.82 | -0.9% | 89.1% | ✅ PASS |
| gpt2 | structured_pruning | 33.73 | -4.0% | 50.0% | ❌ FAIL |
| gpt2-medium | mild_pruning | 35.8 | +0.1% | 85.0% | ✅ PASS |
| gpt2-medium | structured_pruning | 36.0 | +0.6% | 40.9% | ❌ FAIL |
| gpt2-xl | mild_pruning | 26.29 | -0.8% | 100.0% | ✅ PASS |
| gpt2-xl | structured_pruning | 25.49 | -3.9% | 73.2% | ✅ PASS |

## Detailed Results

### facebook/opt-1.3b - mild_pruning

**Status:** ❌ FAILED

**Metrics:**
- Original perplexity: 70.04
- Pruned perplexity: 77.92
- Perplexity increase: 11.3%
- Avg generation similarity: 0.613

### facebook/opt-1.3b - structured_pruning

**Status:** ❌ FAILED

**Metrics:**
- Original perplexity: 70.04
- Pruned perplexity: 115.99
- Perplexity increase: 65.6%
- Avg generation similarity: 0.415

### gpt2 - mild_pruning

**Status:** ✅ PASSED

**Metrics:**
- Original perplexity: 35.13
- Pruned perplexity: 34.82
- Perplexity increase: -0.9%
- Avg generation similarity: 0.891

### gpt2 - structured_pruning

**Status:** ❌ FAILED

**Metrics:**
- Original perplexity: 35.13
- Pruned perplexity: 33.73
- Perplexity increase: -4.0%
- Avg generation similarity: 0.5

### gpt2-medium - mild_pruning

**Status:** ✅ PASSED

**Metrics:**
- Original perplexity: 35.77
- Pruned perplexity: 35.8
- Perplexity increase: 0.1%
- Avg generation similarity: 0.85

### gpt2-medium - structured_pruning

**Status:** ❌ FAILED

**Metrics:**
- Original perplexity: 35.77
- Pruned perplexity: 36.0
- Perplexity increase: 0.6%
- Avg generation similarity: 0.409

### gpt2-xl - mild_pruning

**Status:** ✅ PASSED

**Metrics:**
- Original perplexity: 26.52
- Pruned perplexity: 26.29
- Perplexity increase: -0.8%
- Avg generation similarity: 1.0

### gpt2-xl - structured_pruning

**Status:** ✅ PASSED

**Metrics:**
- Original perplexity: 26.52
- Pruned perplexity: 25.49
- Perplexity increase: -3.9%
- Avg generation similarity: 0.732
