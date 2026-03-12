# Pruning Exercise Case Study

**Generated:** 2026-03-12T19:27:17.940292

## Executive Summary

This case study validates Atropos ROI projections against actual pruning results
using magnitude-based unstructured pruning on 5 candidate models.

### Key Metrics

- **Total Models Analyzed:** 8
- **Avg Projected Annual Savings:** 1695.75
- **Avg Actual Annual Savings:** 790.46
- **Savings Variance Pct:** -53.4
- **Viable Pruning Scenarios:** 1
- **Non Viable Scenarios:** 6

## Detailed Analysis

| Model | Strategy | Proj Savings | Actual Savings | Variance | Break-even | Quality Impact | Recommendation |
|-------|----------|--------------|----------------|----------|------------|----------------|----------------|
| gpt2 | mild_pruning | $414 | $128 | -69.0% | 468mo | -64.0% | Not recommended - break-even exceeds 10 years |
| gpt2 | structured_pruning | $910 | $282 | -69.0% | 213mo | -92.0% | Not recommended - break-even exceeds 10 years |
| gpt2-medium | mild_pruning | $414 | $60 | -85.5% | 1000mo | +50.0% | Not recommended - excessive quality degradation |
| gpt2-medium | structured_pruning | $910 | $132 | -85.5% | 455mo | +86.7% | Not recommended - excessive quality degradation |
| gpt2-xl | mild_pruning | $1,707 | $88 | -94.8% | 681mo | N/A | Not recommended - break-even exceeds 10 years |
| gpt2-xl | structured_pruning | $3,752 | $194 | -94.8% | 310mo | N/A | Not recommended - break-even exceeds 10 years |
| facebook/opt-1.3b | mild_pruning | $1,707 | $1,701 | -0.4% | 35mo | N/A | Conditionally recommended - break-even within 5 years |
| facebook/opt-1.3b | structured_pruning | $3,752 | $3,738 | -0.4% | 16mo | N/A | Recommended - break-even within 2 years |

## Key Findings

1. Average projected savings: $1,696/year
1. Average actual savings: $790/year
1. Savings variance: -53.4%
1. Viable scenarios: 1/8
1. Unstructured pruning (magnitude-based) achieves lower memory savings than projected
1. Quality degradation varies significantly by model architecture
1. OPT models achieve target sparsity better than GPT models

## Recommendations

1. Use structured pruning (LLM-Pruner) for actual memory savings vs unstructured
1. Limit pruning to <20% sparsity for models requiring high quality
1. Consider quantization + pruning combination for better ROI
1. Test thoroughly on target workload before deployment
1. Update Atropos projections to distinguish structured vs unstructured pruning

## Methodology

### Models Tested
- gpt2 (124M parameters)
- gpt2-medium (355M parameters)
- gpt2-xl (1.5B parameters)
- facebook/opt-1.3b (1.3B parameters)

### Pruning Strategy
- Method: PyTorch magnitude-based unstructured pruning
- Target sparsity: 10% (mild) and 22% (structured)
- Quality evaluation: Perplexity + code completion score

### ROI Calculation
- Savings proportional to achieved sparsity vs target
- One-time project cost assumption: $5,000
- Break-even = (Project Cost / Annual Savings) x 12 months

## Conclusion

The pruning exercise reveals significant variance between Atropos projections
and actual results when using unstructured magnitude pruning:

- **Savings variance:** -53.4%
- **Root cause:** Unstructured pruning doesn't reduce memory without sparse inference
- **Recommendation:** Use structured pruning (LLM-Pruner) for production deployments

This validates the importance of testing actual pruning methods against projections
before making deployment decisions.
