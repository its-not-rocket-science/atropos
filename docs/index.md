# Atropos Documentation

**Atropos** estimates whether pruning and related optimizations for a coding LLM are worth the engineering effort.

## Quick Links

- [Installation](installation.md)
- [CLI Usage](cli.md)
- [Python API](api.md)
- [Examples](examples.md)

## What Atropos Does

Named after the Fate who cuts the thread, Atropos is built for practical deployment questions:

- How much memory, throughput, energy, and cost improvement is realistic?
- When does a pruning project break even?
- How do pruning-only and pruning-plus-quantization compare?
- Which deployment scenarios justify optimization work?

## Example

```bash
atropos-llm preset medium-coder --strategy structured_pruning --report text
```

Output:
```
Scenario: medium-coder
Strategy: structured_pruning

Model memory:      14.00 GB -> 10.92 GB
Throughput:        40.00 tok/s -> 48.00 tok/s
Annual savings:    $12,500.50
Break-even:        21.60 months
Quality risk:      medium
```
