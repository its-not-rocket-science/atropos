# CLI Usage

## Commands

### `list-presets`

Show available built-in scenarios and strategies.

```bash
atropos-llm list-presets
```

### `preset`

Run analysis with a built-in scenario.

```bash
atropos-llm preset medium-coder --strategy structured_pruning
```

Options:
- `--strategy`: Strategy to use (default: structured_pruning)
- `--with-quantization`: Add quantization effects
- `--report`: Output format (text, json, markdown, html)

### `scenario`

Run analysis from a YAML scenario file.

```bash
atropos-llm scenario my_scenario.yaml --report markdown
```

### `compare`

Compare multiple strategies.

```bash
atropos-llm compare medium-coder \
  --strategies mild_pruning structured_pruning \
  --format markdown
```

Options:
- `--format`: text, markdown, or json
- `--sort-by`: Sort by savings, breakeven, or risk
- `--ascending`: Sort in ascending order
- `--output`: Save to file

### `batch`

Process multiple scenario files.

```bash
atropos-llm batch scenarios/ \
  --strategies mild_pruning structured_pruning \
  --output results.csv
```

### `sensitivity`

Run sensitivity analysis on a parameter.

```bash
atropos-llm sensitivity medium-coder \
  --strategy structured_pruning \
  --param memory_reduction_fraction \
  --format json \
  --output sensitivity.json
```

### `csv-to-markdown`

Convert batch CSV results to markdown report.

```bash
atropos-llm csv-to-markdown results.csv --output report.md
```

## Cloud pricing commands

```bash
atropos-llm cloud-pricing list-providers
atropos-llm cloud-pricing estimate --scenario scenario.yaml --provider aws
atropos-llm cloud-pricing compare --scenario scenario.yaml --providers aws,azure,lambda-labs
```

Use `--fetch-live-pricing` with `estimate` or `compare` to refresh provider catalogs before estimation.
