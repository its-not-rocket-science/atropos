# Batch Mode Error Handling Guide

Atropos batch mode now supports resilient processing with per-scenario isolation and partial success reporting.

## Key behaviors

- **Per-scenario isolation**: each scenario/strategy row is executed in a subprocess.
- **Timeout enforcement**: long-running rows are terminated after `--scenario-timeout-seconds` (default `600`).
- **Retry policy**: recoverable errors are retried with exponential backoff (default `--retry-attempts 3`).
- **Partial success**: successful rows are persisted even when later rows fail.
- **Checkpointing**: progress is saved every `--checkpoint-every` rows.
- **Resume**: use `--resume <existing.csv>` to skip previously completed rows.

## CLI options

```bash
atropos-llm batch examples/ \
  --strategies mild_pruning structured_pruning \
  --output results.csv \
  --error-log errors.json \
  --retry-attempts 3 \
  --scenario-timeout-seconds 600 \
  --checkpoint-every 5 \
  --max-errors 5
```

### Fail fast mode

```bash
atropos-llm batch examples/ --strategies mild_pruning --output out.csv --fail-fast
```

### Resume mode

```bash
atropos-llm batch examples/ --strategies mild_pruning --output out.csv --resume out.csv
```

## CSV output additions

Batch CSV output includes three additional columns:

- `status`: `success` or `failed`
- `error_message`: root cause for failed rows
- `retry_count`: retries consumed before final result

## Error categories

Errors are categorized to make triage easier:

- `timeout`
- `network`
- `resource`
- `config`
- `recoverable`
- `fatal`
- `unknown`

## Methodology and SLA notes

- **Academic workflows**: include `errors.json` in appendices to document observed failure modes.
- **Commercial workflows**: compute success-rate/SLA from the batch summary (`successful / total`).
- **User trust**: failed rows include concise root-cause messages and preserve successful results.
