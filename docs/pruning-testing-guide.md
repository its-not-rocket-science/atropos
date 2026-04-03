# Pruning Integration Testing Guide

## 1. Setup

```bash
atropos-llm setup-pruning
```

For dependency repair:

```bash
atropos-llm setup-pruning --fix
```

## 2. Framework smoke tests

```bash
atropos-llm test-pruning --framework wanda
atropos-llm test-pruning --framework sparsegpt
atropos-llm test-pruning --framework llm-pruner
```

## 3. Container build smoke test

```bash
docker compose -f docker-compose.yml build
```

(or `podman compose -f docker-compose.yml build`)

## 4. Tiny model integration test in CI

Use tiny models to verify fallback behavior:

- `sshleifer/tiny-gpt2`
- `TinyLlama/TinyLlama-1.1B-Chat-v1.0`

## 5. Performance regression tracking

Track per-framework runtime from `PruningResult.duration_seconds` and store it in CI artifacts.
