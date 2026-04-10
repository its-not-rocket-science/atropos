# Idempotency Design: `/scored_data` and `/scored_data_list`

## Context

Distributed RL training often retries API writes during transient failures (timeouts, connection resets, worker restarts). Without request-scoped idempotency, duplicate deliveries can silently append duplicate scored samples and corrupt downstream training batches.

## Goals

- Make `POST /scored_data` safe under aggressive retries.
- Keep `GET /scored_data_list` read-only and deterministic for a given storage snapshot.
- Provide a concrete deduplication contract that clients can rely on.
- Minimize silent data corruption risk during retry storms.

## API contract

### `POST /scored_data`

- Requires `X-Request-ID` header.
- Accepts JSON payload:
  - `environment_id` (string, default `"default"`)
  - `records` (list of scored record objects)
- Deduplicates at **request level** keyed by `X-Request-ID`.
- Response includes:
  - `request_id`
  - `accepted_count`
  - `deduplicated`

Behavior:
- First request ID seen: data is ingested, `deduplicated=false`.
- Retried same request ID: no additional ingestion, `deduplicated=true`, `accepted_count=0`.

### `GET /scored_data_list`

- Query params:
  - `environment_id` (required)
  - `limit` (default `100`, bounded to `<=1000`)
- Returns data snapshot for the environment with no mutation.

## Deduplication strategy

### In-memory backend

- Maintain:
  - `set` of seen request IDs.
  - append-only per-environment scored record list.
- On ingest:
  - if request ID already seen → short-circuit (deduplicated).
  - else mark seen and append records once.

### Redis backend

- Use `SET key value NX EX <ttl>` for atomic first-writer-wins on request ID keys.
- If `SET ... NX` fails, treat request as duplicate.
- If first writer wins, append records to environment list.
- Reuse existing idempotency TTL window to bound memory footprint.

## Failure and retry semantics

- Client retries with same `X-Request-ID` are safe and non-amplifying.
- Network failure after server commit is safe: retry returns deduplicated response.
- Distinct request IDs are treated as distinct writes (intentional replays remain possible).

## Client guidance

- Generate UUID request IDs per logical ingest attempt and reuse the same ID across retries.
- Never generate new request IDs for retry attempts of the same logical write.

## Validation and tests

Added tests cover:
- Missing `X-Request-ID` rejected.
- Retry storm (50 repeated identical writes) ingests data exactly once.
- Distinct request IDs ingest independently.
- Existing runtime idempotency behavior for `/jobs` remains intact.
