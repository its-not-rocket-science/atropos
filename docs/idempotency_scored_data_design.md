# Idempotency Design: `/scored_data` and `/scored_data_list`

## Context

Distributed RL training often retries API writes during transient failures (timeouts, connection resets, worker restarts). Without request-scoped idempotency, duplicate deliveries can silently append duplicate scored samples and corrupt downstream training batches.

## Goals

- Make `POST /scored_data` safe under aggressive retries.
- Make `POST /scored_data_list` (batch ingestion) safe under aggressive retries.
- Keep `GET /scored_data_list` read-only and deterministic for a given storage snapshot.
- Provide a concrete deduplication contract that clients can rely on.
- Minimize silent data corruption risk during retry storms.

## API contract

### `POST /scored_data`

- Requires `X-Request-ID` **or** `X-Idempotency-Key` header.
- Accepts JSON payload:
  - `environment_id` (string, default `"default"`)
  - `group_id` (optional string; if omitted server derives a deterministic hash)
  - `records` (list of scored record objects)
- Deduplicates at **request level** keyed by request UUID / idempotency key.
- Also deduplicates accepted `group_id`s to avoid re-enqueueing groups after partial success.
- Response includes:
  - `request_id`
  - `accepted_count`
  - `accepted_groups`
  - `failed_groups`
  - `status` (`completed` or `partial_failed`)
  - `deduplicated`

Behavior:
- First request ID seen: data is ingested, `deduplicated=false`.
- Retried same request ID:
  - no additional ingestion when request is already `completed`.
  - resume semantics for `partial_failed` requests (already accepted groups are skipped).

### `POST /scored_data_list`

- Requires `X-Request-ID` **or** `X-Idempotency-Key` header.
- Accepts JSON payload:
  - `groups`: list of per-group payloads with `environment_id`, optional `group_id`, and `records`.
- Uses the same request-status and group-level dedupe behavior as `POST /scored_data`.

### `GET /scored_data_list`

- Query params:
  - `environment_id` (required)
  - `limit` (default `100`, bounded to `<=1000`)
- Returns data snapshot for the environment with no mutation.

## Deduplication strategy

### In-memory backend

- Maintain:
  - request status map (`processing` / `partial_failed` / `completed`).
  - accepted group keys (`environment_id:group_id`) to prevent duplicate enqueue.
  - append-only per-environment scored record list.
- On ingest:
  - if request status is `completed` → short-circuit (deduplicated).
  - else process each group independently and only append first-seen groups.
  - mark request `partial_failed` if any group fails validation.

### Redis backend

- Persist request status keys and group-claim keys with TTL.
- Use `SET key value NX EX <ttl>` for atomic first-writer-wins on each group key.
- Treat already claimed groups as deduplicated skips.
- Reuse existing idempotency TTL window to bound memory footprint.

## Failure and retry semantics

- Client retries with same request key are safe and non-amplifying.
- Network failure after server commit is safe: retry returns deduplicated response.
- Distinct request IDs remain distinct writes, but previously accepted groups are skipped.

## Client guidance

- Generate UUID request IDs per logical ingest attempt and reuse the same ID across retries.
- Prefer stable `group_id` per logical scored-data group to get dedupe across request IDs.
- Never generate new request IDs for retry attempts of the same logical write.

## Validation and tests

Added tests cover:
- Missing request idempotency headers are rejected.
- Retry storm (50 repeated identical writes) ingests data exactly once.
- Distinct request IDs ingest independently.
- Batch ingestion dedupe for repeated submissions.
- Partial failures and retry resume behavior (already accepted groups are not re-enqueued).
- Existing runtime idempotency behavior for `/jobs` remains intact.
