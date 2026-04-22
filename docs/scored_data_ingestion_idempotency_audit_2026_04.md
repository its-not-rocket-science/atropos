# Scored-Data Ingestion Idempotency Audit (April 2026)

## Scope

This audit traces scored-data flow from environment submission through trainer batch consumption and documents the idempotency contract enforced at each step.

## Path trace: environment submission → batch consumption

1. **Environment runtime submission**
   - Environment workers call the runtime API with `X-Request-ID` (or `X-Idempotency-Key`) on:
     - `POST /scored_data`
     - `POST /scored_data_list`
2. **HTTP ingestion layer**
   - Request headers are validated, and each payload is coerced into `ScoredDataGroup`.
   - Missing idempotency header returns `400`.
3. **Store ingestion (in-memory or Redis)**
   - Request status (`processing`, `partial_failed`, `completed`) is tracked by request ID.
   - Group dedupe key is `environment_id:group_id`.
   - Duplicate groups are skipped and never appended to scored record storage.
4. **Batch consumption**
   - Trainers consume via `GET /scored_data_list?environment_id=...&limit=...`.
   - Only previously accepted unique groups contribute records.

## Idempotency guarantees

- **Request-level idempotency:** replaying a completed request ID is a no-op.
- **Group-level idempotency:** replayed or duplicated groups are skipped even across different request IDs.
- **Partial retry safety:** retries for `partial_failed` requests accept only groups that were not previously accepted.
- **No duplicate queue entries:** deduped groups do not create additional queued/buffered/batched group transitions and do not append duplicate records.

## Visibility and metrics

The runtime now reports duplicate behavior through:

- `atropos_duplicate_ingestion_rejections_total{env,endpoint}`
  - count of fully deduplicated replay requests.
- `atropos_duplicate_ingestion_groups_total{env,endpoint}`
  - count of duplicate groups skipped during ingestion.
- API response fields:
  - `deduplicated` (request-level replay signal)
  - `duplicate_groups` (group-level duplicate skip count)

## Client expectations

- Generate a stable request ID per logical ingest operation.
- Reuse the same request ID on timeout/network retries.
- Keep `group_id` stable for each logical group so cross-request dedupe remains effective.
- For partial retries, resend the original request ID with corrected failed groups; previously accepted groups will be skipped safely.

## Validated scenarios

- Exact retry after timeout.
- Duplicated groups inside `scored_data_list`.
- Partial retry after network failure / partial acceptance.
- Duplicate request replay after successful prior ingestion.
