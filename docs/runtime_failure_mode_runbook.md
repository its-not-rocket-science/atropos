# Atropos runtime failure-mode matrix and operator runbook

## Failure-mode matrix

| Failure mode | Detection signal | API behavior | Data integrity behavior | Operator action |
|---|---|---|---|---|
| API restart (durable store) | Process restart + `/health/ready` recovered_items > 0 | Service becomes ready after startup + dependency health checks | Job idempotency and scored-ingestion dedupe state are preserved | Validate `/health/ready`, then resume traffic |
| API restart (in-memory store) | Process restart | Service returns ready, but state is empty | Queue + dedupe state are lost (expected for non-durable mode) | Use only for dev/test; use Redis in production |
| Worker restart/retry | Duplicate request IDs from workers | First attempt accepted, retries deduplicated | Exactly-once-per-request-id semantics for accepted groups | Ensure workers persist/reuse request IDs across retries |
| Store outage | `/health/dependencies` degraded; write endpoints return HTTP 503 | Writes/reads fail explicitly with `Runtime store unavailable during <endpoint>` | No silent success; retries required after dependency recovery | Restore dependency, verify readiness=200, replay failed requests with same IDs |
| Duplicate environment registration | Repeated `POST /environments` same id | 200 with `created=false` for duplicates | No duplicate environment records | None required |
| Partial ingestion | Ingestion response `status=partial_failed`, `failed_groups>0` | API returns 200 + partial status | Accepted groups committed, failed groups recoverable by retry | Retry failed request with same request id and corrected failed groups |
| Batch delivery interruption | Group status `interrupted`; partial_failed result | API returns partial result for request | Interrupted group claim is released; later retry can complete group | Replay same request id after dependency/network issue resolved |

## Operational procedures

### 1) Triage checklist

1. Check `GET /health/live` (process up).
2. Check `GET /health/dependencies` (dependency health).
3. Check `GET /health/ready` (control-plane + dependency + shutdown status).
4. For ingestion incidents, inspect request responses for:
   - `status`
   - `failed_groups`
   - `duplicate_groups`
   - `deduplicated`

### 2) Recovery playbooks

#### Store outage

1. Mitigate dependency issue (Redis/network).
2. Wait for `/health/dependencies` = 200 and `/health/ready` = 200.
3. Replay failed writes with **same request IDs**.
4. Confirm dedupe via response fields and data count checks.

#### Batch interruption

1. Identify impacted request id/group id.
2. Retry ingestion with the same request id after transport/store recovery.
3. Verify group lifecycle reaches `acknowledged` and request `status=completed`.

#### Restart event

1. Validate recovered state using `/health/ready` (`recovered_items`).
2. Re-run idempotent retries from workers/clients if needed.
3. Confirm no duplicate records by spot-checking `/scored_data_list` counts.

## SLO-oriented alerts (recommended)

- Alert on `/health/dependencies` non-200 for > 2 minutes.
- Alert on sustained growth of duplicate retries without corresponding success.
- Alert on repeated `partial_failed` ingestion responses per environment.
