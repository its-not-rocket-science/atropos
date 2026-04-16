# Durable Queued-Group State Machine

This document defines the durable lifecycle for scored-data ingestion groups in the runtime store.

## Lifecycle states

A queued group MUST progress through these ordered states:

1. `accepted`
2. `buffered`
3. `batched`
4. `delivered`
5. `acknowledged`

## Transition rules

- `accepted -> buffered`: the service has durably captured the group metadata and payload.
- `buffered -> batched`: a batching step has assigned the group for delivery.
- `batched -> delivered`: the batch was delivered to the downstream sink.
- `delivered -> acknowledged`: delivery was confirmed and the group is complete.

The store layer persists each transition timestamp (`*_at`) and the latest `state`, enabling restart-safe recovery and status inspection.

## Durability and recovery behavior

- For Redis-backed deployments, state is stored under durable keyed hashes.
- Restarting a process does not erase group state.
- Replaying an already-completed request remains idempotent by request ID and group key.

## Queue metrics

The store exposes queue health metrics:

- `depth`: count of groups not yet acknowledged.
- `oldest_age_seconds`: age of the oldest non-acknowledged group.

These metrics are emitted through runtime observability to support alerting and SLO tracking.
