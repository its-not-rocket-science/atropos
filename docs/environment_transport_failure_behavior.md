# Environment transport failure behavior

The environment-to-API transport path (`TransportClient`) now enforces explicit request identity,
bounded retries, and typed terminal failures so failed sends do not get silently dropped.

## Request identity

- Every call to `TransportClient.send(payload)` guarantees a `request_id`.
- If the caller provides `payload["request_id"]`, that value is reused.
- Otherwise, the client generates one via `request_id_factory` (default UUID4).
- The same `request_id` is reused across all retries for that logical send operation.

## Retry policy

`TransportClient` retries **only retryable errors**:

- `TransportTimeoutError` (timeouts)
- `TransportServerError` (HTTP/transport 5xx)
- `RetryableTransportError` (custom mapped retriable exception classes)

Backoff is bounded exponential:

- `delay = min(base_backoff_seconds * backoff_multiplier^(retry_count-1), max_backoff_seconds)`
- default: `base_backoff_seconds=0.05`, `backoff_multiplier=2.0`, `max_backoff_seconds=1.0`

## Error taxonomy

- `TransportError`: base class
- `RetryableTransportError` / `NonRetryableTransportError`: policy boundary
- `TransportTimeoutError`: timeout retryable
- `TransportServerError`: 5xx retryable
- `TransportClientError`: 4xx non-retryable
- `TransportMalformedResponseError`: malformed/non-conformant response non-retryable
- `TransportRetriesExhaustedError`: raised when retry budget is consumed

## Metrics surfaced by transport

`TransportClient.metrics` tracks counters for operations and failures:

- `transport_success_total`
- `transport_retries_total`
- `transport_retry_count_total`
- `transport_terminal_failures_total`
- `transport_error_total:<ErrorType>`
- `transport_retries_total:<ErrorType>`
- `transport_terminal_failures_total:<ErrorType>`

These counters make retry pressure and terminal failures visible for alerting and debugging.

## Terminal failure semantics

On failure after retries, `send()` raises a typed exception (`TransportRetriesExhaustedError` or a
non-retryable subtype) and does **not** swallow the payload. The caller receives the exception and
can route the failed request to durable storage / dead-letter handling.
