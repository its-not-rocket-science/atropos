# Atropos observability dashboard examples

Import `atropos-observability-dashboard.json` into Grafana and point panels to your Prometheus data source.

Tracked signals:

- rollout latency (p95, env label)
- queue size per env
- worker utilization
- API error rates

The dashboard assumes the runtime app exposes `/metrics` and BaseEnv emits utilization/queue gauges.
