# Stable error decisions

- `CONFIG_*`: repair configuration; do not route.
- `AUTH_*` or `QUOTA_*`: stop or select a separately eligible route; never buy
  credits automatically.
- `ROUTE_*`: return the explicit unavailable reason; no fallback.
- `HARNESS_*`: fail closed and keep the interactive service disabled.
- `GATE_*`: retain the artifact for review; never apply it.
- `ARTIFACT_*`: reject the artifact and preserve its metadata receipt.
- `SERVICE_*`: diagnose broker, OmniRoute, slot, and profile composition in that
  order.

Retries are reserved for transient infrastructure failures and use the same
model. A policy, authentication, quota, gate, or contract failure is not a
retryable infrastructure failure.

