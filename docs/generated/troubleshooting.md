# Troubleshooting

Use `doctor --check readiness` for runtime dependencies and `doctor --check compliance` for policy drift. `AUTH_*`, `QUOTA_*`, and `ROUTE_*` do not permit silent fallback. `HARNESS_*` keeps the service disabled until the active slot and both profile compositions pass.
