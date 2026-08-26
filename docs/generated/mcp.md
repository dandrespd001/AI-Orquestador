# Strict CHUNSA MCP

The server is bound to profile `chunsa` and exposes exactly four tools.

- `orchestrator_inventory` — Inspect secret-free providers, auth status, models, profiles, usage, extensions, updates, sessions or capability packages.
- `orchestrator_diagnostics` — Run secret-free configuration and policy diagnostics.
- `orchestrator_simulate_route` — Simulate routing without executing a model or enabling fallback.
- `orchestrator_submit_task` — Run an explicit AI-Orchestrator task spec through the common policy, provider adapters, isolation, gates and metadata-only telemetry. Writer tasks can only return a patch pending review. Without output_path the accepted output is returned in the payload and never written to disk; output_path must stay outside the task repository, which is staged and validated before the model runs.
