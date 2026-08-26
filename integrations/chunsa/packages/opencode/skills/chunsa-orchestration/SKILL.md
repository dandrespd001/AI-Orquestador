---
name: chunsa-orchestration
description: Route, submit, review, and accept CHUNSA engineering tasks through AI-Orchestrator when work needs model selection, an isolated writer, a patch artifact, or deterministic acceptance gates.
---

# CHUNSA orchestration

Keep the canonical checkout under human control. Use `codebase-memory-mcp` first
for structural discovery, then build an explicit TaskSpec bound to profile
`chunsa`. Simulate the route before submitting whenever model choice, quota, or
write authority is uncertain. Never replace a rejected or unavailable route
with a silent fallback.

Automated writers work in an isolated worktree and return a patch artifact.
Review the artifact and its receipt, run the risk-appropriate gates, then accept
or reject explicitly. Direct interactive Harness edits are the sole exception
and must remain inside an approved Git workspace.

Read only the reference needed for the current operation:

- TaskSpec, receipt, MCP, and artifact shapes: [contracts.md](references/contracts.md)
- Gate selection and deterministic invariants: [gates.md](references/gates.md)
- Stable failures and recovery decisions: [errors.md](references/errors.md)

