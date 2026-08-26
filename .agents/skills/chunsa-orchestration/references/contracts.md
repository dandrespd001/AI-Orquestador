# Contracts

Use TaskSpec schema v2. Bind `repo` to the canonical CHUNSA realpath, put task
instructions in a repository file named by `prompt_file`, declare `important`
only when the task may consume the shared Codex allowance, and use
`allowed_write_paths` for every non-read-only task.

The strict MCP exposes exactly inventory, diagnostics, route simulation, and
task submission. It is fixed to profile `chunsa`; callers cannot override the
profile. Automated writer output is a patch artifact pending review, never an
implicit apply.

HarnessReceiptV2 contains identifiers, task hash, profile, provider/model,
Harness version/transport, timing, usage, status, artifact metadata, and gates.
It must never contain prompts, messages, reasoning, responses, or other content.

