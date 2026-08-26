# Gates

- Documentation or configuration only: run `git diff --check` and relevant
  schema/config tests.
- Normal code: configure Release with GCC, build with at most two jobs, then run
  the complete CTest suite.
- Determinism surfaces (`core`, command stream, save/replay, RNG, checksum, or
  scheduler): run `bash scripts_ci/local_gates.sh`.

Workers may repair within their declared budget, but cannot weaken, delete, or
skip a gate. Confirm after the run that the canonical checkout is unchanged.

