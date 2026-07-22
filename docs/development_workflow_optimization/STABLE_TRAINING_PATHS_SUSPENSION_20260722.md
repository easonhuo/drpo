# Stable training-path restoration — 2026-07-22

Status: **ACTIVE STABILIZATION DECISION**

Repository base: `f6dac3b624e6ee293d7522f6af0a42c8eff5bad2`.

This record preserves the repository-owner decision to reduce the active execution
surface after real startup failures and unvalidated runtime-path expansion:

- remove the E7 PR #248 P2 profile runtime integration from `main` and restore the
  preceding E7 squared-night execution files;
- remove the executable E8 PR #250 config-driven adapter path and restore the
  preceding fixed-profile/fixed-matrix runtime;
- suspend M0 publication use while retaining its historical evidence and documents;
- leave ReplayAB, completed experiment results, `docs/handoff.md`,
  `experiments/registry.yaml`, and paper artifacts unchanged.

The completed P1/P2 scientific result records remain historical evidence. Removing
current-main launcher/profile integration does not rewrite those results.

Reactivation of either optimized runtime requires a separate documentation-first
scope, real end-to-end liveness on the intended execution environment, checkpoint
save/reload verification, and explicit repository-owner authorization. Smoke tests,
static checks, Replay, or finite-step pilots alone are insufficient.
