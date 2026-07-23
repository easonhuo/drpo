# E8 fixed-profile TOPR dense-path reactivation — 2026-07-23

Status: **candidate; generic config-driven E8 remains suspended**.

Base repository commit: `4677f33ffba16060c3289f8371aa326568de67ea`.

The stable-training-path restoration deliberately removed the generic E8 config-driven runtime after startup failures and unvalidated execution-surface expansion. This update does not reverse that decision. It introduces one narrowly reviewed fixed profile for:

`EXT-C-E8-ORACLE-OFFLINE-V2-JOINT-FITTED-REFERENCE-BETA-TOPR-DENSE-0.5B-01`.

The permitted execution surface is limited to the exact configuration and exact shell entrypoint recorded in the experiment scope. The existing launcher may recognize this one experiment ID and require exactly two GPUs, but it does not infer arbitrary profile definitions or reactivate experiment-matrix execution.

The path remains blocked from a full pilot until all of the following are true:

1. exact-head compile, focused tests, full pytest, Ruff, authority, and governance checks pass;
2. the implementation commit is frozen and the RunSpec is bound to it;
3. the pilot is registered through the normal schema-v3 code-first route;
4. two-step real Qwen/PEFT/CUDA liveness at `beta=0.25` passes;
5. both policy and reference adapters receive finite nonzero updates;
6. the saved dual-adapter checkpoint is reloaded in a fresh process;
7. both reloaded adapters produce finite forward outputs;
8. the RunSpec is explicitly claimed.

Static checks, the liveness run, and checkpoint reload verification are engineering evidence only. They do not count among the 16 scientific cells and do not support performance, saturation, convergence, or ranking claims.

This candidate does not modify or erase the historical stabilization decision. Any broader E8 runtime reactivation requires a separate scope and explicit repository-owner approval.
