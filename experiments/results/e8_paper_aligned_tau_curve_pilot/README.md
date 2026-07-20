# E8 paper-aligned tau-curve pilot result

## Identity

- experiment: `EXT-C-E8-ORACLE-OFFLINE-V2-PAPER-ALIGNED-TAU-CURVE-0.5B-01`;
- run: `E8_PAPER_ALIGNED_TAU_CURVE_20260717_01`;
- scientific status: `pilot`;
- role: Countdown external-validity response-curve localization;
- result repository: `easonhuo/drpo-results`;
- immutable result commit: `8baae3f728043fc85f14b56c246091e64aeb9dfe`;
- result manifest SHA-256:
  `e737f94d9bb3f8c4dc08551ac2606a49cea39a639ba8b951d5f99d6d810524a0`.

The full text result contains all 32 cell summaries, metrics, logs, aggregate files,
terminal audit, source-artifact manifest, and READY marker. Checkpoints remain
server-local and are not part of this compact repository evidence.

## Completion audit

- `32/32` expected cell summaries are present;
- failed cells: `0`;
- terminal audit: `PASS`;
- NaN/Inf numerical failures: `0`;
- test data used: `false`;
- fixed 1200 steps are not convergence or steady state.

Task performance, valid-structure/support proxy behavior, and NaN/Inf numerical
failure remain separate. The run did not formally instrument a support-boundary
event; valid rate is only an auxiliary structure proxy.

## Main descriptive result

The preregistered primary metric is mean validation Pass@8 over steps
`800,900,1000,1100,1200`. Across the four `c` anchors, moderate `tau` values
produce several improvements over each same-run `tau=0` cell, while the
descriptive cross-`c` means at `tau=1.0` and `tau=1.25` fall below the `tau=0`
mean. The best local `tau` varies with `c`, so this is a `c x tau` response
surface rather than evidence for one global optimum.

The complete 32-point table is stored in `RESULT_SUMMARY.json`. Raw trajectories
remain authoritative in the result repository.

## Provenance limitation and repair

The completed run records a clean local source commit
`f9ea5a155ada50e9a4aebbe8ed08e8ffec82d66a`, but that Git object is not remotely
resolvable. This limitation is not erased or rewritten.

The implementation has been rebuilt directly from merged current main. The
RunSpec pins the remote-resolvable implementation commit
`613bddf54c5314fba1459b02d829d2d4fed72bd1`. Exact SHA-256 values recorded by the
run for six protected source files and two configurations are enforced by the
existing E8 pytest module. A passing exact-head CI therefore proves source-byte
equivalence while preserving the unresolved historical Git-commit fact.

## Claim boundary

This single-seed pilot supports only descriptive response-surface localization.
It does not establish statistical significance, an exact optimum, cross-seed
method ranking, convergence, steady state, OOD generalization, or controlled
causal identification.
