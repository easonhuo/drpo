# GOV-RUNTIME-GPU-PLACEMENT-SHADOW-ENTRYPOINT-01 — safe selection-only server shadow

**Approval:** user explicitly authorized completion of the server acceptance path on 2026-07-15.  
**Stacked base:** `a378c4359777d7ae6202b001d9318241373f23a8` from Draft PR `#53`.  
**Dependency:** `GOV-RUNTIME-GPU-PLACEMENT-AUTOTUNE-02`.  
**Scientific impact:** none.  
**Default-policy impact:** none.

## Objective

Add one explicit stop boundary to the existing E8 GPU-placement auto entrypoint so a
real-H20 hardware shadow can run calibration, the phase-aware representative envelope,
bounded concurrency candidates, selection creation, cleanup, and provenance without
continuing into the full scientific sweep.

The current entrypoint always calls the slot runtime after placement selection. That
behavior is correct for an actual opt-in run but unsafe for an acceptance harness whose
only authority is engineering selection evidence.

## Authorized change

Add one CLI flag:

```text
--selection-only
```

When supplied, the entrypoint must:

1. preserve all applicable model/bank/validation/config validation;
2. preserve calibration identity and calibration execution;
3. preserve static GPU filtering;
4. preserve the full phase-aware measured-CPU placement selector;
5. write the same immutable `RUNTIME_SELECTION.json`;
6. print the same selected devices, slots per GPU, total slots, selector policy, and
   probe contract metadata, plus explicit `selection_only=true` and
   `test_split_access=not_accessed_selection_only` markers;
7. return zero immediately after selection output;
8. never call `countdown_e8_oracle_offline_v2_taper_slot_runtime.run`;
9. never require, hash, open, or otherwise access the test split;
10. record `test_sha256=null` in the selection fingerprint;
11. leave no probe process groups or model/checkpoint payload beyond the selector's
    existing cleanup contract.

Without the flag, a full run must still require `--test`; selection is followed by the
existing slot runtime and the historical full-run identity hash is retained.

## Explicit exclusions

- no selector arithmetic change;
- no probe phase, timeout, memory, CPU, VRAM, or candidate-policy change;
- no scientific dataset, split, seed, model, bank, method, coefficient, batch,
  accumulation, horizon, or evaluation change;
- no test-split access during engineering selection-only acceptance;
- no new GPU scheduler or resource pool;
- no thread tuning;
- no merge or default activation of PR `#53`;
- no handoff, registry, formal execution-channel, or experiment-status update;
- no claim that a hardware shadow is a scientific result.

## Deterministic acceptance

1. parser default remains full-run mode;
2. `--selection-only` is opt-in and does not require `--test`;
3. selection-only fingerprinting cannot read a supplied or missing test path;
4. full-run fingerprinting still requires a test path;
5. selection-only returns zero without invoking the slot runtime;
6. normal mode still delegates once with the selected placement artifact;
7. output includes selection-only and test-access markers;
8. Python compile, focused tests, full pytest, Ruff, and governance gates pass on the
   exact stacked head.

## Real-server acceptance

The later acceptance harness may invoke this exact commit with `--selection-only`. A
PASS requires at least the registered selector contract: complete required phases,
clean zero exits, no controller termination, no OOM, no live descendants, no test
access, and an immutable selection. If resources do not permit a candidate above one,
the result is `INCONCLUSIVE` for multi-slot capacity rather than a fabricated PASS.

## Rollback

Stop passing `--selection-only`; the historical full-run path remains unchanged. This
stacked change can be reverted independently of PR `#53`.
