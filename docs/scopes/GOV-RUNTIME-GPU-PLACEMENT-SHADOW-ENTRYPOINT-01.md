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

1. preserve all existing input/config validation;
2. preserve calibration identity and calibration execution;
3. preserve static GPU filtering;
4. preserve the full phase-aware measured-CPU placement selector;
5. write the same immutable `RUNTIME_SELECTION.json`;
6. print the same selected devices, slots per GPU, total slots, selector policy, and
   probe contract metadata, plus an explicit `selection_only=true` marker;
7. return zero immediately after selection output;
8. never call `countdown_e8_oracle_offline_v2_taper_slot_runtime.run`;
9. leave no probe process groups or model/checkpoint payload beyond the selector's
   existing cleanup contract.

Without the flag, behavior must remain byte-for-byte equivalent at the control-flow
boundary: selection is followed by the existing slot runtime.

## Explicit exclusions

- no selector arithmetic change;
- no probe phase, timeout, memory, CPU, VRAM, or candidate-policy change;
- no scientific dataset, split, seed, model, bank, method, coefficient, batch,
  accumulation, horizon, or evaluation change;
- no new GPU scheduler or resource pool;
- no thread tuning;
- no merge or default activation of PR `#53`;
- no handoff, registry, formal execution-channel, or experiment-status update;
- no claim that a hardware shadow is a scientific result.

## Deterministic acceptance

1. parser default remains full-run mode;
2. `--selection-only` is opt-in;
3. selection-only returns zero without invoking the slot runtime;
4. normal mode still delegates once with the selected placement artifact;
5. output includes an explicit selection-only marker;
6. Python compile, focused tests, full pytest, Ruff, and governance gates pass on the
   exact stacked head.

## Real-server acceptance

The later acceptance harness may invoke this exact commit with `--selection-only`. A
PASS requires at least the registered selector contract: complete required phases,
clean zero exits, no controller termination, no OOM, no live descendants, and an
immutable selection. If resources do not permit a candidate above one, the result is
`INCONCLUSIVE` for multi-slot capacity rather than a fabricated PASS.

## Rollback

Stop passing `--selection-only`; the historical full-run path remains unchanged. This
stacked change can be reverted independently of PR `#53`.