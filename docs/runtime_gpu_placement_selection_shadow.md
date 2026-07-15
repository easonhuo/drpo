# GPU placement selection-only hardware shadow

**Claim:** `GOV-RUNTIME-GPU-PLACEMENT-SHADOW-ENTRYPOINT-01`  
**Dependency:** Draft PR `#53` at `a378c4359777d7ae6202b001d9318241373f23a8`  
**Scientific impact:** none

The GPU-placement auto entrypoint normally continues from placement selection into the
slot runtime. A server acceptance run must instead stop after the immutable placement
artifact is created.

Use the opt-in flag:

```bash
python scripts/run_countdown_e8_oracle_offline_v2_taper_auto.py \
  --selection-only \
  --model_path /absolute/model \
  --work_dir /absolute/new-shadow-work \
  --bank /absolute/bank.jsonl \
  --val /absolute/validation.jsonl \
  --global_calibration /absolute/global_calibration.json \
  --base_config /absolute/base.yaml \
  --sweep_config /absolute/sweep.yaml \
  --gpus 0,1,2,3
```

Selection-only mode does not require, hash, or read the test split. Its workload
fingerprint records:

```text
test_split_access: not_accessed_selection_only
test_sha256: null
```

A normal full run still requires `--test` and preserves its historical identity hash.

The command still performs calibration, static GPU filtering, the full phase-aware
single-worker envelope, bounded concurrency candidates, process-group cleanup, and
selection creation. It writes:

```text
<work_dir>/RUNTIME_SELECTION.json
```

and then exits without calling the scientific slot runtime. Standard output includes:

```text
selection_only: true
test_split_access: not_accessed_selection_only
scientific_matrix_changed: false
```

Omitting `--selection-only` preserves the historical opt-in full-run behavior.

A selection-only run is engineering evidence. It cannot establish task performance,
method ranking, convergence, steady state, or external-validity conclusions.
