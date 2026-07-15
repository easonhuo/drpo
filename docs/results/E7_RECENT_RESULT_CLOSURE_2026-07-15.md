# Recent E7 Result Closure Audit — 2026-07-15

**Repository base:** `easonhuo/drpo@0da7bd7bea1684c922fe7d4b25890be3c0327666`  
**Scientific class:** Hopper/D4RL external-validity development evidence  

**Consolidated machine-readable evidence:** `experiments/results/e7_recent_result_closure_20260715/RESULT_CLOSURE.json`

**Authority:** this report is a compact evidence audit, not a second research master. `docs/handoff.md` and `experiments/registry.yaml` remain authoritative after schema-v3 materialization.

## Scope and reporting rules

This closure preserves the original experiment matrices, seeds, horizons, failures, and terminal-audit outcomes. It does not rerun or impute any branch. Fixed horizons are not convergence or steady state. Task-performance degradation, support/variance-boundary events, and NaN/Inf numerical failures remain separate.

## Closure inventory

| Experiment | Run commit / identity | Terminal state | Branches | Numerical failures | Compact evidence |
|---|---|---:|---:|---:|---|
| `EXT-H-E7-PPO-STABILITY-01` | `118b9c39578dfefbc6e184a470cf8dca6be57775` | recorded failure terminal | 95/96 | 1 | consolidated closure JSON |
| `EXT-H-E7-PPO-W0-EXP-GRID-01` | `5383af681de79ee433788dd77beeb9cb561935af` | audit FAIL | 185/186 | 1 | consolidated closure JSON |
| `EXT-H-E7-W0-HIGHC-ACTOR-01` | `6ac6d4cc8bc245eaa48907fb38a73059eef8906f` | audit PASS | 84/84 | 0 | consolidated closure JSON |
| `EXT-H-E7-SQUARED-EXP-NIGHT-01` | existing archived result | audit PASS | 126/126 | 0 | existing `experiments/results/e7_squared_exp_night_1m_pilot/` |
| `EXT-H-E7-SQUARED-EXP-KL-TUNE-01` Stage A | `2d4d295022c75b0c2cde283d2d9c3402779c5764` | audit FAIL | 149/150 | 1 | consolidated closure JSON |
| `EXT-H-E7-SQEXP-ACTOR-DECISION-01` | `d1afb5ff094f69986e0ecc3bf7f9385485add62b` | audit PASS | 192/192 | 0 | consolidated closure JSON |
| `EXT-H-E7-SQEXP-HIGHC-BOUNDARY-01` | `6795aa6f086c44e8073c5a995a1612f334a3a067` | audit PASS | 48/48 | 0 | consolidated closure JSON |
| Combined squared-EXP high-c boundary | exact join of preceding two | both audits PASS | 120 compact rows | 0 | consolidated closure JSON |
| `EXT-H-E7-BENCH-01` canonical nine-task archive | archive run identity `99185455...` | archive validation PASS | 576/576 | 0 reported | consolidated closure JSON |
| `EXT-H-E7-BENCH-01` ExpRank-MR sanity | archive identity | archive validation PASS | 8/8 | 0 reported | consolidated closure JSON |

## Locked result interpretations

### PPO stability

The predeclared universal PPO-stability hypothesis is not supported. Across 47 complete matched actor pairs, the mean PPO-minus-A2C late-window difference is `-14.934461` and the mean FINAL difference is `-16.894195`. Dataset effects are heterogeneous: Hopper-medium-expert and Walker2d-medium degrade on average, while Walker2d-medium-replay has a positive late/final signal. One Walker2d-medium PPO branch fails with a non-finite likelihood ratio at 500k.

### Direct `w(0)` grid

The 500k grid reaches terminal state with 185/186 branches. The missing Walker2d-medium `w(0)=0.025,c=0` cell is a recorded non-finite PPO-ratio failure and is not imputed. The compact evidence preserves the viable-cell audit and failure identity, but this incomplete two-seed screen does not select a common parameter setting.

### High-`c` actor screen

The 84-branch screen completes without numerical failure. Across 21 dataset/control actor cells, the pooled PPO-minus-A2C late difference is `-1.687041` with PPO late wins in `8/21` cells. Effects vary by dataset and coefficient; no universal actor or coefficient is selected.

### KL Stage A

Stage A reaches terminal state with 149/150 branches. The fixed-K16 Positive-only Walker2d-medium seed-200 branch fails at 750k with a non-finite PPO ratio. Because the predeclared gate requires all 150 branches to be present and terminal-audited, Stage-A qualification is `NOT_EVALUABLE`; no KL threshold qualifies and Stage B remains unauthorized. Descriptive lifecycle comparisons are retained but cannot be promoted to selection.

### Squared-EXP actor decision and high-c boundary

The exact predecessor and successor packages restore the full Positive-only → `c64` → `c128` → `c256` → `c512` boundary. The predecessor PPO-retention gate is `FAIL`; PPO is not retained for the E7 mainline under this backbone. Adjacent high-c late-window effects are mixed rather than universally monotonic, while effective negative mass falls sharply as `c` increases. No common `c`, PPO selection, or `c1024` follow-up is authorized.

### Nine-task BENCH archive

The original archive filename and manifest say seed 200, but `RUN_IDENTITY.json` and the complete compact table contain seeds 200 and 201: 288 rows each, 576 total. The machine-readable closure records this metadata correction without modifying any score. The original report's post-hoc per-task best cells and broad-grid ranking remain descriptive only. The source repository commit is absent from the archive and remains unresolved, so this result remains pilot provenance rather than formal benchmark ranking.

## Remaining provenance limits

1. The canonical nine-task BENCH archive does not contain a resolvable repository commit SHA.
2. Full raw packages remain external and are bound by SHA-256; only compact decision evidence is committed.
3. No result here accesses held-out seeds `204--207`.
4. No result establishes convergence, steady state, controlled causal identification, or OOD generalization.
