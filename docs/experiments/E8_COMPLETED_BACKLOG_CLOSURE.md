# Countdown E8 completed-result backlog closure

This record closes six completed validation-only external-validity pilots without rerunning training or strengthening their claims.

- Experiments: **6**
- Completed cells: **222/222**
- Terminal audits: **PASS for all six pilot lines**
- NaN/Inf numerical failures: **0**
- Separately materialized `test.jsonl` used: **no**
- Structurally disjoint held-out evaluation used: **yes (`val.jsonl`)**

## Temporary held-out evaluation convention

For the current Countdown E8 coefficient-response evidence, the file named `val.jsonl` temporarily substitutes for the separately materialized `test.jsonl` as the task-performance evaluation split. It is structurally disjoint from the training bank in canonical structure families and `(numbers, target)` problem keys, and its rows do not enter the training loss.

The paper-facing evidence is the complete declared coefficient curve under the fixed late window and terminal horizon. A validation-selected best checkpoint may be stored as supplementary diagnostic or recovery evidence, but it is not the source of the reported response curve. Therefore, the historical statement that the test split was unused refers only to the separate `test.jsonl`; it does not mean that held-out evaluation was absent or that task performance was measured on the training bank.

This is a temporary evaluation convention rather than an untouched one-shot confirmatory-test claim. Several later grids were extended after inspecting earlier held-out curves, so the combined evidence must be called **staged held-out evaluation response curves**. It must not be described as a pristine final test, and it does not authorize formal method ranking, significance, convergence, steady state, OOD generalization, or universal exponential superiority. The 222 cells do not require rerunning solely because the additional `test.jsonl` was not accessed.

## Evidence inventory

- `EXT-C-E8-ORACLE-OFFLINE-V2-CONTINUOUS-EXP-GRID-0.5B-01` — 62/62 cells; source `9de742532ac8559a0aba1282151e66cc1ce22f9e`; package `9bc0b3a7623717bd17da29d2478ea4ed52150176e6b2d269fc81d88f9fb1964e`.
- `EXT-C-E8-ORACLE-OFFLINE-V2-ALPHA1-C-SCAN-0.5B-01` — 32/32 cells; source `a54dc74b849561c15f6195336fca446ed36f0640`; package `58522afed3072337138c29752efbf99ca8a4b65fe54a79adf1f7c153354416fb`.
- `EXT-C-E8-ORACLE-OFFLINE-V2-ALPHA1-HIGHC-SCAN-0.5B-01` — 32/32 cells; source `929142930a3e2efaa7cafc8e4afe3866600027a5`; package `73fd7e21b7921e02bb67a0d8ddf4842431a3f6ccd07f80aea7aed2b273c6f53f`.
- `EXT-C-E8-ORACLE-OFFLINE-V2-ALPHA1-LOGC-BOUNDARY-SCAN-0.5B-01` — 32/32 cells; source `05e8704770bda9a8682cd1031fa8b67bc3b55a41`; package `4f4237c1261289ca8a6f85850b216f6dd7c107598be1299f6d5f0f6587b941ec`.
- `EXT-C-E8-ORACLE-OFFLINE-V2-PAPER-ALIGNED-LINEAR-SCAN-0.5B-01` — 32/32 cells; source `f957e7f63c376e328e3d677cb143d526f6937c51`; package `00d53286a6642998b5563045bbb278876ba38713f98819c679ed4825e49bdd48`.
- `EXT-C-E8-ORACLE-OFFLINE-V2-PAPER-ALIGNED-TAU-CURVE-0.5B-01` — 32/32 cells; source `f9ea5a155ada50e9a4aebbe8ed08e8ffec82d66a`; package `8487365fc15be097733cad2df3fd235dba7a4cb44d4f5e19e37f40c937ae8982`.

## Locked interpretation

The four historical continuous-EXP scans support a robust qualitative conclusion: uncontrolled Global negative pressure is harmful, while sufficient tapering opens a broad usable coefficient region. They do not support a sharp best coefficient because seed/trajectory variation is large and the historical evaluator changed global RNG state without restoration.

The paper-aligned Linear scan localized a near-tied region around the best observed coefficient and the tested right boundary, but did not close the descending branch. The Tau curve is a single-seed response surface and cannot establish an optimum.

Task performance, valid-expression/structure diagnostics, and NaN/Inf numerical failure remain separate. Every line remains a pilot: no significance, convergence, steady-state, formal method ranking, OOD claim, or universal exponential-superiority claim is authorized.

## Provenance limitation

The first four source packages predate automatic `drpo-results` delivery. Their full raw packages remain external and are bound here by the previously audited source commits and SHA-256 values. The Linear and Tau compact evidence is copied from the original result-closure branches and also bound to immutable manifests in `easonhuo/drpo-results`.
