# PAPER-CODE-VALIDATION-01 integration-freshness audit

## 1. Audit identity

- Repository: `easonhuo/drpo`
- Development branch: `dev/paper-code-reference-01`
- Audited development head: `987069546398d9535b76fa58cfcc1e3a3b169fe0`
- Observed current `main`: `cd3271f844bcaf2550beb6247451dd3104258d0b`
- Common merge base: `4544005bd7df69c53bad70a9dcac846af01285e4`
- Relation at audit time: development branch ahead by 208 commits and behind by
  226 commits
- Claim: `PAPER-CODE-VALIDATION-01`
- Parent claim: `PAPER-CODE-REFERENCE-01`
- Scientific status impact: none
- Formal experiment authorized or launched: no

This audit decides whether the paper-facing migration object changed while
`main` advanced. It is not a scientific result, a real-stack liveness run, a
terminal scientific audit, or an instruction to merge the Draft PR.

## 2. Authority and governance inherited from current main

Current `main` keeps `docs/handoff.md` as the read master and activates the
schema-v3 delta authority. This branch must not directly edit
`docs/handoff.md` or `experiments/registry.yaml` during integration. A future
research-status update must use one schema-v3 handoff delta and trusted-current-
main normalization. The present paper-code task remains code- and task-local-
document-only, so no handoff or registry mutation is required.

Current `main` also enforces the exact-path human gate for new Python files.
Draft PR #149 contains durable path-specific approval records, but the separate
protected human-review environment is not observable as satisfied from the
connector. Therefore this audit does not clear that merge gate.

## 3. Direct path-collision audit

Relative to current `main`, the entire `paper_code/` tree,
`docs/paper_code_reference/`, and `.github/workflows/paper-code-validation.yml`
remain additions. Current `main` has not independently introduced or modified
those same paths.

Verdict: **no direct same-path content collision was found** for the paper-code
package or its task-local validation documents.

This does not make the branch merge-ready. A merge with current `main` and a
new exact-head validation cycle are still required because repository-wide
governance, tests, handoff views, registry contents, and legacy differential
oracles changed after the common base.

## 4. Authoritative scientific-source freshness

### 4.1 C-U1

The current-main changes after the common base do not include the frozen
paper-facing C-U1 source set:

- `src/drpo/cu1_core.py`;
- `src/drpo/drpo_cu1_e1_e4_oneclick.py`;
- `src/drpo/cu1_distance_taper_formal.py`;
- `src/drpo/cu1_taper_near_retention_formal.py`;
- `src/drpo/cu1_taper_budget_match_formal.py`.

Freshness verdict: **unchanged migration source**. C-U1 remains
same-distribution held-out-context / unseen-state generalization and is not an
OOD protocol.

### 4.2 D-U1 revision 4

The current-main changes after the common base do not include:

- `src/drpo/du1_e6_cartesian_taper_v4.py`;
- `configs/du1_e6_cartesian_taper_v4.yaml`.

Freshness verdict: **unchanged migration source**. Newer D-U1 experiments and
closures in the research master do not silently replace revision 4 as the
selected paper-code migration oracle.

### 4.3 Hopper E7-Q2 mechanism validation

The exact current-main and development-branch blobs match:

| Path | Git blob |
|---|---|
| `src/drpo/e7_hopper_q2.py` | `a93eda3e94429a8764de9d6c35a98e09e6a0a14d` |
| `scripts/run_e7_hopper_q2.py` | `d3d964f3576f8c66a911c6b9f3cb1023e1985287` |
| `configs/e7_hopper_q2_medium_replay_v2.yaml` | `02e6e463256dcde96d640a42b6fbf8534fac8774` |

The audit found and repaired a task-local documentation error that named the
nonexistent `configs/e7_hopper_q2.yaml`; see
`VALIDATION_DEFECT_HOPPER_CONFIG_PATH_01.md`. The bounded repair changed only
three live validation documents and did not change the actual configuration.

Freshness verdict: **mechanism migration source unchanged**.

The newer E7 squared-EXP / TD-versus-GAE work on current `main` is a distinct
external development or diagnostic line. Its archived frozen-critic result is
explicitly non-formal and does not replace Hopper E7-Q2 mechanism evidence or
authorize the paper-code package to switch advantage or critic lifecycles.

### 4.4 D4RL-9 task-performance runtime

The current-main post-base change inventory does not modify the selected
canonical vendor files used as the D4RL reviewer-code differential boundary:

- `src/drpo/e7_canonical_vendor/d4rl/agents.py`;
- `src/drpo/e7_canonical_vendor/d4rl/train_sna2c_variant.py`;
- `src/drpo/e7_canonical_vendor/d4rl/d4rl_common/train_loop.py`;
- `src/drpo/e7_canonical_vendor/d4rl/d4rl_common/normalize.py`.

Freshness verdict: **reviewer-code oracle unchanged**, but scientific closure
remains blocked. Current-main registry state still does not freeze all nine
dataset identities, the final method matrix, formal seeds, budgets, or
checkpoint policy. No D4RL method-ranking claim is allowed.

### 4.5 Countdown

The registered active-tail v79 source and frozen YAML remain byte-identical
between current `main` and the development branch:

| Path | Git blob |
|---|---|
| `src/drpo/countdown_e8_taper.py` | `4cec05c3e21bf1e4a95abdda366e7890f8f7e053` |
| `configs/countdown_e8_taper_0p5b.yaml` | `8bda6181738543988a3671caf01aafdee2e3df82` |

One listed differential/provenance oracle did change:

- development branch `src/drpo/countdown_e8_alpha1_highc_scan_common.py`:
  `8ea17e3d07292bfc15827d7cdb906c17a5e95201`;
- current main:
  `7b831e778819e77f26ed3a3b7f63c025e904e393`.

The current-main version explicitly preserves the historical linear-surprisal
EXP profile and adds separate C-extension and reciprocal-family profiles. It
does not change the active-tail v79 source or configuration migrated by the
reviewer runtime. Current `main` also closes several completed offline E8 pilot
lines and reciprocal scans as pilots only; those closures do not convert
`EXT-C-E8-TAPER-0.5B-01` into a completed run and do not authorize convergence,
formal ranking, OOD, or universal exponential-superiority claims.

Freshness verdict: **active-tail migration source unchanged; expanded legacy
oracle requires post-merge differential regression**. No pre-merge source
rewrite is justified. After integrating current `main`, the exact-head
Countdown characterization tests must prove that historical Round-1
compatibility and the v79 active-tail behavior remain intact.

## 5. Engineering evidence before integration

At exact head `692adbd64cbecf89f78696e08a3f8f33533eedc2`:

- Evidence Locator Gate passed;
- PR Gate passed compile, shell syntax, handoff authority, formal execution
  channel, governance inventory/stage, full pytest, and Ruff;
- Paper Code Validation passed exact-head package construction, manifest
  verification, isolated installation, compile, self-contained tests, C-U1 and
  D-U1 tests, D-U1 public CPU liveness, Ruff check, Ruff format check, all public
  CLI entry points, and validated-package upload;
- validated artifact `8463571441` had digest
  `sha256:4f74db454886bbf41a041262327277b39b97217b869bbf83c160351f75994f31`;
- the downloaded artifact manifest and 68-file inventory were independently
  checked.

Subsequent commits only synchronized evidence documents and repaired the stale
Hopper config-path reference. These records establish a clean pre-integration
engineering baseline. They do not substitute for validation after current
`main` is incorporated.

## 6. Integration verdict

### Passed

- paper-code implementation paths do not collide with independently developed
  current-main paths;
- C-U1, D-U1 revision 4, Hopper E7-Q2, D4RL vendor oracle, and Countdown v79
  selected migration sources are not superseded by current-main source changes;
- the Hopper config-path documentation defect is registered and repaired;
- no scientific variable or status was changed by the audit.

### Still blocked

- the development branch has not yet incorporated current `main`;
- exact-head tests have not yet run on the merged tree;
- the expanded Countdown high-c oracle must pass the post-merge differential
  suite;
- the protected human new-Python-file review gate is not confirmed;
- Hopper real HDF5/MuJoCo liveness is not run;
- Countdown real Qwen/PEFT/CUDA liveness and interrupted optimizer/scheduler
  resume remain open;
- D4RL-9 real nine-task liveness and formal protocol freeze remain open;
- V5 scientific reproduction and terminal result review remain pending;
- explicit user instruction is required before merging Draft PR #149 to `main`.

Overall integration-freshness result:

```text
SOURCE_FRESHNESS_PRECHECK = PASS_WITH_POST_MERGE_REGRESSION_REQUIRED
MERGE_READINESS = BLOCKED
SCIENTIFIC_STATUS_CHANGE = NONE
```

## 7. Required continuation order

1. Incorporate the audited current-main commit into
   `dev/paper-code-reference-01` without dropping branch history.
2. Confirm the resulting merge contains no unreviewed handoff/registry mutation
   and no scientific-coordinate change.
3. Run exact-head Evidence Locator, PR Gate, and Paper Code Validation.
4. Inspect Countdown historical-linear and active-tail differential tests on the
   merged tree.
5. Record the merged-head artifact identity and final integration status.
6. Keep the PR Draft and unmerged until the protected human gate and explicit
   user merge instruction are satisfied.
7. Treat real Hopper, Countdown, and D4RL liveness and all V5 reproduction work
   as separate later gates; do not call the engineering merge a scientific
   reproduction.
