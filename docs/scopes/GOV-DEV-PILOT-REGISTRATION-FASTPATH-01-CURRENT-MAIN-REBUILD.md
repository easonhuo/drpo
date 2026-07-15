# GOV-DEV-PILOT-REGISTRATION-FASTPATH-01 — Current-Main Rebuild Amendment

**Claim:** `GOV-DEV-PILOT-REGISTRATION-FASTPATH-01`  
**Rebuild date:** 2026-07-15  
**Current-main base:** `d02c0ec29564aaa215ba82d952f62b6e20025c1a`  
**Reviewed source head:** `10022feabbe3260f4b8a3d8cb52ab5c0210596ef`  
**Target branch:** `dev/gov-dev-pilot-registration-fastpath-01`  
**Scientific impact:** none

## 1. Purpose

PR #62 was originally developed on an older `main`. The temporary transition
rule was subsequently merged and `main` advanced through unrelated governance
and manuscript work. This amendment authorizes a clean, single-parent rebuild
of the already reviewed PR-A tree on the exact current-main commit above.

The rebuild is not a new architecture phase. It imports the reviewed PR-A files
by exact Git blob identity and adds only this amendment plus a navigation index
under `docs/development_workflow_incidents/`.

## 2. Imported reviewed blobs

| Path | Reviewed blob SHA |
|---|---|
| `docs/dev_pilot_registration_fastpath.md` | `ce700f70fccec213c8bb53a7cf8bbd07bf458034` |
| `docs/development_workflow_incidents/DEVOPT-2026-07-14-PILOT-REGISTRATION-MERGE-01.md` | `85c0956c28f29db63dac0f178f7b38ea2e6327f2` |
| `docs/scopes/GOV-DEV-PILOT-REGISTRATION-FASTPATH-01-REAL-SHADOW-NOTE.md` | `e1d3ba5cc1f9fef2137d8c07b7bd8eefe94d3dc7` |
| `docs/scopes/GOV-DEV-PILOT-REGISTRATION-FASTPATH-01.md` | `204d082b10212fd12f733f85463c0141bd79e6ee` |
| `docs/templates/DEV_PILOT_REGISTRATION_SPEC.yaml` | `09538c961f23c040248cc819d9521b8f3f7a2d24` |
| `scripts/prepare_dev_pilot_registration.py` | `6196c83cc26a4846f90d0c11ae13db581628f274` |
| `tests/conftest.py` | `e3dc8c61aa76e0026cb2b65594836759a2b5498c` |
| `tests/test_prepare_dev_pilot_registration.py` | `d6adb1de2aaa719159adc30ea642fb91f2812745` |
| `tests/test_prepare_dev_pilot_registration_isolation.py` | `28309b91a12440d884c28faca7f4d4c8cf0dd07e` |
| `tests/test_prepare_dev_pilot_registration_real_shadow.py` | `2941f215b8e52b6084ba149841e5d87561dbeb4a` |

## 3. Preserved boundaries

The rebuilt PR must not:

- modify `docs/handoff.md` or `experiments/registry.yaml`;
- change an E7/E8 experiment matrix, seed, threshold, state, result, or priority;
- modify Stage 1/2/5 protected authority responsibilities;
- activate tiered CI;
- publish or merge generated registration candidates;
- create a second authority, transaction engine, registry renderer, publisher,
  or test-impact map;
- treat the historical real shadow as a scientific result.

The accepted V1 transaction remains the only path to local `READY`.

## 4. Rebuild acceptance

The rebuilt exact head must demonstrate:

1. its sole parent is the current-main base recorded above;
2. the ten imported paths are byte-identical to the reviewed source blobs;
3. only the two additive navigation/rebuild documents differ from the reviewed
   PR-A file set;
4. no unrelated current-main file is changed;
5. Python compilation, shell syntax, authority, formal-channel, governance,
   full pytest, Ruff, and the real V1 registration shadow pass;
6. the final PR remains Draft and unmerged until explicit user approval.

## 5. Relationship to activation

Merging PR-A makes the preparation adapter available on `main`; it does not by
itself make the adapter the default route. A later, separately reviewed
activation change must supersede the temporary transition rule in `AGENTS.md`,
state the default applicability and fallback conditions, and preserve an
immediate rollback to the existing manual V1 input path.
