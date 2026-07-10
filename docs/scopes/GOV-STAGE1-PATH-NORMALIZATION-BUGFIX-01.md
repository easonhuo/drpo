# GOV-STAGE1-PATH-NORMALIZATION-BUGFIX-01 Scope Contract

## Identity

- Claim: `GOV-STAGE1-PATH-NORMALIZATION-BUGFIX-01`
- User authorization: approved on 2026-07-10 after the Phase 2A real PR run exposed the selector defect.
- Authoritative base: `30b6a9cfd9da2feca1e5be22f5c002d1459298ff`.
- Classification: Red maintenance bugfix because one frozen Stage 1 protected file and its ledger binding change.

## Root cause

`str.lstrip("./")` removes any leading dot or slash character rather than one exact `./` prefix. It therefore rewrites `.github/workflows/x.yml` to `github/workflows/x.yml`, preventing the registered `.github/**` control-plane pattern from matching. Unknown-path fail-closed behavior still selected the full suite, but risk classification and audit provenance were wrong.

## Allowed files

- `tools/drpo-update/test_selection.py` — exact normalization repair.
- `tests/test_update_test_selector_hidden_paths.py` — focused regression tests.
- `docs/governance_stage_authorizations/GOV-STAGE1-PATH-NORMALIZATION-BUGFIX-2026-07-10.yaml` — closed-stage maintenance authorization.
- `docs/governance_pipeline_stage_status.yaml` — update only the protected selector hash and authorization binding.
- `docs/scopes/GOV-STAGE1-PATH-NORMALIZATION-BUGFIX-01.md` — this scope contract.

## Forbidden changes

- No impact-map patterns, commands, risk levels, selector modes, or fail-closed policy changes.
- No GitHub Actions, branch-protection, updater authority, handoff authority, formal execution, scientific source, config, data, seed, threshold, budget, formula, result, or experiment-order changes.

## Required behavior

- Preserve `.github/**`, `.env`, and other leading-dot repository paths.
- Remove one exact leading `./` prefix only.
- Normalize Windows and POSIX separators to `/`.
- Make `.github/workflows/*.yml` match `test_control_plane`, classify as high risk, and select the full suite with no unknown path.
- Keep unmatched paths fail-closed to the full suite.

## Validation

```bash
python -m py_compile tools/drpo-update/test_selection.py tests/test_update_test_selector_hidden_paths.py
python -m pytest -q tests/test_update_test_selector.py tests/test_update_test_selector_hidden_paths.py tests/test_governance_pipeline_stage_status.py
python scripts/validate_governance_pipeline_stage_status.py --repo-root .
python -m compileall -q src scripts tools tests
bash -n tools/drpo-update/drpo-update tools/drpo-update/install.sh
python scripts/validate_formal_execution_channel.py --repo-root .
python scripts/validate_governance_rule_inventory.py --repo-root .
python -m pytest -q
ruff check .
```

## Rollback

Revert the selector change, focused test, authorization record, scope file, and exact ledger binding together. No historical authorization or scientific material is deleted.
