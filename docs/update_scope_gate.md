# Update Scope Gate

`GOV-CODE-SCOPE-GATE-01` adds a standalone producer-side checker for DRPO
update packages. It answers one narrow question before normal tests run:

> Does this package appear to modify the files that match its declared task,
> claim, and authorization scope?

It does not prove the code is correct and it does not replace ruff, pytest,
`verify_update_package.py`, handoff authority checks, or user review. Its purpose
is to catch obvious scope drift early: a documentation update that changes
updater code, a bug fix that touches control-plane paths, a summary that omits
modified files, or a package whose real patch does not match the declared scope.

## Required CHANGE_SUMMARY metadata

Every package checked by this gate must include these fields in
`CHANGE_SUMMARY.md`:

```text
Task type: bug_fix | doc_update | experiment_code | experiment_registration | governance_control_plane | governance_tooling | pipeline_tooling | test_update | code_update
Claim or experiment ID: <claim or experiment ID>
User-requested scope: <one-sentence scope>
First-failure classification: not_applicable | direct_root_cause | baseline_red_unrelated | scope_expansion_authorized | governance_control_plane_authorized | new_feature_authorized
Control-plane touched: yes | no
```

It must also list the exact changed files under `## Modified files`.

## Usage

```bash
python3 scripts/validate_update_scope.py --package /path/to/update.zip
python3 scripts/validate_update_scope.py --package-root /path/to/extracted/package --json
```

The first version is standalone. It is intentionally not wired into
`drpo-update` or `verify_update_package.py`; those integrations can be considered
only after the checker has been exercised on real packages.
