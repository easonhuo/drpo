# Update Scope Gate

`GOV-CODE-SCOPE-GATE-01` added a standalone producer-side checker for DRPO
update packages. `GOV-CODE-SCOPE-GATE-02` wires that checker into the package
release preflight used by `scripts/package_update.py` before a runnable package
is emitted.

The gate answers one narrow question before normal tests run:

> Does this package appear to modify the files that match its declared task,
> claim, and authorization scope?

It does not prove the code is correct and it does not replace ruff, pytest,
handoff authority checks, or user review. Its purpose is to catch obvious scope
drift early: a documentation update that changes updater code, a bug fix that
touches control-plane paths, a summary that omits modified files, a package
whose real patch does not match the declared scope, or a stale package that
re-ships content already present on the current repository head.

## Required CHANGE_SUMMARY metadata

Every package checked by this gate must include these fields in
`CHANGE_SUMMARY.md`:

```text
Task type: bug_fix | doc_update | experiment_code | experiment_registration | governance_control_plane | governance_tooling | pipeline_tooling | test_update | code_update
Claim or experiment ID: <claim or experiment ID>
User-requested scope: <one-sentence scope>
First-failure classification: not_applicable | direct_root_cause | baseline_red_unrelated | scope_expansion_authorized | governance_control_plane_authorized | new_feature_authorized
Control-plane touched: yes | no
Scope justification: <optional justification when a package intentionally exceeds the small-diff thresholds>
```

It must also list the exact changed files under `## Modified files`.

## Standalone usage

```bash
python3 scripts/validate_update_scope.py --package /path/to/update.zip
python3 scripts/validate_update_scope.py --package-root /path/to/extracted/package --json
```

Pass `--repo` to enable current-base and duplicate-content checks:

```bash
python3 scripts/validate_update_scope.py --package /path/to/update.zip --repo . --json
```

Without `--repo`, the checker validates package-internal metadata only.

## Release-preflight integration

`GOV-CODE-SCOPE-GATE-02` makes
`docs/update_packaging_hardening/preflight_update_package` run the scope checker
by default after package extraction and base resolution. This keeps the change
outside the protected `scripts/verify_update_package.py` control-plane path while
still affecting the canonical producer path, because `scripts/package_update.py`
already runs release preflight before emitting a package.

Default mode is `--scope-check warn`:

* scope `FAIL` blocks preflight and package emission;
* scope `PASS_WITH_WARNINGS` is reported but does not block;
* scope `PASS` passes normally.

Use strict mode to make warnings blocking:

```bash
python3 docs/update_packaging_hardening/preflight_update_package \
  --repo /path/to/drpo \
  --package /path/to/update.zip \
  --scope-check strict
```

Use the rollback/bypass switch only for diagnosed false positives or emergency
compatibility:

```bash
python3 docs/update_packaging_hardening/preflight_update_package \
  --repo /path/to/drpo \
  --package /path/to/update.zip \
  --scope-check off
```

When invoking `scripts/package_update.py`, the corresponding emergency bypass is:

```bash
DRPO_UPDATE_SCOPE_CHECK=off python3 scripts/package_update.py ...
```

That bypass affects preflight scope checking only. It does not bypass bundle,
patch, manifest, executable-mode, handoff, governance, ruff, or pytest checks.

## V5-style failure coverage

The phase-2 checks are designed to catch the class of package failures where a
small bug fix accidentally re-ships a previous package or carries unrelated
governance materialization files. With repository context, a stale-base package
that supplies after-images already present on current `HEAD` fails. A bug-fix
package that directly modifies generated Stage 4A materialization files also
fails unless the task is explicitly declared as an authorized governance
control-plane update.
