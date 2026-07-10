# GOV-PR-GATE-OBSERVER-01 Scope Contract

## Identity

- Claim: `GOV-PR-GATE-OBSERVER-01`
- User request: add Phase-1 telemetry for existing PR/update gates before deciding which gates to retain, downgrade, or remove.
- Base commit: `30b6a9cfd9da2feca1e5be22f5c002d1459298ff`
- Dev branch: `dev/gov-pr-gate-observer-01`
- Implementation agent: ChatGPT
- Reviewer/gatekeeper: independent review required; implementation must not self-merge.

## Classification

- Minimal Sufficient Diff class: **Yellow**.
- Rationale: the change is narrow but requires one observer entry point, one focused regression-test file, and this scope contract.
- The observer is external, read-only instrumentation around the frozen Stage-1 selector. It does not change selector decisions, gate commands, enforcement policy, or test-impact mapping.

## Allowed changes

- `docs/scopes/GOV-PR-GATE-OBSERVER-01.md` — lock the implementation scope before code changes.
- `scripts/run_pr_gate_observer.py` — execute the existing selector in observe-only mode and persist structured telemetry.
- `tests/test_pr_gate_observer.py` — focused regression coverage for pass/fail/unavailable gates, annotations, and infrastructure errors.

## Forbidden changes

- `AGENTS.md`, `docs/handoff.md`, `experiments/registry.yaml`, and `docs/handoff_deltas/**`.
- `.github/**` in Phase 1; CI integration is a later separately reviewed change.
- Frozen Stage-1 files: `tools/drpo-update/test_selection.py`, `tools/drpo-update/test_impact_map.json`, `scripts/select_update_tests.py`, package/update authority code, and their governance fingerprints.
- Scientific source, configs, datasets, seeds, budgets, optimizers, thresholds, formulas, convergence criteria, result status, or experiment ordering.

## Scientific variables explicitly unchanged

No scientific variable is in scope. C-U1, D-U1, Hopper, and Countdown experiment definitions and statuses remain unchanged.

## Required behavior

- Reuse the existing selector and impact map read-only.
- Preserve each selected command's existing log.
- Write `gate_report.json`, `gate_summary.md`, and `logs/*.log`.
- Record changed paths, selector trigger reason, gate duration, pass/fail/unavailable status, classification, merge-blocking annotation, and follow-up commit.
- Allow reviewer annotations only from an explicit JSON input.
- Default failed gates to `actionable`, unavailable gates to `environment`, and passed gates to no failure classification.
- Remain observe-only: gate failures are recorded but do not make the observer exit nonzero. Selector/reporting infrastructure failures exit nonzero.

## Test commands

```bash
python -m py_compile scripts/run_pr_gate_observer.py tests/test_pr_gate_observer.py
python -m pytest -q tests/test_pr_gate_observer.py
python scripts/handoff_authority.py verify --repo-root .
python scripts/validate_governance_pipeline_stage_status.py --repo-root .
```

## Run command

```bash
python scripts/run_pr_gate_observer.py \
  --repo . \
  --base <reviewed-base-commit> \
  --head <reviewed-head-commit> \
  --output-dir outputs/pr_gate_observer/<run-id>
```

The placeholders above are runtime commit identities, not filesystem paths. A real invocation must substitute exact commits.

## Merge criteria

- Actual diff contains only the three allowed paths.
- Frozen selector, impact map, governance authority, and scientific files are unchanged.
- Focused tests pass on the reviewed head commit.
- Governance authority verification and stage-status validation pass in a full checkout or CI.
- Generated reports are observe-only and do not silently alter merge policy.
- Independent reviewer returns `merge_ready`; the implementation agent does not merge its own PR.
