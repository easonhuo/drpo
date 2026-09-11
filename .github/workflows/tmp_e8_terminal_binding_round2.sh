#!/usr/bin/env bash
set -euo pipefail

python - <<'PY'
from pathlib import Path
path = Path('docs/scopes/E8_MULTITASK_BASELINES_CAPABILITY_DEV.md')
text = path.read_text(encoding='utf-8')
marker = '## Terminal chain mutation closure (2026-09-11)'
if marker not in text:
    text += '''

## Terminal chain mutation closure (2026-09-11)

Continued pre-run review found that the already-authorized terminal-evidence contract must bind not only the aggregate to the exact cell-manifest set, but also the terminal audit to the exact scheduler result/event bytes it audited and the aggregate summary to its user-facing derived CSV/diagnostic files. Otherwise a post-audit scheduler-event mutation, or a post-aggregate derived-output mutation, could leave stale downstream completion markers apparently usable.

For the formal baseline matrix, `aggregate_summary.json` therefore records and revalidates SHA-256 for `all_cells.csv`, `plot_curve_points.csv`, `task_summary.csv`, and `countdown_protocol_diagnostic.json`. `terminal_audit.json` records and revalidates SHA-256 for `scheduler/dynamic_run.json` and `scheduler/queue_events.jsonl`, in addition to its existing cell-manifest-set and aggregate-summary bindings. Recovery planning, finalize/package completion markers, and successful-attempt reuse must reject stale or mutated members of this chain. Packaged completed-attempt reuse must byte-match these bound scheduler and aggregate artifacts to the live workload.

This closes the already-approved exact scheduler-event identity and terminal evidence binding requirements; it introduces no new scientific threshold or experiment responsibility. The 176 cells, seeds `[4000,5000]`, hard seed barrier, AsymRE/TOPR/DPO grids, 1,200-update horizon, optimizer/evaluator settings, historical PR #268 DPO semantics, and **not_run** status are unchanged.
'''
    path.write_text(text, encoding='utf-8')
PY

git add docs/scopes/E8_MULTITASK_BASELINES_CAPABILITY_DEV.md
git diff --cached --check
git commit -m 'docs: close E8 terminal chain mutation semantics'

python - <<'PY'
from pathlib import Path

path = Path('src/drpo/e8_multitask_exp_tuning.py')
text = path.read_text(encoding='utf-8')

# 1. Replace the aggregate identity helper block with a fully bound version.
start = text.index('def _cell_manifest_set_sha256(')
end = text.index('\ndef _aggregate_coldstart_matrix_unranked(', start)
replacement = '''BASELINE_AGGREGATE_BOUND_ARTIFACTS = (
    "all_cells.csv",
    "plot_curve_points.csv",
    "task_summary.csv",
    "countdown_protocol_diagnostic.json",
)


def _baseline_aggregate_artifact_sha256(output_root: Path) -> dict[str, str]:
    """Hash deterministic baseline-matrix aggregate outputs used for reporting."""

    aggregate_root = output_root / "aggregate"
    return {
        name: sha256_file(aggregate_root / name)
        for name in BASELINE_AGGREGATE_BOUND_ARTIFACTS
    }


def _cell_manifest_set_sha256(
    config: Mapping[str, Any],
    output_root: Path,
) -> str:
    """Hash the ordered exact cell-manifest bytes for terminal evidence binding."""

    records = []
    for cell in build_cells(config):
        manifest_path = output_root / "cells" / cell.key / "cell_manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(
                f"Cell manifest is missing for evidence binding: {cell.key}"
            )
        records.append({"cell_key": cell.key, "sha256": sha256_file(manifest_path)})
    return stable_hash(records)


def _baseline_aggregate_identity_matches(
    config: Mapping[str, Any],
    output_root: Path,
    aggregate: Mapping[str, Any],
) -> bool:
    """Verify that a baseline aggregate is bound to source, config, cells, and outputs."""

    if not _is_baseline_matrix(config):
        return True
    try:
        provenance = _read_json_object(output_root / "source_provenance.json")
        source_commit = str(provenance.get("source_commit", ""))
        cell_set = _cell_manifest_set_sha256(config, output_root)
        aggregate_artifacts = _baseline_aggregate_artifact_sha256(output_root)
        cell_count = int(aggregate.get("cell_count", -1))
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return False
    return (
        aggregate.get("experiment_id") == experiment_id(config)
        and aggregate.get("config_hash") == stable_config_hash(config)
        and aggregate.get("source_commit") == source_commit
        and cell_count == len(build_cells(config))
        and aggregate.get("cell_manifest_set_sha256") == cell_set
        and aggregate.get("aggregate_artifact_sha256") == aggregate_artifacts
    )


def _baseline_terminal_audit_identity_matches(
    config: Mapping[str, Any],
    output_root: Path,
    audit: Mapping[str, Any],
    aggregate: Mapping[str, Any],
) -> bool:
    """Verify that a baseline terminal audit matches the exact evidence it audited."""

    if not _is_baseline_matrix(config):
        return True
    try:
        provenance = _read_json_object(output_root / "source_provenance.json")
        source_commit = str(provenance.get("source_commit", ""))
        expected_cells = len(build_cells(config))
        aggregate_path = output_root / "aggregate" / "aggregate_summary.json"
        scheduler_path = output_root / "scheduler" / "dynamic_run.json"
        event_path = output_root / "scheduler" / "queue_events.jsonl"
        current_cell_set = _cell_manifest_set_sha256(config, output_root)
        aggregate_sha = sha256_file(aggregate_path)
        scheduler_sha = sha256_file(scheduler_path)
        event_sha = sha256_file(event_path)
        nan_inf_event_count = int(audit.get("nan_inf_event_count", -1))
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return False
    empty_failure_fields = (
        "missing_cells",
        "incomplete_cells",
        "nan_inf_cells",
        "terminal_contract_failures",
        "cell_identity_failures",
        "scientific_execution_provenance_failures",
        "dpo_reference_identity_failures",
    )
    return (
        _baseline_aggregate_identity_matches(config, output_root, aggregate)
        and audit.get("experiment_id") == experiment_id(config)
        and audit.get("base_commit") == source_commit
        and audit.get("expected_cells") == expected_cells
        and audit.get("execution_class") == _execution_class(config)
        and audit.get("scientific_status") == _audited_scientific_status(config, True)
        and audit.get("all_training_and_evaluation_complete") is True
        and audit.get("aggregate_complete") is True
        and audit.get("test_partition_accessed") is False
        and all(audit.get(field) == [] for field in empty_failure_fields)
        and nan_inf_event_count == 0
        and audit.get("seed_batch_protocol_complete") is True
        and audit.get("seed_batch_event_identity_complete") is True
        and audit.get("seed_batch_execution_provenance_complete") is True
        and audit.get("seed_batch_temporal_order_complete") is True
        and audit.get("cell_manifest_set_sha256") == current_cell_set
        and audit.get("cell_manifest_set_sha256")
        == aggregate.get("cell_manifest_set_sha256")
        and audit.get("aggregate_summary_sha256") == aggregate_sha
        and audit.get("scheduler_dynamic_run_sha256") == scheduler_sha
        and audit.get("scheduler_queue_events_sha256") == event_sha
    )

'''
text = text[:start] + replacement + text[end + 1:]

# 2. Make the aggregate summary bind every deterministic reporting artifact.
anchor = '''    method_metadata = {
        METHOD_ASYMRE: _canonical_baseline_grid_identity(METHOD_ASYMRE),
'''
insert = '''    protocol_diagnostic = _countdown_protocol_diagnostic(
        config,
        output_root,
        destination=output_root / "aggregate" / "countdown_protocol_diagnostic.json",
    )
    method_metadata = {
        METHOD_ASYMRE: _canonical_baseline_grid_identity(METHOD_ASYMRE),
'''
if text.count(anchor) != 1:
    raise SystemExit(f'matrix method metadata anchor count={text.count(anchor)}')
text = text.replace(anchor, insert, 1)
anchor = '''        "cell_manifest_set_sha256": _cell_manifest_set_sha256(config, output_root),
        "cell_count": len(rows),
'''
insert = '''        "cell_manifest_set_sha256": _cell_manifest_set_sha256(config, output_root),
        "aggregate_artifact_sha256": _baseline_aggregate_artifact_sha256(output_root),
        "cell_count": len(rows),
'''
if text.count(anchor) < 1:
    raise SystemExit('matrix aggregate hash anchor missing')
# Limit to the matrix aggregate function slice by using the first occurrence after its def.
matrix_pos = text.index('def _aggregate_coldstart_matrix_unranked(')
pos = text.index(anchor, matrix_pos)
text = text[:pos] + text[pos:].replace(anchor, insert, 1)
anchor = '''        "countdown_protocol_diagnostic": _countdown_protocol_diagnostic(
            config,
            output_root,
            destination=output_root / "aggregate" / "countdown_protocol_diagnostic.json",
        ),
'''
if text.count(anchor) < 1:
    raise SystemExit('matrix diagnostic anchor missing')
pos = text.index(anchor, matrix_pos)
text = text[:pos] + text[pos:].replace(anchor, '        "countdown_protocol_diagnostic": protocol_diagnostic,\n', 1)

# 3. Recovery stage completion must use the same full terminal identity check.
old = '''        try:
            audit_value = _read_json_object(audit_path)
            audit_complete = bool(audit_value.get("all_training_and_evaluation_complete"))
            if _is_baseline_matrix(config):
                audit_complete = (
                    audit_complete
                    and aggregate_complete
                    and audit_value.get("cell_manifest_set_sha256")
                    == _cell_manifest_set_sha256(config, output_root)
                    and audit_value.get("aggregate_summary_sha256")
                    == sha256_file(aggregate_path)
                )
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            audit_complete = False
'''
new = '''        try:
            audit_value = _read_json_object(audit_path)
            audit_complete = bool(audit_value.get("all_training_and_evaluation_complete"))
            if _is_baseline_matrix(config):
                aggregate_value = _read_json_object(aggregate_path)
                audit_complete = audit_complete and _baseline_terminal_audit_identity_matches(
                    config, output_root, audit_value, aggregate_value
                )
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            audit_complete = False
'''
if text.count(old) != 1:
    raise SystemExit(f'recovery audit anchor count={text.count(old)}')
text = text.replace(old, new, 1)

# 4. Terminal audit records the scheduler/event bytes that its identity checks consumed.
old = '''    cell_manifest_set_sha256: str | None = None
    aggregate_summary_sha256: str | None = None
    if _is_baseline_matrix(config):
        try:
            if not missing:
                cell_manifest_set_sha256 = _cell_manifest_set_sha256(config, output_root)
            aggregate_path = output_root / "aggregate" / "aggregate_summary.json"
            if aggregate_path.is_file():
                aggregate_summary_sha256 = sha256_file(aggregate_path)
        except (OSError, ValueError, TypeError):
            cell_manifest_set_sha256 = None
            aggregate_summary_sha256 = None
'''
new = '''    cell_manifest_set_sha256: str | None = None
    aggregate_summary_sha256: str | None = None
    scheduler_dynamic_run_sha256: str | None = None
    scheduler_queue_events_sha256: str | None = None
    if _is_baseline_matrix(config):
        try:
            if not missing:
                cell_manifest_set_sha256 = _cell_manifest_set_sha256(config, output_root)
            aggregate_path = output_root / "aggregate" / "aggregate_summary.json"
            scheduler_path = output_root / "scheduler" / "dynamic_run.json"
            event_path = output_root / "scheduler" / "queue_events.jsonl"
            if aggregate_path.is_file():
                aggregate_summary_sha256 = sha256_file(aggregate_path)
            if scheduler_path.is_file():
                scheduler_dynamic_run_sha256 = sha256_file(scheduler_path)
            if event_path.is_file():
                scheduler_queue_events_sha256 = sha256_file(event_path)
        except (OSError, ValueError, TypeError):
            cell_manifest_set_sha256 = None
            aggregate_summary_sha256 = None
            scheduler_dynamic_run_sha256 = None
            scheduler_queue_events_sha256 = None
'''
if text.count(old) != 1:
    raise SystemExit(f'audit hash anchor count={text.count(old)}')
text = text.replace(old, new, 1)
old = '''        "cell_manifest_set_sha256": cell_manifest_set_sha256,
        "aggregate_summary_sha256": aggregate_summary_sha256,
        "seed_batch_protocol_complete": seed_batch_protocol_complete,
'''
new = '''        "cell_manifest_set_sha256": cell_manifest_set_sha256,
        "aggregate_summary_sha256": aggregate_summary_sha256,
        "scheduler_dynamic_run_sha256": scheduler_dynamic_run_sha256,
        "scheduler_queue_events_sha256": scheduler_queue_events_sha256,
        "seed_batch_protocol_complete": seed_batch_protocol_complete,
'''
if text.count(old) != 1:
    raise SystemExit(f'audit dict hash anchor count={text.count(old)}')
text = text.replace(old, new, 1)

# 5. Finalization revalidates the full audit evidence rather than trusting booleans/hashes alone.
old = '''    if _is_baseline_matrix(config):
        if not _baseline_aggregate_identity_matches(config, output_root, aggregate):
            raise RuntimeError("Baseline aggregate is not bound to the current cell manifests")
        if (
            audit.get("cell_manifest_set_sha256")
            != aggregate.get("cell_manifest_set_sha256")
            or audit.get("aggregate_summary_sha256") != sha256_file(aggregate_path)
        ):
            raise RuntimeError("Baseline terminal audit is stale relative to aggregate/cell evidence")
'''
new = '''    if _is_baseline_matrix(config):
        if not _baseline_terminal_audit_identity_matches(
            config, output_root, audit, aggregate
        ):
            raise RuntimeError(
                "Baseline terminal audit is stale or inconsistent with current terminal evidence"
            )
'''
if text.count(old) != 1:
    raise SystemExit(f'completion audit anchor count={text.count(old)}')
text = text.replace(old, new, 1)

# 6. RUN_COMPLETE carries the scheduler/event binding for later fast-path checks.
old = '''    atomic_json(
        output_root / "RUN_COMPLETE.json",
        {
            **run_manifest,
            "all_training_and_evaluation_complete": bool(
                audit["all_training_and_evaluation_complete"]
            ),
            "terminal_audit_sha256": sha256_file(output_root / "terminal_audit.json"),
            "aggregate_sha256": sha256_file(aggregate_path),
            "complete": True,
        },
    )
'''
new = '''    completion = {
        **run_manifest,
        "all_training_and_evaluation_complete": bool(
            audit["all_training_and_evaluation_complete"]
        ),
        "terminal_audit_sha256": sha256_file(output_root / "terminal_audit.json"),
        "aggregate_sha256": sha256_file(aggregate_path),
        "complete": True,
    }
    if _is_baseline_matrix(config):
        completion.update(
            {
                "scheduler_dynamic_run_sha256": audit["scheduler_dynamic_run_sha256"],
                "scheduler_queue_events_sha256": audit["scheduler_queue_events_sha256"],
            }
        )
    atomic_json(output_root / "RUN_COMPLETE.json", completion)
'''
if text.count(old) != 1:
    raise SystemExit(f'RUN_COMPLETE anchor count={text.count(old)}')
text = text.replace(old, new, 1)

# 7. Completed-workload reuse re-runs the exact terminal audit identity check and
#    confirms RUN_COMPLETE was created from the same scheduler/event evidence.
old = '''    if _is_baseline_matrix(config):
        if not _baseline_aggregate_identity_matches(config, workload_root, aggregate):
            return False
        if not all(
'''
new = '''    if _is_baseline_matrix(config):
        if not _baseline_terminal_audit_identity_matches(
            config, workload_root, audit, aggregate
        ):
            return False
        if (
            complete.get("scheduler_dynamic_run_sha256")
            != audit.get("scheduler_dynamic_run_sha256")
            or complete.get("scheduler_queue_events_sha256")
            != audit.get("scheduler_queue_events_sha256")
        ):
            return False
        if not all(
'''
if text.count(old) != 1:
    raise SystemExit(f'completed workload baseline anchor count={text.count(old)}')
text = text.replace(old, new, 1)

# 8. Packaged completed-attempt reuse byte-compares every bound terminal artifact.
old = '''            "scheduler/dynamic_run.json",
            "aggregate/aggregate_summary.json",
            "aggregate/plot_curve_points.csv",
            "terminal_audit.json",
'''
new = '''            "scheduler/dynamic_run.json",
            "scheduler/queue_events.jsonl",
            "aggregate/aggregate_summary.json",
            "aggregate/all_cells.csv",
            "aggregate/plot_curve_points.csv",
            "aggregate/task_summary.csv",
            "aggregate/countdown_protocol_diagnostic.json",
            "terminal_audit.json",
'''
if text.count(old) != 1:
    raise SystemExit(f'package terminal member anchor count={text.count(old)}')
text = text.replace(old, new, 1)

path.write_text(text, encoding='utf-8')
PY

python - <<'PY'
from pathlib import Path

path = Path('tests/test_e8_multitask_p0.py')
text = path.read_text(encoding='utf-8')

# Update the manually synthesized formal terminal fixture so it represents the
# same evidence shape as the production matrix aggregate/audit.
old = '''    p0.atomic_json(
        tmp_path / "aggregate" / "aggregate_summary.json",
        {
            "schema_version": 1,
            "experiment_id": exp_tuning.experiment_id(config),
            "source_commit": "a" * 40,
            "config_hash": config_hash,
            "cell_count": 176,
            "cell_manifest_set_sha256": exp_tuning._cell_manifest_set_sha256(config, tmp_path),
        },
    )
    p0.atomic_json(
        tmp_path / "aggregate" / "countdown_protocol_diagnostic.json", {"status": "NOT_RUN"}
    )
'''
new = '''    aggregate_root = tmp_path / "aggregate"
    aggregate_root.mkdir(parents=True, exist_ok=True)
    (aggregate_root / "all_cells.csv").write_text("cell_key\\n", encoding="utf-8")
    (aggregate_root / "plot_curve_points.csv").write_text("cell_key\\n", encoding="utf-8")
    (aggregate_root / "task_summary.csv").write_text("task\\n", encoding="utf-8")
    p0.atomic_json(aggregate_root / "countdown_protocol_diagnostic.json", {"status": "NOT_RUN"})
    p0.atomic_json(
        aggregate_root / "aggregate_summary.json",
        {
            "schema_version": 1,
            "experiment_id": exp_tuning.experiment_id(config),
            "source_commit": "a" * 40,
            "config_hash": config_hash,
            "cell_count": 176,
            "cell_manifest_set_sha256": exp_tuning._cell_manifest_set_sha256(config, tmp_path),
            "aggregate_artifact_sha256": exp_tuning._baseline_aggregate_artifact_sha256(tmp_path),
        },
    )
'''
if text.count(old) != 1:
    raise SystemExit(f'formal terminal aggregate fixture anchor count={text.count(old)}')
text = text.replace(old, new, 1)

# The compact terminal-evidence helper fixture needs all deterministic aggregate
# files plus the new scheduler/event hashes and exact audit success fields.
old = '''    aggregate = {
        "experiment_id": exp_tuning.experiment_id(config),
        "source_commit": source_commit,
        "config_hash": exp_tuning.stable_config_hash(config),
        "cell_count": expected_cells,
        "cell_manifest_set_sha256": exp_tuning._cell_manifest_set_sha256(config, tmp_path),
    }
    p0.atomic_json(tmp_path / "aggregate" / "aggregate_summary.json", aggregate)
    (tmp_path / "aggregate" / "plot_curve_points.csv").write_text("cell_key\\n", encoding="utf-8")
'''
new = '''    aggregate_root = tmp_path / "aggregate"
    aggregate_root.mkdir(parents=True, exist_ok=True)
    (aggregate_root / "all_cells.csv").write_text("cell_key\\n", encoding="utf-8")
    (aggregate_root / "plot_curve_points.csv").write_text("cell_key\\n", encoding="utf-8")
    (aggregate_root / "task_summary.csv").write_text("task\\n", encoding="utf-8")
    p0.atomic_json(
        aggregate_root / "countdown_protocol_diagnostic.json",
        {"status": "NOT_RUN_ENGINEERING"},
    )
    aggregate = {
        "experiment_id": exp_tuning.experiment_id(config),
        "source_commit": source_commit,
        "config_hash": exp_tuning.stable_config_hash(config),
        "cell_count": expected_cells,
        "cell_manifest_set_sha256": exp_tuning._cell_manifest_set_sha256(config, tmp_path),
        "aggregate_artifact_sha256": exp_tuning._baseline_aggregate_artifact_sha256(tmp_path),
    }
    p0.atomic_json(aggregate_root / "aggregate_summary.json", aggregate)
'''
if text.count(old) != 1:
    raise SystemExit(f'completed aggregate fixture anchor count={text.count(old)}')
text = text.replace(old, new, 1)

old = '''    audit = {
        "experiment_id": exp_tuning.experiment_id(config),
        "base_commit": source_commit,
        "expected_cells": expected_cells,
        "all_training_and_evaluation_complete": True,
        "test_partition_accessed": False,
        "seed_batch_protocol_complete": True,
        "seed_batch_event_identity_complete": True,
        "seed_batch_execution_provenance_complete": True,
        "seed_batch_temporal_order_complete": True,
        "cell_manifest_set_sha256": aggregate["cell_manifest_set_sha256"],
        "aggregate_summary_sha256": exp_tuning.sha256_file(
            tmp_path / "aggregate" / "aggregate_summary.json"
        ),
    }
'''
new = '''    (tmp_path / "scheduler" / "queue_events.jsonl").write_text("{}\\n", encoding="utf-8")
    audit = {
        "experiment_id": exp_tuning.experiment_id(config),
        "base_commit": source_commit,
        "expected_cells": expected_cells,
        "missing_cells": [],
        "incomplete_cells": [],
        "nan_inf_cells": [],
        "terminal_contract_failures": [],
        "cell_identity_failures": [],
        "scientific_execution_provenance_failures": [],
        "dpo_reference_identity_failures": [],
        "nan_inf_event_count": 0,
        "all_training_and_evaluation_complete": True,
        "aggregate_complete": True,
        "test_partition_accessed": False,
        "execution_class": exp_tuning._execution_class(config),
        "scientific_status": exp_tuning._audited_scientific_status(config, True),
        "seed_batch_protocol_complete": True,
        "seed_batch_event_identity_complete": True,
        "seed_batch_execution_provenance_complete": True,
        "seed_batch_temporal_order_complete": True,
        "cell_manifest_set_sha256": aggregate["cell_manifest_set_sha256"],
        "aggregate_summary_sha256": exp_tuning.sha256_file(
            tmp_path / "aggregate" / "aggregate_summary.json"
        ),
        "scheduler_dynamic_run_sha256": exp_tuning.sha256_file(
            tmp_path / "scheduler" / "dynamic_run.json"
        ),
        "scheduler_queue_events_sha256": exp_tuning.sha256_file(
            tmp_path / "scheduler" / "queue_events.jsonl"
        ),
    }
'''
if text.count(old) != 1:
    raise SystemExit(f'completed audit fixture anchor count={text.count(old)}')
text = text.replace(old, new, 1)

old = '''            "aggregate_sha256": exp_tuning.sha256_file(
                tmp_path / "aggregate" / "aggregate_summary.json"
            ),
            "complete": True,
'''
new = '''            "aggregate_sha256": exp_tuning.sha256_file(
                tmp_path / "aggregate" / "aggregate_summary.json"
            ),
            "scheduler_dynamic_run_sha256": audit["scheduler_dynamic_run_sha256"],
            "scheduler_queue_events_sha256": audit["scheduler_queue_events_sha256"],
            "complete": True,
'''
if text.count(old) != 1:
    raise SystemExit(f'RUN_COMPLETE fixture anchor count={text.count(old)}')
text = text.replace(old, new, 1)

extra = r'''


def test_baseline_terminal_chain_rejects_post_audit_scheduler_and_derived_mutation(
    tmp_path: Path,
) -> None:
    from drpo import e8_multitask_exp_tuning as exp_tuning

    config = exp_tuning.load_config("configs/e8_multitask_baseline_matrix_formal.yaml")
    source_commit = "e" * 40
    root = tmp_path / "terminal-chain"
    result = exp_tuning.cmd_engineering_self_test(config, root, source_commit=source_commit)
    assert result["complete"] is True
    effective = exp_tuning.load_config(root / "engineering_self_test_config.yaml")
    aggregate_path = root / "aggregate" / "aggregate_summary.json"
    audit_path = root / "terminal_audit.json"
    aggregate = json.loads(aggregate_path.read_text(encoding="utf-8"))
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    assert exp_tuning._baseline_aggregate_identity_matches(effective, root, aggregate)
    assert exp_tuning._baseline_terminal_audit_identity_matches(
        effective, root, audit, aggregate
    )
    assert exp_tuning._completed_workload_terminal_evidence_matches(
        effective, root, source_commit=source_commit
    )

    plot_path = root / "aggregate" / "plot_curve_points.csv"
    plot_bytes = plot_path.read_bytes()
    plot_path.write_bytes(plot_bytes + b"tampered\n")
    assert not exp_tuning._baseline_aggregate_identity_matches(effective, root, aggregate)
    plan = exp_tuning._recovery_stage_plan(
        effective,
        root,
        base_model_path=str(root / "engineering_fixtures" / "placeholder_model"),
    )
    assert plan["aggregate_complete"] is False
    assert plan["audit_complete"] is False
    assert plan["finalized"] is False
    plot_path.write_bytes(plot_bytes)

    event_path = root / "scheduler" / "queue_events.jsonl"
    event_bytes = event_path.read_bytes()
    event_path.write_bytes(event_bytes + b"{}\n")
    aggregate = json.loads(aggregate_path.read_text(encoding="utf-8"))
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    assert not exp_tuning._baseline_terminal_audit_identity_matches(
        effective, root, audit, aggregate
    )
    assert not exp_tuning._completed_workload_terminal_evidence_matches(
        effective, root, source_commit=source_commit
    )
    plan = exp_tuning._recovery_stage_plan(
        effective,
        root,
        base_model_path=str(root / "engineering_fixtures" / "placeholder_model"),
    )
    assert plan["audit_complete"] is False
    assert plan["finalized"] is False
    event_path.write_bytes(event_bytes)

    stale_audit = json.loads(audit_path.read_text(encoding="utf-8"))
    stale_audit["base_commit"] = "f" * 40
    p0.atomic_json(audit_path, stale_audit)
    with pytest.raises(RuntimeError, match="stale or inconsistent"):
        exp_tuning.cmd_finalize(effective, root)
'''
if 'test_baseline_terminal_chain_rejects_post_audit_scheduler_and_derived_mutation' not in text:
    text += extra

path.write_text(text, encoding='utf-8')
PY

python -m py_compile src/drpo/e8_multitask_exp_tuning.py tests/test_e8_multitask_p0.py
python -m pytest -q tests/test_e8_multitask_p0.py
ruff check src/drpo/e8_multitask_exp_tuning.py tests/test_e8_multitask_p0.py
python scripts/handoff_authority.py verify --repo-root .
python scripts/validate_governance_pipeline_stage_status.py --repo-root .
python scripts/preflight_e8_multitask_config.py --config configs/e8_multitask_baseline_matrix_formal.yaml
git diff --check

git add src/drpo/e8_multitask_exp_tuning.py tests/test_e8_multitask_p0.py
git diff --cached --check
git commit -m 'fix: bind E8 terminal scheduler and aggregate artifacts'

git rm .github/workflows/tmp_e8_terminal_binding_round2.sh .github/workflows/tmp_e8_terminal_binding_round2.yml
git commit -m 'chore: remove temporary E8 terminal chain transport'
git push origin HEAD:dev/e8-multitask-baselines-capability-01
