#!/usr/bin/env bash
set -euo pipefail

python - <<'PY'
from pathlib import Path

scope = Path('docs/scopes/E8_MULTITASK_BASELINES_CAPABILITY_DEV.md')
text = scope.read_text(encoding='utf-8')
marker = '## Owner-authorized terminal evidence binding hardening (2026-09-10)'
if marker not in text:
    text = text.rstrip() + '''\n\n## Owner-authorized terminal evidence binding hardening (2026-09-10)\n\nThe owner requested continued fail-closed review and repair before any scientific launch. A completed-attempt fast path must therefore not accept a live workload merely because its source provenance and prepare config still match while terminal E8 evidence has been changed, removed, or mixed with another completion. When a guarded completed artifact is reused, the live terminal workload evidence must remain byte-bound to the packaged workload evidence, and the E8 terminal manifests must independently remain consistent with the current experiment/config/source commit, expected cell count, scheduler completion, terminal audit, and aggregate.\n\nFor the formal baseline matrix, the terminal aggregate must additionally bind the exact current set of 176 cell-manifest bytes through an ordered content fingerprint. Terminal audit and recovery stage planning must verify that binding before treating aggregate/audit/finalize as complete. A post-aggregate cell-manifest change therefore invalidates the aggregate and all downstream completion markers until aggregation and audit are rerun. `RUN_COMPLETE.json` must remain hash-bound to the terminal audit and aggregate that it certifies.\n\nThis is engineering/evidence hardening only. It changes no task, seed, AsymRE/TOPR/DPO hyperparameter, 1,200-update horizon, optimizer, evaluator, DPO initialization, historical PR #268 DPO mathematics, method ranking policy, or scientific status. The 176-cell experiment remains **not_run**.\n'''
    scope.write_text(text, encoding='utf-8')
PY

git add docs/scopes/E8_MULTITASK_BASELINES_CAPABILITY_DEV.md
git diff --cached --check
git commit -m 'docs: bind E8 terminal completion evidence'

python - <<'PY'
from pathlib import Path

path = Path('src/drpo/e8_multitask_exp_tuning.py')
text = path.read_text(encoding='utf-8')

# 1) Strengthen completed-attempt fast-path validation while preserving the
# legacy no-artifact identity probe used by older callers/tests.
start = text.index('def _successful_attempt_matches_current_identity(')
end = text.index('\ndef _effective_recovery_config(', start)
replacement = '''def _completed_workload_terminal_evidence_matches(\n    config: Mapping[str, Any],\n    workload_root: Path,\n    *,\n    source_commit: str,\n) -> bool:\n    """Validate terminal E8 evidence before a guarded completed attempt is reused."""\n\n    expected_id = experiment_id(config)\n    expected_hash = stable_config_hash(config)\n    expected_cells = len(build_cells(config))\n    try:\n        scheduler_path = workload_root / "scheduler" / "dynamic_run.json"\n        aggregate_path = workload_root / "aggregate" / "aggregate_summary.json"\n        audit_path = workload_root / "terminal_audit.json"\n        run_manifest_path = workload_root / "run_manifest.json"\n        scientific_manifest_path = workload_root / "scientific_run_manifest.json"\n        complete_path = workload_root / "RUN_COMPLETE.json"\n        plot_path = workload_root / "aggregate" / "plot_curve_points.csv"\n        if not all(\n            path.is_file()\n            for path in (\n                scheduler_path,\n                aggregate_path,\n                audit_path,\n                run_manifest_path,\n                scientific_manifest_path,\n                complete_path,\n                plot_path,\n            )\n        ):\n            return False\n        scheduler = _read_json_object(scheduler_path)\n        aggregate = _read_json_object(aggregate_path)\n        audit = _read_json_object(audit_path)\n        run_manifest = _read_json_object(run_manifest_path)\n        scientific_manifest = _read_json_object(scientific_manifest_path)\n        complete = _read_json_object(complete_path)\n    except (OSError, ValueError, TypeError, json.JSONDecodeError):\n        return False\n\n    if not (\n        scheduler.get("experiment_id") == expected_id\n        and scheduler.get("complete") is True\n        and int(scheduler.get("expected_cells", -1)) == expected_cells\n        and int(scheduler.get("completed_cells", -1)) == expected_cells\n        and not scheduler.get("failed_cells")\n        and not scheduler.get("unscheduled_cells")\n        and aggregate.get("experiment_id") == expected_id\n        and aggregate.get("source_commit") == source_commit\n        and int(aggregate.get("cell_count", -1)) == expected_cells\n        and audit.get("experiment_id") == expected_id\n        and audit.get("base_commit") == source_commit\n        and int(audit.get("expected_cells", -1)) == expected_cells\n        and audit.get("all_training_and_evaluation_complete") is True\n        and audit.get("test_partition_accessed") is False\n        and run_manifest.get("experiment_id") == expected_id\n        and run_manifest.get("base_commit") == source_commit\n        and run_manifest.get("source_commit") == source_commit\n        and run_manifest.get("config_hash") == expected_hash\n        and int(run_manifest.get("expected_cells", -1)) == expected_cells\n        and int(run_manifest.get("completed_cells", -1)) == expected_cells\n        and run_manifest.get("scheduler_run_id") == scheduler.get("scheduler_run_id")\n        and scientific_manifest == run_manifest\n        and complete.get("experiment_id") == expected_id\n        and complete.get("base_commit") == source_commit\n        and complete.get("source_commit") == source_commit\n        and complete.get("config_hash") == expected_hash\n        and int(complete.get("expected_cells", -1)) == expected_cells\n        and int(complete.get("completed_cells", -1)) == expected_cells\n        and complete.get("scheduler_run_id") == scheduler.get("scheduler_run_id")\n        and complete.get("all_training_and_evaluation_complete") is True\n        and complete.get("complete") is True\n        and complete.get("terminal_audit_sha256") == sha256_file(audit_path)\n        and complete.get("aggregate_sha256") == sha256_file(aggregate_path)\n    ):\n        return False\n\n    if _is_baseline_matrix(config):\n        if not _baseline_aggregate_identity_matches(config, workload_root, aggregate):\n            return False\n        if not all(\n            audit.get(field) is True\n            for field in (\n                "seed_batch_protocol_complete",\n                "seed_batch_event_identity_complete",\n                "seed_batch_execution_provenance_complete",\n                "seed_batch_temporal_order_complete",\n            )\n        ):\n            return False\n        if (\n            audit.get("cell_manifest_set_sha256")\n            != aggregate.get("cell_manifest_set_sha256")\n            or audit.get("aggregate_summary_sha256") != sha256_file(aggregate_path)\n        ):\n            return False\n    return True\n\n\ndef _successful_attempt_matches_current_identity(\n    config: Mapping[str, Any],\n    workload_root: Path,\n    *,\n    source_commit: str,\n    artifact_path: Path | None = None,\n) -> bool:\n    """Return whether live and packaged completed evidence matches this invocation."""\n\n    expected_id = experiment_id(config)\n    expected_hash = stable_config_hash(config)\n    try:\n        provenance = _read_json_object(workload_root / "source_provenance.json")\n        prepare = _read_json_object(workload_root / "prepare_manifest.json")\n        if not (\n            provenance.get("source_commit") == source_commit\n            and prepare.get("experiment_id") == expected_id\n            and prepare.get("config_hash") == expected_hash\n        ):\n            return False\n        if artifact_path is None:\n            return True\n        if not _completed_workload_terminal_evidence_matches(\n            config, workload_root, source_commit=source_commit\n        ):\n            return False\n        prefix = f"results/{expected_id}"\n        workload_members = (\n            "source_provenance.json",\n            "prepare_manifest.json",\n            "scheduler/dynamic_run.json",\n            "aggregate/aggregate_summary.json",\n            "aggregate/plot_curve_points.csv",\n            "terminal_audit.json",\n            "run_manifest.json",\n            "scientific_run_manifest.json",\n            "RUN_COMPLETE.json",\n        )\n        with zipfile.ZipFile(artifact_path) as archive:\n            artifact_manifest = json.loads(archive.read("ARTIFACT_MANIFEST.json"))\n            base_commit = archive.read("BASE_COMMIT.txt").decode("utf-8").strip()\n            outer_run_manifest = json.loads(archive.read(f"{prefix}/run_manifest.json"))\n            packaged_payloads = {\n                relative: archive.read(f"{prefix}/workload/{relative}")\n                for relative in workload_members\n            }\n    except (\n        OSError,\n        ValueError,\n        TypeError,\n        KeyError,\n        UnicodeDecodeError,\n        zipfile.BadZipFile,\n    ):\n        return False\n    if not isinstance(artifact_manifest, dict) or not isinstance(outer_run_manifest, dict):\n        return False\n    if not (\n        artifact_manifest.get("package_kind") == "experiment-raw-complete"\n        and artifact_manifest.get("experiment_id") == expected_id\n        and artifact_manifest.get("base_commit") == source_commit\n        and base_commit == source_commit\n        and outer_run_manifest.get("experiment_id") == expected_id\n        and outer_run_manifest.get("base_commit") == source_commit\n    ):\n        return False\n    for relative, packaged in packaged_payloads.items():\n        live = workload_root / relative\n        if not live.is_file():\n            return False\n        try:\n            if hashlib.sha256(packaged).hexdigest() != sha256_file(live):\n                return False\n        except OSError:\n            return False\n    return True\n\n'''
text = text[:start] + replacement + text[end+1:]

# 2) Add exact cell-manifest byte binding for the baseline-matrix aggregate.
anchor = 'def _aggregate_coldstart_matrix_unranked(\n'
idx = text.index(anchor)
helper = '''def _cell_manifest_set_sha256(\n    config: Mapping[str, Any],\n    output_root: Path,\n) -> str:\n    """Hash the ordered exact cell-manifest bytes for terminal evidence binding."""\n\n    records = []\n    for cell in build_cells(config):\n        path = output_root / "cells" / cell.key / "cell_manifest.json"\n        if not path.is_file():\n            raise FileNotFoundError(f"Cell manifest is missing for evidence binding: {cell.key}")\n        records.append({"cell_key": cell.key, "sha256": sha256_file(path)})\n    return stable_hash(records)\n\n\ndef _baseline_aggregate_identity_matches(\n    config: Mapping[str, Any],\n    output_root: Path,\n    aggregate: Mapping[str, Any],\n) -> bool:\n    """Verify that a baseline aggregate is bound to this source/config/cell set."""\n\n    if not _is_baseline_matrix(config):\n        return True\n    try:\n        provenance = _read_json_object(output_root / "source_provenance.json")\n        source_commit = str(provenance.get("source_commit", ""))\n        cell_set = _cell_manifest_set_sha256(config, output_root)\n    except (OSError, ValueError, TypeError, json.JSONDecodeError):\n        return False\n    return (\n        aggregate.get("experiment_id") == experiment_id(config)\n        and aggregate.get("config_hash") == stable_config_hash(config)\n        and aggregate.get("source_commit") == source_commit\n        and int(aggregate.get("cell_count", -1)) == len(build_cells(config))\n        and aggregate.get("cell_manifest_set_sha256") == cell_set\n    )\n\n\n'''
text = text[:idx] + helper + text[idx:]

matrix_start = text.index('def _aggregate_coldstart_matrix_unranked(')
matrix_end = text.index('\ndef _aggregate_coldstart(', matrix_start)
matrix = text[matrix_start:matrix_end]
needle = '        "source_commit": source_commit,\n        "cell_count": len(rows),\n'
assert matrix.count(needle) == 1, matrix.count(needle)
matrix = matrix.replace(
    needle,
    '        "source_commit": source_commit,\n'
    '        "config_hash": stable_config_hash(config),\n'
    '        "cell_manifest_set_sha256": _cell_manifest_set_sha256(config, output_root),\n'
    '        "cell_count": len(rows),\n',
)
text = text[:matrix_start] + matrix + text[matrix_end:]

# 3) Make terminal audit reject a stale/mixed aggregate and persist the exact
# evidence hashes it certified.
audit_start = text.index('def cmd_audit(')
audit_end = text.index('\n\nPACKAGE_REQUIRED_MEMBERS', audit_start)
audit = text[audit_start:audit_end]
old = '''        aggregate_complete = (\n            aggregate_path.is_file()\n            and int(json.loads(aggregate_path.read_text(encoding="utf-8")).get("cell_count", 0))\n            == len(cells)\n            and reproduction_gate_status == expected_protocol_status\n        )\n'''
new = '''        aggregate_value = (\n            _read_json_object(aggregate_path) if aggregate_path.is_file() else {}\n        )\n        aggregate_complete = (\n            aggregate_path.is_file()\n            and int(aggregate_value.get("cell_count", 0)) == len(cells)\n            and reproduction_gate_status == expected_protocol_status\n        )\n        if _is_baseline_matrix(config):\n            aggregate_complete = aggregate_complete and _baseline_aggregate_identity_matches(\n                config, output_root, aggregate_value\n            )\n'''
assert old in audit
audit = audit.replace(old, new, 1)
all_complete_anchor = '    all_complete = (\n'
assert all_complete_anchor in audit
binding = '''    cell_manifest_set_sha256: str | None = None\n    aggregate_summary_sha256: str | None = None\n    if _is_baseline_matrix(config):\n        try:\n            if not missing:\n                cell_manifest_set_sha256 = _cell_manifest_set_sha256(config, output_root)\n            aggregate_path = output_root / "aggregate" / "aggregate_summary.json"\n            if aggregate_path.is_file():\n                aggregate_summary_sha256 = sha256_file(aggregate_path)\n        except (OSError, ValueError, TypeError):\n            cell_manifest_set_sha256 = None\n            aggregate_summary_sha256 = None\n\n'''
audit = audit.replace(all_complete_anchor, binding + all_complete_anchor, 1)
dict_anchor = '        "dpo_reference_identity_failures": sorted(set(dpo_reference_identity_failures)),\n'
assert dict_anchor in audit
audit = audit.replace(
    dict_anchor,
    dict_anchor
    + '        "cell_manifest_set_sha256": cell_manifest_set_sha256,\n'
    + '        "aggregate_summary_sha256": aggregate_summary_sha256,\n',
    1,
)
text = text[:audit_start] + audit + text[audit_end:]

# 4) Recovery may skip aggregate/audit/finalize only when each downstream
# marker is still byte-bound to the current cell set.
recovery_start = text.index('def _recovery_stage_plan(')
recovery_end = text.index('\ndef cmd_recovery_plan(', recovery_start)
recovery = text[recovery_start:recovery_end]
old = '''    aggregate_path = output_root / "aggregate" / "aggregate_summary.json"\n    aggregate_complete = False\n    if aggregate_path.is_file():\n        try:\n            aggregate_complete = int(_read_json_object(aggregate_path).get("cell_count", 0)) == (\n                expected_cells\n            )\n        except (OSError, ValueError, TypeError, json.JSONDecodeError):\n            aggregate_complete = False\n    audit_path = output_root / "terminal_audit.json"\n    audit_complete = False\n    if audit_path.is_file():\n        try:\n            audit_complete = bool(\n                _read_json_object(audit_path).get("all_training_and_evaluation_complete")\n            )\n        except (OSError, ValueError, TypeError, json.JSONDecodeError):\n            audit_complete = False\n    finalized = False\n    complete_path = output_root / "RUN_COMPLETE.json"\n    if complete_path.is_file():\n        try:\n            finalized = bool(_read_json_object(complete_path).get("complete")) and audit_complete\n        except (OSError, ValueError, TypeError, json.JSONDecodeError):\n            finalized = False\n'''
new = '''    aggregate_path = output_root / "aggregate" / "aggregate_summary.json"\n    aggregate_complete = False\n    aggregate_value: dict[str, Any] = {}\n    if aggregate_path.is_file():\n        try:\n            aggregate_value = _read_json_object(aggregate_path)\n            aggregate_complete = int(aggregate_value.get("cell_count", 0)) == expected_cells\n            if _is_baseline_matrix(config):\n                aggregate_complete = aggregate_complete and _baseline_aggregate_identity_matches(\n                    config, output_root, aggregate_value\n                )\n        except (OSError, ValueError, TypeError, json.JSONDecodeError):\n            aggregate_complete = False\n    audit_path = output_root / "terminal_audit.json"\n    audit_complete = False\n    audit_value: dict[str, Any] = {}\n    if audit_path.is_file():\n        try:\n            audit_value = _read_json_object(audit_path)\n            audit_complete = bool(audit_value.get("all_training_and_evaluation_complete"))\n            if _is_baseline_matrix(config):\n                audit_complete = (\n                    audit_complete\n                    and aggregate_complete\n                    and audit_value.get("cell_manifest_set_sha256")\n                    == _cell_manifest_set_sha256(config, output_root)\n                    and audit_value.get("aggregate_summary_sha256")\n                    == sha256_file(aggregate_path)\n                )\n        except (OSError, ValueError, TypeError, json.JSONDecodeError):\n            audit_complete = False\n    finalized = False\n    complete_path = output_root / "RUN_COMPLETE.json"\n    if complete_path.is_file():\n        try:\n            complete_value = _read_json_object(complete_path)\n            finalized = bool(complete_value.get("complete")) and audit_complete\n            if _is_baseline_matrix(config):\n                finalized = (\n                    finalized\n                    and complete_value.get("terminal_audit_sha256") == sha256_file(audit_path)\n                    and complete_value.get("aggregate_sha256") == sha256_file(aggregate_path)\n                )\n        except (OSError, ValueError, TypeError, json.JSONDecodeError):\n            finalized = False\n'''
assert old in recovery
recovery = recovery.replace(old, new, 1)
text = text[:recovery_start] + recovery + text[recovery_end:]

# 5) Finalization refuses a stale baseline aggregate/audit even if counts alone
# look terminal-complete.
write_start = text.index('def _write_completion_manifests(')
write_end = text.index('\ndef _result_payload_paths(', write_start)
block = text[write_start:write_end]
needle = '''    if (\n        scheduler.get("experiment_id") != experiment_id(config)\n        or not scheduler.get("complete")\n        or int(scheduler.get("expected_cells", 0)) != expected_cells\n        or int(scheduler.get("completed_cells", 0)) != expected_cells\n        or int(aggregate.get("cell_count", 0)) != expected_cells\n    ):\n        raise RuntimeError("Scheduler or aggregate is not terminal-complete")\n'''
assert needle in block
insert = needle + '''    if _is_baseline_matrix(config):\n        if not _baseline_aggregate_identity_matches(config, output_root, aggregate):\n            raise RuntimeError("Baseline aggregate is not bound to the current cell manifests")\n        if (\n            audit.get("cell_manifest_set_sha256")\n            != aggregate.get("cell_manifest_set_sha256")\n            or audit.get("aggregate_summary_sha256") != sha256_file(aggregate_path)\n        ):\n            raise RuntimeError("Baseline terminal audit is stale relative to aggregate/cell evidence")\n'''
block = block.replace(needle, insert, 1)
text = text[:write_start] + block + text[write_end:]

path.write_text(text, encoding='utf-8')
PY

python - <<'PY'
from pathlib import Path

path = Path('tests/test_e8_multitask_p0.py')
text = path.read_text(encoding='utf-8')

# Update the full formal audit fixture so its synthetic aggregate satisfies the
# newly documented exact cell-manifest binding.
old = '    p0.atomic_json(tmp_path / "aggregate" / "aggregate_summary.json", {"cell_count": 176})\n'
new = '''    p0.atomic_json(\n        tmp_path / "aggregate" / "aggregate_summary.json",\n        {\n            "schema_version": 1,\n            "experiment_id": exp_tuning.experiment_id(config),\n            "source_commit": "a" * 40,\n            "config_hash": config_hash,\n            "cell_count": 176,\n            "cell_manifest_set_sha256": exp_tuning._cell_manifest_set_sha256(config, tmp_path),\n        },\n    )\n'''
assert text.count(old) == 1, text.count(old)
text = text.replace(old, new, 1)

# Make the existing end-to-end synthetic matrix test assert aggregate identity.
needle = '    assert summary["cell_count"] == 176\n    assert summary["method_ranking_allowed"] is False\n'
assert needle in text
text = text.replace(
    needle,
    '    assert summary["cell_count"] == 176\n'
    '    assert summary["config_hash"] == exp_tuning.stable_config_hash(config)\n'
    '    assert summary["cell_manifest_set_sha256"] == exp_tuning._cell_manifest_set_sha256(config, tmp_path)\n'
    '    assert summary["method_ranking_allowed"] is False\n',
    1,
)

append_marker = 'def test_baseline_terminal_evidence_binding_rejects_post_aggregate_cell_change('
if append_marker not in text:
    text = text.rstrip() + r'''\n\n\ndef test_baseline_terminal_evidence_binding_rejects_post_aggregate_cell_change(\n    tmp_path: Path, monkeypatch: pytest.MonkeyPatch\n) -> None:\n    from drpo import e8_multitask_exp_tuning as exp_tuning\n\n    config = exp_tuning._engineering_self_test_config(\n        _baseline_matrix_capability_test_config()\n    )\n    cells = exp_tuning.build_cells(config)\n    source_commit = "c" * 40\n    p0.atomic_json(\n        tmp_path / "source_provenance.json",\n        {"source_commit": source_commit, "run_id": "terminal-binding-test"},\n    )\n    for cell in cells:\n        score = 0.1\n        p0.atomic_json(\n            tmp_path / "cells" / cell.key / "cell_manifest.json",\n            {\n                "schema_version": 1,\n                "experiment_id": exp_tuning.experiment_id(config),\n                "config_hash": exp_tuning.stable_config_hash(config),\n                "cell": exp_tuning._cell_descriptor(cell),\n                "complete": True,\n                "evaluation_status": "complete",\n                "nan_inf_failure": False,\n                "engineering_placeholder_backend": True,\n                "validation_late_window_pass8_mean": score,\n                "validation_late_window_greedy_mean": score,\n                "validation_best_pass8": score,\n                "validation_terminal_pass8": score,\n                "validation_best_greedy": score,\n                "validation_terminal_greedy": score,\n                "validation_best_greedy_valid_rate": 1.0,\n                "validation_terminal_greedy_valid_rate": 1.0,\n                "best_step": 1200,\n                "terminal_step": 1200,\n                "stop_reason": "max_steps",\n            },\n        )\n    aggregate = exp_tuning.cmd_aggregate(config, tmp_path)\n    assert exp_tuning._baseline_aggregate_identity_matches(config, tmp_path, aggregate)\n\n    victim = tmp_path / "cells" / cells[0].key / "cell_manifest.json"\n    changed = json.loads(victim.read_text(encoding="utf-8"))\n    changed["validation_terminal_pass8"] = 0.2\n    p0.atomic_json(victim, changed)\n    assert not exp_tuning._baseline_aggregate_identity_matches(config, tmp_path, aggregate)\n\n\ndef test_completed_workload_terminal_evidence_rejects_stale_hash_chain(\n    tmp_path: Path, monkeypatch: pytest.MonkeyPatch\n) -> None:\n    from drpo import e8_multitask_exp_tuning as exp_tuning\n\n    config = exp_tuning._engineering_self_test_config(\n        _baseline_matrix_capability_test_config()\n    )\n    source_commit = "d" * 40\n    expected_cells = len(exp_tuning.build_cells(config))\n    scheduler_run_id = "terminal-evidence-run"\n    p0.atomic_json(tmp_path / "source_provenance.json", {"source_commit": source_commit})\n    for cell in exp_tuning.build_cells(config):\n        p0.atomic_json(\n            tmp_path / "cells" / cell.key / "cell_manifest.json",\n            {\n                "experiment_id": exp_tuning.experiment_id(config),\n                "config_hash": exp_tuning.stable_config_hash(config),\n                "complete": True,\n                "evaluation_status": "complete",\n                "nan_inf_failure": False,\n                "engineering_placeholder_backend": True,\n            },\n        )\n    aggregate = {\n        "experiment_id": exp_tuning.experiment_id(config),\n        "source_commit": source_commit,\n        "config_hash": exp_tuning.stable_config_hash(config),\n        "cell_count": expected_cells,\n        "cell_manifest_set_sha256": exp_tuning._cell_manifest_set_sha256(config, tmp_path),\n    }\n    p0.atomic_json(tmp_path / "aggregate" / "aggregate_summary.json", aggregate)\n    (tmp_path / "aggregate" / "plot_curve_points.csv").write_text("cell_key\\n", encoding="utf-8")\n    p0.atomic_json(\n        tmp_path / "scheduler" / "dynamic_run.json",\n        {\n            "experiment_id": exp_tuning.experiment_id(config),\n            "scheduler_run_id": scheduler_run_id,\n            "expected_cells": expected_cells,\n            "completed_cells": expected_cells,\n            "failed_cells": [],\n            "unscheduled_cells": [],\n            "complete": True,\n        },\n    )\n    audit = {\n        "experiment_id": exp_tuning.experiment_id(config),\n        "base_commit": source_commit,\n        "expected_cells": expected_cells,\n        "all_training_and_evaluation_complete": True,\n        "test_partition_accessed": False,\n        "seed_batch_protocol_complete": True,\n        "seed_batch_event_identity_complete": True,\n        "seed_batch_execution_provenance_complete": True,\n        "seed_batch_temporal_order_complete": True,\n        "cell_manifest_set_sha256": aggregate["cell_manifest_set_sha256"],\n        "aggregate_summary_sha256": exp_tuning.sha256_file(\n            tmp_path / "aggregate" / "aggregate_summary.json"\n        ),\n    }\n    p0.atomic_json(tmp_path / "terminal_audit.json", audit)\n    run_manifest = {\n        "experiment_id": exp_tuning.experiment_id(config),\n        "base_commit": source_commit,\n        "source_commit": source_commit,\n        "config_hash": exp_tuning.stable_config_hash(config),\n        "expected_cells": expected_cells,\n        "completed_cells": expected_cells,\n        "scheduler_run_id": scheduler_run_id,\n    }\n    p0.atomic_json(tmp_path / "run_manifest.json", run_manifest)\n    p0.atomic_json(tmp_path / "scientific_run_manifest.json", run_manifest)\n    p0.atomic_json(\n        tmp_path / "RUN_COMPLETE.json",\n        {\n            **run_manifest,\n            "all_training_and_evaluation_complete": True,\n            "terminal_audit_sha256": exp_tuning.sha256_file(tmp_path / "terminal_audit.json"),\n            "aggregate_sha256": exp_tuning.sha256_file(\n                tmp_path / "aggregate" / "aggregate_summary.json"\n            ),\n            "complete": True,\n        },\n    )\n    assert exp_tuning._completed_workload_terminal_evidence_matches(\n        config, tmp_path, source_commit=source_commit\n    )\n\n    stale = json.loads((tmp_path / "terminal_audit.json").read_text(encoding="utf-8"))\n    stale["seed_batch_temporal_order_complete"] = False\n    p0.atomic_json(tmp_path / "terminal_audit.json", stale)\n    assert not exp_tuning._completed_workload_terminal_evidence_matches(\n        config, tmp_path, source_commit=source_commit\n    )\n'''.replace('\\n', '\n')

path.write_text(text, encoding='utf-8')
PY

python -m py_compile src/drpo/e8_multitask_exp_tuning.py tests/test_e8_multitask_p0.py
python -m pytest -q tests/test_e8_multitask_p0.py
ruff check src/drpo/e8_multitask_exp_tuning.py tests/test_e8_multitask_p0.py
python scripts/handoff_authority.py verify --repo-root .
python scripts/validate_governance_pipeline_stage_status.py --repo-root .

git diff --check
git add src/drpo/e8_multitask_exp_tuning.py tests/test_e8_multitask_p0.py
git commit -m 'fix: bind E8 completed attempt terminal evidence'

git rm .github/workflows/tmp_e8_terminal_evidence_fix.sh .github/workflows/tmp_e8_terminal_evidence_fix.yml
git commit -m 'chore: remove temporary E8 terminal evidence transport'
git push origin HEAD:dev/e8-multitask-baselines-capability-01
