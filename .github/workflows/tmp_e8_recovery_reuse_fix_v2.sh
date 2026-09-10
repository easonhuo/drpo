#!/usr/bin/env bash
set -euo pipefail

python - <<'PY'
from pathlib import Path

scope = Path('docs/scopes/E8_MULTITASK_BASELINES_CAPABILITY_DEV.md')
text = scope.read_text(encoding='utf-8').rstrip()
block = '''

## Recovery seed-chronology implementation closure (2026-09-10)

Repeated pre-run review found one implementation gap inside the already-authorized recovery contract. Per-cell identity/provenance validity is necessary but not sufficient for reuse under the frozen seed barrier: if any seed-4000 cell must execute again, preserving an older seed-5000 result can make its original scientific start precede the newly executed seed-4000 finish. Likewise, a recovery snapshot whose individually valid cells already violate the persisted cross-seed chronology cannot be treated as queue-complete.

Recovery reuse therefore uses one shared global eligibility rule across recovery planning, checkpoint snapshots, import, and dynamic scheduling. Individually valid cells in the currently incomplete seed may still be reused when their persisted chronology is compatible with the preceding fully reusable seed. Once a seed has any non-reusable cell, every later seed's older cells are ineligible for reuse and must execute again; if a fully/partially reusable later seed already starts before the preceding reusable seed finished, that seed and all later seeds are likewise ineligible. This is an implementation repair of the existing hard `4000 -> 5000` scientific barrier and original-provenance requirement, not a new scientific gate or threshold. It changes no task, cell, seed, hyperparameter, optimizer, training horizon, evaluation metric, AsymRE/TOPR objective, or historical PR #268 DPO semantics, and does not launch the experiment.
'''
marker = '## Recovery seed-chronology implementation closure (2026-09-10)'
if marker not in text:
    scope.write_text(text + block + '\n', encoding='utf-8')
PY

git diff --check
git config user.name 'github-actions[bot]'
git config user.email '41898282+github-actions[bot]@users.noreply.github.com'
git add docs/scopes/E8_MULTITASK_BASELINES_CAPABILITY_DEV.md
git commit -m 'docs: close E8 recovery seed chronology semantics'

python - <<'PY'
from pathlib import Path

path = Path('src/drpo/e8_multitask_exp_tuning.py')
text = path.read_text(encoding='utf-8')

anchor = '''    return reusable, rejected


def _recovery_stage_plan(
'''
insert = '''    return reusable, rejected


def _recovery_reuse_eligible_keys(
    config: Mapping[str, Any],
    cells: Sequence[Cell],
    reusable: Mapping[str, Mapping[str, Any]],
) -> set[str]:
    """Apply the frozen seed chronology to otherwise reusable cell evidence."""

    eligible = set(reusable)
    if not _is_baseline_matrix(config) or _is_engineering_self_test(config):
        return eligible
    cascade_rerun = False
    previous_seed_keys: set[str] | None = None
    for seed_value in config["execution"]["seed_batch_order"]:
        seed = int(seed_value)
        seed_keys = {cell.key for cell in cells if cell.seed == seed}
        if cascade_rerun:
            eligible.difference_update(seed_keys)
            continue
        present = seed_keys & eligible
        if previous_seed_keys is not None and present:
            previous_finish = max(
                float(
                    reusable[key]["scientific_execution_provenance"][
                        "successful_finish_unix"
                    ]
                )
                for key in previous_seed_keys
            )
            current_start = min(
                float(reusable[key]["scientific_execution_provenance"]["started_unix"])
                for key in present
            )
            if previous_finish > current_start:
                eligible.difference_update(seed_keys)
                cascade_rerun = True
                continue
        if present != seed_keys:
            cascade_rerun = True
            continue
        previous_seed_keys = seed_keys
    return eligible


def _recovery_reusable_cell_manifests(
    config: Mapping[str, Any],
    output_root: Path,
) -> tuple[dict[str, dict[str, Any]], dict[str, str]]:
    reusable, rejected = _reusable_cell_manifests(config, output_root)
    cells = build_cells(config)
    eligible = _recovery_reuse_eligible_keys(config, cells, reusable)
    for key in sorted(set(reusable) - eligible):
        rejected[key] = "baseline_seed_chronology_requires_reexecution"
    return {key: reusable[key] for key in sorted(eligible)}, rejected


def _recovery_stage_plan(
'''
if text.count(anchor) != 1:
    raise SystemExit(f'recovery helper anchor count={text.count(anchor)}')
text = text.replace(anchor, insert, 1)

old = '    reusable, rejected = _reusable_cell_manifests(config, output_root)\n'
# The first remaining occurrence is recovery_stage_plan and the second is checkpoint snapshot.
if text.count(old) < 2:
    raise SystemExit(f'recovery reusable call count={text.count(old)}')
text = text.replace(old, '    reusable, rejected = _recovery_reusable_cell_manifests(config, output_root)\n', 1)
# Find checkpoint function and replace within that suffix only.
checkpoint = text.index('def _recovery_checkpoint_snapshot(')
head, tail = text[:checkpoint], text[checkpoint:]
if tail.count(old) < 1:
    raise SystemExit('checkpoint reusable call missing')
tail = tail.replace(old, '    reusable, rejected = _recovery_reusable_cell_manifests(config, output_root)\n', 1)
text = head + tail

old_init = '''    initially_reusable, _ = _reusable_cell_manifests(config, output_root)
    last_checkpoint_count = (len(initially_reusable) // recovery_interval) * recovery_interval
'''
new_init = '''    initially_reusable, initially_rejected = _recovery_reusable_cell_manifests(
        config,
        output_root,
    )
    reuse_eligible_keys = set(initially_reusable)
    last_checkpoint_count = (len(initially_reusable) // recovery_interval) * recovery_interval
'''
if text.count(old_init) != 1:
    raise SystemExit(f'dynamic initial reusable anchor count={text.count(old_init)}')
text = text.replace(old_init, new_init, 1)

old_current = '                        current_reusable, _ = _reusable_cell_manifests(config, output_root)\n'
new_current = '                        current_reusable, _ = _recovery_reusable_cell_manifests(config, output_root)\n'
if text.count(old_current) != 1:
    raise SystemExit(f'checkpoint count anchor={text.count(old_current)}')
text = text.replace(old_current, new_current, 1)

old_worker = '''            reusable_complete = False
            if manifest_path.is_file():
                try:
                    reusable_complete = bool(
                        json.loads(manifest_path.read_text(encoding="utf-8")).get("complete")
                    )
                except (OSError, json.JSONDecodeError):
                    reusable_complete = False
            child_force = force or (
                retry_incomplete and cell_root.exists() and not reusable_complete
            )
            execution_origin = (
                "reused_existing"
                if reusable_complete and not child_force
                else "executed_current_run"
            )
            record(
                {
                    "event": "start",
                    "cell_key": cell.key,
                    "seed": cell.seed,
                    "slot": slot,
                    "gpu_id": gpu_id,
                    "unix_time": time.time(),
                    "retry_incomplete": child_force and not force,
                    "execution_origin": execution_origin,
                }
            )
            result = _run_subprocess_cell(
                config_path=config_path.resolve(),
                output_root=output_root.resolve(),
                base_model_path=base_model_path,
                cell=cell,
                gpu_id=gpu_id,
                force=child_force,
            )
'''
new_worker = '''            cell_exists = cell_root.exists()
            reuse_eligible = cell.key in reuse_eligible_keys
            child_force = force or (
                retry_incomplete and cell_exists and not reuse_eligible
            )
            if reuse_eligible and cell_exists and not child_force:
                execution_origin = "reused_existing"
            elif cell_exists and not child_force:
                execution_origin = "rejected_existing_not_executed"
            else:
                execution_origin = "executed_current_run"
            record(
                {
                    "event": "start",
                    "cell_key": cell.key,
                    "seed": cell.seed,
                    "slot": slot,
                    "gpu_id": gpu_id,
                    "unix_time": time.time(),
                    "retry_incomplete": child_force and not force,
                    "execution_origin": execution_origin,
                }
            )
            if execution_origin == "rejected_existing_not_executed":
                result = {
                    "cell_key": cell.key,
                    "gpu_id": gpu_id,
                    "returncode": 76,
                    "log": str((output_root / "logs" / f"{cell.key}.log").resolve()),
                    "cell_completion_error": (
                        "existing cell is not reuse-eligible; rerun with "
                        "--retry-incomplete or --force: "
                        + initially_rejected.get(
                            cell.key,
                            "existing evidence rejected by recovery eligibility",
                        )
                    ),
                }
            else:
                result = _run_subprocess_cell(
                    config_path=config_path.resolve(),
                    output_root=output_root.resolve(),
                    base_model_path=base_model_path,
                    cell=cell,
                    gpu_id=gpu_id,
                    force=child_force,
                )
'''
if text.count(old_worker) != 1:
    raise SystemExit(f'worker reuse anchor count={text.count(old_worker)}')
text = text.replace(old_worker, new_worker, 1)

old_comment = '''        # Independently audit the append-only event history for the same scheduler run.
        # The summary alone is not sufficient evidence that a later seed batch did not
        # start before the preceding batch had finished successfully.
'''
new_comment = '''        # Independently audit append-only scheduler events for exact cell/origin identity.
        # Scientific seed chronology is proved only by the persisted per-cell execution
        # provenance below; queue-event wall times are not authoritative after recovery.
'''
if text.count(old_comment) != 1:
    raise SystemExit(f'audit comment anchor count={text.count(old_comment)}')
text = text.replace(old_comment, new_comment, 1)

old_semantics = '        "recovery_semantics": "reuse complete identity-checked cells; rerun incomplete cells",\n'
new_semantics = '        "recovery_semantics": "reuse complete identity-and-seed-chronology-checked cells; rerun rejected or incomplete cells",\n'
if text.count(old_semantics) != 1:
    raise SystemExit(f'recovery semantics anchor count={text.count(old_semantics)}')
text = text.replace(old_semantics, new_semantics, 1)
path.write_text(text, encoding='utf-8')

# Add focused regressions without changing scientific fixtures.
test_path = Path('tests/test_e8_multitask_p0.py')
tests = test_path.read_text(encoding='utf-8').rstrip()
addition = r'''


def _synthetic_recovery_manifest(cell, *, start: float, finish: float):
    return {
        "scientific_execution_provenance": {
            "scheduler_run_id": f"prior-{cell.seed}",
            "cell_key": cell.key,
            "seed": cell.seed,
            "started_unix": start,
            "successful_finish_unix": finish,
            "execution_origin": "executed_current_run",
        }
    }


def test_baseline_recovery_global_eligibility_cascades_after_earlier_seed_gap() -> None:
    from drpo import e8_multitask_exp_tuning as exp_tuning

    config = exp_tuning.load_config("configs/e8_multitask_baseline_matrix_formal.yaml")
    cells = exp_tuning.build_cells(config)
    seed_4000 = [cell for cell in cells if cell.seed == 4000]
    seed_5000 = [cell for cell in cells if cell.seed == 5000]
    reusable = {
        cell.key: _synthetic_recovery_manifest(
            cell,
            start=10.0 if cell.seed == 4000 else 30.0,
            finish=20.0 if cell.seed == 4000 else 40.0,
        )
        for cell in cells
    }
    missing = seed_4000[0]
    reusable.pop(missing.key)
    eligible = exp_tuning._recovery_reuse_eligible_keys(config, cells, reusable)
    assert eligible == {cell.key for cell in seed_4000[1:]}
    assert not ({cell.key for cell in seed_5000} & eligible)


def test_baseline_recovery_global_eligibility_reruns_temporally_invalid_later_seed() -> None:
    from drpo import e8_multitask_exp_tuning as exp_tuning

    config = exp_tuning.load_config("configs/e8_multitask_baseline_matrix_formal.yaml")
    cells = exp_tuning.build_cells(config)
    seed_4000 = [cell for cell in cells if cell.seed == 4000]
    seed_5000 = [cell for cell in cells if cell.seed == 5000]
    reusable = {
        cell.key: _synthetic_recovery_manifest(
            cell,
            start=10.0 if cell.seed == 4000 else 15.0,
            finish=20.0 if cell.seed == 4000 else 25.0,
        )
        for cell in cells
    }
    eligible = exp_tuning._recovery_reuse_eligible_keys(config, cells, reusable)
    assert eligible == {cell.key for cell in seed_4000}
    assert not ({cell.key for cell in seed_5000} & eligible)


def test_baseline_recovery_partial_later_seed_keeps_compatible_same_seed_reuse() -> None:
    from drpo import e8_multitask_exp_tuning as exp_tuning

    config = exp_tuning.load_config("configs/e8_multitask_baseline_matrix_formal.yaml")
    cells = exp_tuning.build_cells(config)
    seed_5000 = [cell for cell in cells if cell.seed == 5000]
    reusable = {
        cell.key: _synthetic_recovery_manifest(
            cell,
            start=10.0 if cell.seed == 4000 else 30.0,
            finish=20.0 if cell.seed == 4000 else 40.0,
        )
        for cell in cells
    }
    missing = seed_5000[0]
    reusable.pop(missing.key)
    eligible = exp_tuning._recovery_reuse_eligible_keys(config, cells, reusable)
    assert eligible == set(reusable)


def test_retry_incomplete_forces_complete_manifest_rejected_from_authoritative_reuse(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from drpo import e8_multitask_exp_tuning as exp_tuning

    config = exp_tuning._engineering_self_test_config(
        _baseline_matrix_capability_test_config()
    )
    cells = exp_tuning.build_cells(config)
    victim = cells[0]
    p0.atomic_json(tmp_path / "source_provenance.json", {"source_commit": "f" * 40})
    p0.atomic_json(
        tmp_path / "cells" / victim.key / "cell_manifest.json",
        {
            "experiment_id": exp_tuning.experiment_id(config),
            "config_hash": exp_tuning.stable_config_hash(config),
            "complete": True,
            "evaluation_status": "complete",
            "nan_inf_failure": False,
            "engineering_placeholder_backend": True,
        },
    )
    monkeypatch.delenv("E8_COLDSTART_RECOVERY_PACKAGE", raising=False)
    monkeypatch.setattr(exp_tuning, "_require_calibration_gate", lambda *args, **kwargs: None)
    monkeypatch.setattr(exp_tuning, "_require_liveness_gate", lambda *args, **kwargs: None)
    monkeypatch.setattr(exp_tuning, "_coldstart_completed_task_rows", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        exp_tuning,
        "_recovery_reusable_cell_manifests",
        lambda *args, **kwargs: ({}, {victim.key: "synthetic_reuse_rejection"}),
    )
    observed_force: dict[str, bool] = {}

    def fake_subprocess_cell(**kwargs):
        cell = kwargs["cell"]
        observed_force[cell.key] = bool(kwargs["force"])
        manifest_path = kwargs["output_root"] / "cells" / cell.key / "cell_manifest.json"
        p0.atomic_json(
            manifest_path,
            {
                "experiment_id": exp_tuning.experiment_id(config),
                "config_hash": exp_tuning.stable_config_hash(config),
                "complete": True,
                "evaluation_status": "complete",
                "nan_inf_failure": False,
                "engineering_placeholder_backend": True,
            },
        )
        now = time.time()
        return {
            "cell_key": cell.key,
            "returncode": 0,
            "started_unix": now,
            "finished_unix": now + 0.001,
        }

    monkeypatch.setattr(exp_tuning, "_run_subprocess_cell", fake_subprocess_cell)
    result = exp_tuning.cmd_run_dynamic(
        config,
        tmp_path / "synthetic.yaml",
        tmp_path,
        base_model_path="unused",
        force=False,
        retry_incomplete=True,
    )
    assert result["complete"] is True
    assert observed_force[victim.key] is True
    assert all(observed_force[cell.key] is False for cell in cells[1:])


def test_non_retry_path_never_stamps_rejected_complete_manifest_as_new_execution(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from drpo import e8_multitask_exp_tuning as exp_tuning

    config = exp_tuning._engineering_self_test_config(
        _baseline_matrix_capability_test_config()
    )
    cells = exp_tuning.build_cells(config)
    victim = cells[0]
    p0.atomic_json(tmp_path / "source_provenance.json", {"source_commit": "a" * 40})
    p0.atomic_json(
        tmp_path / "cells" / victim.key / "cell_manifest.json",
        {
            "experiment_id": exp_tuning.experiment_id(config),
            "config_hash": exp_tuning.stable_config_hash(config),
            "complete": True,
            "evaluation_status": "complete",
            "nan_inf_failure": False,
            "engineering_placeholder_backend": True,
        },
    )
    monkeypatch.delenv("E8_COLDSTART_RECOVERY_PACKAGE", raising=False)
    monkeypatch.setattr(exp_tuning, "_require_calibration_gate", lambda *args, **kwargs: None)
    monkeypatch.setattr(exp_tuning, "_require_liveness_gate", lambda *args, **kwargs: None)
    monkeypatch.setattr(exp_tuning, "_coldstart_completed_task_rows", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        exp_tuning,
        "_recovery_reusable_cell_manifests",
        lambda *args, **kwargs: ({}, {victim.key: "synthetic_reuse_rejection"}),
    )
    called = set()

    def fake_subprocess_cell(**kwargs):
        cell = kwargs["cell"]
        called.add(cell.key)
        raise AssertionError("rejected existing cell must not enter subprocess path")

    monkeypatch.setattr(exp_tuning, "_run_subprocess_cell", fake_subprocess_cell)
    with pytest.raises(RuntimeError, match="Cold-start scheduling stopped fail-closed"):
        exp_tuning.cmd_run_dynamic(
            config,
            tmp_path / "synthetic.yaml",
            tmp_path,
            base_model_path="unused",
            force=False,
            retry_incomplete=False,
        )
    assert victim.key not in called
    victim_manifest = json.loads(
        (tmp_path / "cells" / victim.key / "cell_manifest.json").read_text()
    )
    assert "scientific_execution_provenance" not in victim_manifest
'''
marker = 'def test_baseline_recovery_global_eligibility_cascades_after_earlier_seed_gap()'
if marker not in tests:
    tests += addition
    test_path.write_text(tests + '\n', encoding='utf-8')
PY

git diff --check
python -m pytest -q tests/test_e8_multitask_p0.py
ruff check --ignore BLE001,B023 src/drpo/e8_multitask_exp_tuning.py src/drpo/e8_experiment_config.py tests/test_e8_multitask_p0.py
python -m py_compile src/drpo/e8_multitask_exp_tuning.py src/drpo/e8_experiment_config.py tests/test_e8_multitask_p0.py
python scripts/handoff_authority.py verify
python scripts/validate_governance_pipeline_stage_status.py
git diff --check

git add src/drpo/e8_multitask_exp_tuning.py tests/test_e8_multitask_p0.py
git commit -m 'fix: make E8 recovery reuse seed-chronology safe'

git rm -f \
  .github/workflows/tmp_e8_recovery_reuse_fix.yml \
  .github/workflows/tmp_e8_recovery_reuse_fix.sh \
  .github/workflows/tmp_e8_recovery_reuse_fix_v2.sh \
  .github/workflows/tmp_e8_recovery_reuse_fix_v2.yml || true
git commit -m 'chore: remove temporary E8 recovery audit helpers'
git push origin HEAD:dev/e8-multitask-baselines-capability-01
