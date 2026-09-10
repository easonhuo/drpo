#!/usr/bin/env bash
set -euo pipefail

python - <<'PY'
from pathlib import Path

path = Path('src/drpo/e8_multitask_exp_tuning.py')
text = path.read_text(encoding='utf-8')

anchor = 'def cmd_run_dynamic(\n'
helper = '''def _recovery_reuse_eligible_keys(
    config: Mapping[str, Any],
    cells: Sequence[Cell],
    reusable_keys: set[str],
) -> set[str]:
    """Preserve reusable evidence without violating the formal seed chronology."""

    eligible = set(reusable_keys)
    if not _is_baseline_matrix(config) or _is_engineering_self_test(config):
        return eligible
    original = set(reusable_keys)
    rerun_later_seeds = False
    for seed_value in config["execution"]["seed_batch_order"]:
        seed = int(seed_value)
        seed_keys = {cell.key for cell in cells if cell.seed == seed}
        if rerun_later_seeds:
            eligible.difference_update(seed_keys)
        if not seed_keys.issubset(original):
            rerun_later_seeds = True
    return eligible


def cmd_run_dynamic(
'''
if text.count(anchor) != 1:
    raise SystemExit(f'cmd_run_dynamic anchor count={text.count(anchor)}')
text = text.replace(anchor, helper, 1)

old_init = '''    initially_reusable, _ = _reusable_cell_manifests(config, output_root)
    last_checkpoint_count = (len(initially_reusable) // recovery_interval) * recovery_interval
'''
new_init = '''    initially_reusable, initially_rejected = _reusable_cell_manifests(config, output_root)
    reuse_eligible_keys = _recovery_reuse_eligible_keys(
        config,
        cells,
        set(initially_reusable),
    )
    last_checkpoint_count = (len(initially_reusable) // recovery_interval) * recovery_interval
'''
if text.count(old_init) != 1:
    raise SystemExit(f'initial reusable anchor count={text.count(old_init)}')
text = text.replace(old_init, new_init, 1)

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
                            "later-seed reuse invalidated by seed-barrier recovery ordering",
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
path.write_text(text, encoding='utf-8')

test_path = Path('tests/test_e8_multitask_p0.py')
tests = test_path.read_text(encoding='utf-8').rstrip() + '''


def test_baseline_recovery_invalid_earlier_seed_forces_later_seed_reexecution() -> None:
    from drpo import e8_multitask_exp_tuning as exp_tuning

    config = exp_tuning.load_config("configs/e8_multitask_baseline_matrix_formal.yaml")
    cells = exp_tuning.build_cells(config)
    all_keys = {cell.key for cell in cells}
    seed_4000 = {cell.key for cell in cells if cell.seed == 4000}
    seed_5000 = {cell.key for cell in cells if cell.seed == 5000}
    victim_4000 = next(iter(seed_4000))
    eligible = exp_tuning._recovery_reuse_eligible_keys(
        config, cells, all_keys - {victim_4000}
    )
    assert eligible == seed_4000 - {victim_4000}
    victim_5000 = next(iter(seed_5000))
    eligible = exp_tuning._recovery_reuse_eligible_keys(
        config, cells, all_keys - {victim_5000}
    )
    assert eligible == all_keys - {victim_5000}


def test_retry_incomplete_forces_complete_manifest_rejected_from_reuse(
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
        "_reusable_cell_manifests",
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
        "_reusable_cell_manifests",
        lambda *args, **kwargs: ({}, {victim.key: "synthetic_reuse_rejection"}),
    )
    called = set()

    def fake_subprocess_cell(**kwargs):
        cell = kwargs["cell"]
        called.add(cell.key)
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
test_path.write_text(tests, encoding='utf-8')
PY

git diff --check
python -m pytest -q tests/test_e8_multitask_p0.py
ruff check src/drpo/e8_multitask_exp_tuning.py src/drpo/e8_experiment_config.py tests/test_e8_multitask_p0.py
python -m py_compile src/drpo/e8_multitask_exp_tuning.py src/drpo/e8_experiment_config.py tests/test_e8_multitask_p0.py
python scripts/handoff_authority.py verify
python scripts/validate_governance_pipeline_stage_status.py
git diff --check

git config user.name 'github-actions[bot]'
git config user.email '41898282+github-actions[bot]@users.noreply.github.com'
git add src/drpo/e8_multitask_exp_tuning.py tests/test_e8_multitask_p0.py
git commit -m 'fix: make E8 recovery reuse chronology-safe'
git rm .github/workflows/tmp_e8_recovery_reuse_fix.yml .github/workflows/tmp_e8_recovery_reuse_fix.sh
git commit -m 'chore: remove temporary E8 recovery audit helpers'
git push origin HEAD:dev/e8-multitask-baselines-capability-01
