#!/usr/bin/env bash
set -euo pipefail

# Reuse the already-debugged v4 transport to materialize the recovery repair in
# this runner workspace.  At the current branch state v4 is expected to stop at
# Ruff after its focused pytest suite, leaving the documented source/test edits
# uncommitted for the additional audit fixes below.
set +e
bash .github/workflows/tmp_e8_recovery_reuse_fix_v4.sh
V4_RC=$?
set -e
if [[ "$V4_RC" -eq 0 ]]; then
  echo "unexpected: v4 completed fully; refusing to layer v5 blindly" >&2
  exit 91
fi

python - <<'PY'
from pathlib import Path

scope = Path('docs/scopes/E8_MULTITASK_BASELINES_CAPABILITY_DEV.md')
text = scope.read_text(encoding='utf-8')
old = (
    "Recovery reuse therefore uses one shared global eligibility rule across recovery planning, "
    "checkpoint snapshots, import, and dynamic scheduling. Individually valid cells in the "
    "currently incomplete seed may still be reused when their persisted chronology is compatible "
    "with the preceding fully reusable seed. Once a seed has any non-reusable cell, every later "
    "seed's older cells are ineligible for reuse and must execute again; if a fully/partially "
    "reusable later seed already starts before the preceding reusable seed finished, that seed and "
    "all later seeds are likewise ineligible."
)
new = (
    "Recovery reuse therefore uses one shared global eligibility rule across recovery planning, "
    "checkpoint snapshots, import, and dynamic scheduling. Within the first incomplete seed, "
    "individually valid cells remain reusable only when their persisted starts are compatible with "
    "the preceding fully reusable seed; incompatible cells are rerun. Once that seed remains "
    "incomplete after this per-cell chronology filtering, every later seed's older cells are "
    "ineligible for reuse and must execute again."
)
if text.count(old) != 1:
    raise SystemExit(f'scope chronology refinement anchor count={text.count(old)}')
scope.write_text(text.replace(old, new, 1).rstrip() + '\n', encoding='utf-8')

path = Path('src/drpo/e8_multitask_exp_tuning.py')
text = path.read_text(encoding='utf-8')

start = text.index('def _recovery_reuse_eligible_keys(\n')
end = text.index('\n\ndef _recovery_reusable_cell_manifests(', start)
helper = '''def _recovery_reuse_eligible_keys(
    config: Mapping[str, Any],
    cells: Sequence[Cell],
    reusable: Mapping[str, Mapping[str, Any]],
) -> set[str]:
    """Apply the frozen seed chronology to otherwise reusable cell evidence."""

    if not _is_baseline_matrix(config) or _is_engineering_self_test(config):
        return set(reusable)
    eligible: set[str] = set()
    cascade_rerun = False
    previous_finish: float | None = None
    for seed_value in config["execution"]["seed_batch_order"]:
        seed = int(seed_value)
        seed_keys = {cell.key for cell in cells if cell.seed == seed}
        if cascade_rerun:
            continue
        present = seed_keys & set(reusable)
        compatible = set(present)
        if previous_finish is not None:
            compatible = {
                key
                for key in present
                if float(
                    reusable[key]["scientific_execution_provenance"]["started_unix"]
                )
                >= previous_finish
            }
        eligible.update(compatible)
        if compatible != seed_keys:
            cascade_rerun = True
            continue
        previous_finish = max(
            float(
                reusable[key]["scientific_execution_provenance"]["successful_finish_unix"]
            )
            for key in seed_keys
        )
    return eligible
'''
text = text[:start] + helper + text[end:]

queue_start = text.index('def _audit_engineering_queue(\n')
queue_end = text.index('\n\ndef cmd_engineering_self_test(', queue_start)
queue_audit = '''def _audit_engineering_queue(
    config: Mapping[str, Any],
    output_root: Path,
    scheduler: Mapping[str, Any],
) -> dict[str, Any]:
    cells = build_cells(config)
    events = [
        json.loads(line)
        for line in (output_root / "scheduler" / "queue_events.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]
    events = [row for row in events if row["scheduler_run_id"] == scheduler["scheduler_run_id"]]
    active_by_gpu = {int(gpu): 0 for gpu in config["execution"]["gpu_ids"]}
    maximum_by_gpu = dict(active_by_gpu)
    starts: dict[str, float] = {}
    finishes: dict[str, float] = {}
    first_start_by_slot: dict[int, str] = {}
    replacement_keys: set[str] = set()
    for event in events:
        gpu_id = int(event["gpu_id"])
        cell_key = str(event["cell_key"])
        if event["event"] == "start":
            active_by_gpu[gpu_id] += 1
            maximum_by_gpu[gpu_id] = max(maximum_by_gpu[gpu_id], active_by_gpu[gpu_id])
            starts[cell_key] = float(event["unix_time"])
            slot = int(event["slot"])
            if slot not in first_start_by_slot:
                first_start_by_slot[slot] = cell_key
            else:
                replacement_keys.add(cell_key)
        elif event["event"] == "finish":
            active_by_gpu[gpu_id] -= 1
            finishes[cell_key] = float(event["unix_time"])
    expected_keys = {cell.key for cell in cells}
    if set(starts) != expected_keys or set(finishes) != expected_keys:
        raise RuntimeError(f"Engineering queue did not observe all {len(cells)} starts/finishes")
    slots_per_gpu = int(config["execution"]["slots_per_gpu"])
    if (
        any(value != 0 for value in active_by_gpu.values())
        or max(maximum_by_gpu.values()) > slots_per_gpu
    ):
        raise RuntimeError("Engineering queue exceeded the declared per-GPU capacity")
    initial_keys = set(first_start_by_slot.values())
    if not replacement_keys:
        raise RuntimeError("Engineering queue requires replacement cells to audit dynamic refill")
    finished_initial: set[str] = set()
    dynamic_refill_observed = False
    for event in events:
        cell_key = str(event["cell_key"])
        if event["event"] == "finish" and cell_key in initial_keys:
            finished_initial.add(cell_key)
        elif (
            event["event"] == "start"
            and cell_key in replacement_keys
            and len(finished_initial) < len(initial_keys)
        ):
            dynamic_refill_observed = True
            break
    if not dynamic_refill_observed:
        raise RuntimeError("Engineering queue did not dynamically refill before initial slots drained")
    return {
        "all_cells_observed": True,
        "maximum_active_by_gpu": maximum_by_gpu,
        "dynamic_refill_observed": True,
        "nominal_batch_barrier_absent": True,
        "nominal_batch_count": len(build_waves(config)),
        "slots_per_gpu": slots_per_gpu,
    }
'''
text = text[:queue_start] + queue_audit + text[queue_end:]

anchor = '''    expected = [cell for cell in build_cells(config) if cell.task == task]
    if not expected:
        return None
    expected_hash = stable_config_hash(config)
'''
replacement = '''    expected = [cell for cell in build_cells(config) if cell.task == task]
    if not expected:
        return None
    if _is_baseline_matrix(config) and not _is_engineering_self_test(config):
        reusable, _ = _recovery_reusable_cell_manifests(config, output_root)
        reusable_keys = set(reusable)
        if any(cell.key not in reusable_keys for cell in expected):
            return None
    expected_hash = stable_config_hash(config)
'''
if text.count(anchor) != 1:
    raise SystemExit(f'task-result recovery eligibility anchor count={text.count(anchor)}')
text = text.replace(anchor, replacement, 1)
path.write_text(text.rstrip() + '\n', encoding='utf-8')

# Add focused regressions for the two deeper recovery cases found after v4.
test_path = Path('tests/test_e8_multitask_p0.py')
tests = test_path.read_text(encoding='utf-8').rstrip()
addition = r'''


def test_baseline_recovery_keeps_compatible_cells_within_temporally_partial_seed() -> None:
    from drpo import e8_multitask_exp_tuning as exp_tuning

    config = exp_tuning.load_config("configs/e8_multitask_baseline_matrix_formal.yaml")
    cells = exp_tuning.build_cells(config)
    seed_4000 = [cell for cell in cells if cell.seed == 4000]
    seed_5000 = [cell for cell in cells if cell.seed == 5000]
    stale = seed_5000[0]
    reusable = {}
    for cell in cells:
        if cell.seed == 4000:
            start, finish = 10.0, 20.0
        elif cell.key == stale.key:
            start, finish = 15.0, 25.0
        else:
            start, finish = 30.0, 40.0
        reusable[cell.key] = _synthetic_recovery_manifest(
            cell,
            start=start,
            finish=finish,
        )
    eligible = exp_tuning._recovery_reuse_eligible_keys(config, cells, reusable)
    assert {cell.key for cell in seed_4000} <= eligible
    assert stale.key not in eligible
    assert {cell.key for cell in seed_5000[1:]} <= eligible


def test_baseline_task_rows_wait_for_global_recovery_eligibility(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from drpo import e8_multitask_exp_tuning as exp_tuning

    config = exp_tuning.load_config("configs/e8_multitask_baseline_matrix_formal.yaml")
    task = str(config["suite"]["tasks"][0])
    expected = [cell for cell in exp_tuning.build_cells(config) if cell.task == task]
    victim = expected[0]
    eligible = {
        cell.key: {}
        for cell in exp_tuning.build_cells(config)
        if cell.key != victim.key
    }
    monkeypatch.setattr(
        exp_tuning,
        "_recovery_reusable_cell_manifests",
        lambda *args, **kwargs: (eligible, {victim.key: "synthetic_reuse_rejection"}),
    )
    assert exp_tuning._coldstart_completed_task_rows(config, tmp_path, task) is None
'''
marker = 'def test_baseline_recovery_keeps_compatible_cells_within_temporally_partial_seed()'
if marker not in tests:
    tests += addition
    test_path.write_text(tests.rstrip() + '\n', encoding='utf-8')
PY

# The v4 transport committed the first documentation closure before touching
# science code. Commit the clarified recovery semantics before the code repair.
git diff --check
git add docs/scopes/E8_MULTITASK_BASELINES_CAPABILITY_DEV.md
if ! git diff --cached --quiet; then
  git commit -m 'docs: clarify E8 partial-seed recovery reuse'
fi

python -m pytest -q tests/test_e8_multitask_p0.py
ruff check --ignore BLE001,B023 src/drpo/e8_multitask_exp_tuning.py src/drpo/e8_experiment_config.py tests/test_e8_multitask_p0.py
python -m py_compile src/drpo/e8_multitask_exp_tuning.py src/drpo/e8_experiment_config.py tests/test_e8_multitask_p0.py
python scripts/handoff_authority.py verify
python scripts/validate_governance_pipeline_stage_status.py
git diff --check

git add src/drpo/e8_multitask_exp_tuning.py tests/test_e8_multitask_p0.py
git commit -m 'fix: make E8 recovery chronology reuse consistent'

git rm -f \
  .github/workflows/tmp_e8_recovery_reuse_fix.yml \
  .github/workflows/tmp_e8_recovery_reuse_fix.sh \
  .github/workflows/tmp_e8_recovery_reuse_fix_v2.sh \
  .github/workflows/tmp_e8_recovery_reuse_fix_v2.yml \
  .github/workflows/tmp_e8_recovery_reuse_fix_v3.sh \
  .github/workflows/tmp_e8_recovery_reuse_fix_v3.yml \
  .github/workflows/tmp_e8_recovery_reuse_fix_v4.sh \
  .github/workflows/tmp_e8_recovery_reuse_fix_v4.yml \
  .github/workflows/tmp_e8_recovery_reuse_fix_v5.sh \
  .github/workflows/tmp_e8_recovery_reuse_fix_v5.yml || true
git commit -m 'chore: remove temporary E8 recovery repair transport'
git push origin HEAD:dev/e8-multitask-baselines-capability-01
