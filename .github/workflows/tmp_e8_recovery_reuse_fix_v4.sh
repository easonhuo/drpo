#!/usr/bin/env bash
set -euo pipefail

python - <<'PY'
from pathlib import Path

path = Path('.github/workflows/tmp_e8_recovery_reuse_fix_v2.sh')
text = path.read_text(encoding='utf-8')

# Keep docs/tests normalized to one terminal newline.
replacements = {
    "    scope.write_text(text + block + '\\n', encoding='utf-8')": "    scope.write_text((text + block).rstrip() + '\\n', encoding='utf-8')",
    "    test_path.write_text(tests + '\\n', encoding='utf-8')": "    test_path.write_text(tests.rstrip() + '\\n', encoding='utf-8')",
}
for old, new in replacements.items():
    if text.count(old) != 1:
        raise SystemExit(f'normalization anchor count={text.count(old)} for {old}')
    text = text.replace(old, new, 1)

# The previous generator inserted _recovery_reusable_cell_manifests and then
# globally replaced the first _reusable_cell_manifests call, accidentally
# rewriting the new helper into a recursive self-call. Restrict both rewrites
# to the intended function suffixes.
old = '''old = '    reusable, rejected = _reusable_cell_manifests(config, output_root)\\n'
# The first remaining occurrence is recovery_stage_plan and the second is checkpoint snapshot.
if text.count(old) < 2:
    raise SystemExit(f'recovery reusable call count={text.count(old)}')
text = text.replace(old, '    reusable, rejected = _recovery_reusable_cell_manifests(config, output_root)\\n', 1)
# Find checkpoint function and replace within that suffix only.
checkpoint = text.index('def _recovery_checkpoint_snapshot(')
head, tail = text[:checkpoint], text[checkpoint:]
if tail.count(old) < 1:
    raise SystemExit('checkpoint reusable call missing')
tail = tail.replace(old, '    reusable, rejected = _recovery_reusable_cell_manifests(config, output_root)\\n', 1)
text = head + tail
'''
new = '''old = '    reusable, rejected = _reusable_cell_manifests(config, output_root)\\n'
stage_plan = text.index('def _recovery_stage_plan(')
head, tail = text[:stage_plan], text[stage_plan:]
if tail.count(old) < 1:
    raise SystemExit('recovery stage-plan reusable call missing')
tail = tail.replace(
    old,
    '    reusable, rejected = _recovery_reusable_cell_manifests(config, output_root)\\n',
    1,
)
text = head + tail
checkpoint = text.index('def _recovery_checkpoint_snapshot(')
head, tail = text[:checkpoint], text[checkpoint:]
if tail.count(old) < 1:
    raise SystemExit('checkpoint reusable call missing')
tail = tail.replace(
    old,
    '    reusable, rejected = _recovery_reusable_cell_manifests(config, output_root)\\n',
    1,
)
text = head + tail
'''
if text.count(old) != 1:
    raise SystemExit(f'scoped helper rewrite anchor count={text.count(old)}')
text = text.replace(old, new, 1)

# In the non-retry regression, allow unrelated missing cells that were already
# dispatched concurrently to complete. The assertion is specifically that the
# rejected existing victim never reaches the subprocess path.
old_fake = '''    def fake_subprocess_cell(**kwargs):
        cell = kwargs["cell"]
        called.add(cell.key)
        raise AssertionError("rejected existing cell must not enter subprocess path")
'''
new_fake = '''    def fake_subprocess_cell(**kwargs):
        cell = kwargs["cell"]
        called.add(cell.key)
        if cell.key == victim.key:
            raise AssertionError("rejected existing cell must not enter subprocess path")
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
'''
if text.count(old_fake) != 1:
    raise SystemExit(f'non-retry fake anchor count={text.count(old_fake)}')
text = text.replace(old_fake, new_fake, 1)

# Replace the engineering queue audit with an event-order audit. The old
# version assumed build_cells()[:16] were the first cells actually dispatched;
# recovery/reuse can change the observed start order, so that assumption can
# produce a false failure. Event order is authoritative for this engineering
# liveness check, while scientific seed chronology remains proven only from
# persisted scientific_execution_provenance.
old_source_tail = '''if text.count(old_semantics) != 1:
    raise SystemExit(f'recovery semantics anchor count={text.count(old_semantics)}')
text = text.replace(old_semantics, new_semantics, 1)
path.write_text(text, encoding='utf-8')
'''
new_source_tail = """if text.count(old_semantics) != 1:
    raise SystemExit(f'recovery semantics anchor count={text.count(old_semantics)}')
text = text.replace(old_semantics, new_semantics, 1)

queue_audit_start = text.index('def _audit_engineering_queue(\\n')
queue_audit_end = text.index('\\n\\ndef cmd_engineering_self_test(', queue_audit_start)
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
    start_order: list[str] = []
    for event in events:
        gpu_id = int(event["gpu_id"])
        cell_key = str(event["cell_key"])
        if event["event"] == "start":
            if cell_key in starts:
                raise RuntimeError(f"Engineering queue observed duplicate start: {cell_key}")
            active_by_gpu[gpu_id] += 1
            maximum_by_gpu[gpu_id] = max(maximum_by_gpu[gpu_id], active_by_gpu[gpu_id])
            starts[cell_key] = float(event["unix_time"])
            start_order.append(cell_key)
        elif event["event"] == "finish":
            if cell_key not in starts:
                raise RuntimeError(f"Engineering queue observed finish before start: {cell_key}")
            if cell_key in finishes:
                raise RuntimeError(f"Engineering queue observed duplicate finish: {cell_key}")
            active_by_gpu[gpu_id] -= 1
            if active_by_gpu[gpu_id] < 0:
                raise RuntimeError("Engineering queue active-slot accounting became negative")
            finishes[cell_key] = float(event["unix_time"])
        else:
            raise RuntimeError(f"Engineering queue observed unknown event: {event.get('event')}")
    expected_keys = {cell.key for cell in cells}
    if set(starts) != expected_keys or set(finishes) != expected_keys:
        raise RuntimeError(f"Engineering queue did not observe all {len(cells)} starts/finishes")
    slots_per_gpu = int(config["execution"]["slots_per_gpu"])
    if (
        any(value != 0 for value in active_by_gpu.values())
        or max(maximum_by_gpu.values()) > slots_per_gpu
    ):
        raise RuntimeError("Engineering queue exceeded the declared per-GPU capacity")
    slot_count = int(config["execution"]["max_concurrent_cells"])
    initial_order = start_order[:slot_count]
    replacement_order = start_order[slot_count:]
    if len(initial_order) != slot_count or not replacement_order:
        raise RuntimeError("Engineering queue requires replacement cells to audit dynamic refill")
    initial_keys = set(initial_order)
    replacement_keys = set(replacement_order)
    finished_initial: set[str] = set()
    dynamic_refill_observed = False
    for event in events:
        cell_key = str(event["cell_key"])
        if event["event"] == "finish" and cell_key in initial_keys:
            finished_initial.add(cell_key)
        elif event["event"] == "start" and cell_key in replacement_keys:
            if len(finished_initial) < len(initial_keys):
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
text = text[:queue_audit_start] + queue_audit + text[queue_audit_end:]
path.write_text(text, encoding='utf-8')
"""
if text.count(old_source_tail) != 1:
    raise SystemExit(f'queue audit injection anchor count={text.count(old_source_tail)}')
text = text.replace(old_source_tail, new_source_tail, 1)

# Remove every temporary transport workflow/helper once the repaired code has
# passed all local gates and is ready to push.
old_cleanup = '''git rm -f \\
  .github/workflows/tmp_e8_recovery_reuse_fix.yml \\
  .github/workflows/tmp_e8_recovery_reuse_fix.sh \\
  .github/workflows/tmp_e8_recovery_reuse_fix_v2.sh \\
  .github/workflows/tmp_e8_recovery_reuse_fix_v2.yml || true
'''
new_cleanup = '''git rm -f \\
  .github/workflows/tmp_e8_recovery_reuse_fix.yml \\
  .github/workflows/tmp_e8_recovery_reuse_fix.sh \\
  .github/workflows/tmp_e8_recovery_reuse_fix_v2.sh \\
  .github/workflows/tmp_e8_recovery_reuse_fix_v2.yml \\
  .github/workflows/tmp_e8_recovery_reuse_fix_v3.sh \\
  .github/workflows/tmp_e8_recovery_reuse_fix_v3.yml \\
  .github/workflows/tmp_e8_recovery_reuse_fix_v4.sh \\
  .github/workflows/tmp_e8_recovery_reuse_fix_v4.yml || true
'''
if text.count(old_cleanup) != 1:
    raise SystemExit(f'cleanup anchor count={text.count(old_cleanup)}')
text = text.replace(old_cleanup, new_cleanup, 1)

path.write_text(text.rstrip() + '\n', encoding='utf-8')
PY

bash .github/workflows/tmp_e8_recovery_reuse_fix_v2.sh
