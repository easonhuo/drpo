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
