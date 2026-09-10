#!/usr/bin/env bash
set -euo pipefail

# Materialize the reviewed v5 repair.  At the current branch state v5 is known
# to stop only in its newly added task-result regression because that test
# accidentally selected Countdown, which is reused externally and therefore
# has no baseline-matrix cell.
set +e
bash .github/workflows/tmp_e8_recovery_reuse_fix_v5.sh
V5_RC=$?
set -e
if [[ "$V5_RC" -eq 0 ]]; then
  echo "unexpected: v5 completed fully; refusing to layer v6 blindly" >&2
  exit 91
fi

python - <<'PY'
from pathlib import Path

path = Path('tests/test_e8_multitask_p0.py')
text = path.read_text(encoding='utf-8')
old = '    task = str(config["suite"]["tasks"][0])\n'
new = '    task = str(config["suite"]["p0_tasks"][0])\n'
# Only the newly added recovery regression should use this exact selector.
if text.count(old) != 1:
    raise SystemExit(f'baseline task-result test selector anchor count={text.count(old)}')
path.write_text(text.replace(old, new, 1).rstrip() + '\n', encoding='utf-8')
PY

git diff --check
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
  .github/workflows/tmp_e8_recovery_reuse_fix_v5.yml \
  .github/workflows/tmp_e8_recovery_reuse_fix_v6.sh \
  .github/workflows/tmp_e8_recovery_reuse_fix_v6.yml || true
git commit -m 'chore: remove temporary E8 recovery repair transport'
git push origin HEAD:dev/e8-multitask-baselines-capability-01
