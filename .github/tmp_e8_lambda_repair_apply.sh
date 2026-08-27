#!/usr/bin/env bash
set -euo pipefail

BASE="8bdd07590f155ad26bc8cfbd641d40647eab57d2"
SOURCE="71b33ea5a783660008e495a34eda7757bee757d1"
MAIN_WITH_ANCHOR="5260ca58ecb67509838a1e5480f6dcaafdb7a773"
BRANCH="dev/e8-multitask-lambda-completion-repair-02"

# This branch must descend from the restored cold-start execution tree.
git merge-base --is-ancestor "${BASE}" HEAD
git cat-file -e "${SOURCE}^{commit}"
git cat-file -e "${MAIN_WITH_ANCHOR}^{commit}"

python - <<'PY'
import subprocess
allowed = {
    'docs/experiments/E8_MULTITASK_LAMBDA_COMPLETION_PROTOCOL.md',
    '.github/tmp_e8_lambda_repair_apply.sh',
    '.github/tmp_e8_lambda_tests.txt',
    '.github/workflows/tmp-e8-lambda-repair-run.yml',
}
changed = set(subprocess.check_output(
    ['git', 'diff', '--name-only', '8bdd07590f155ad26bc8cfbd641d40647eab57d2..HEAD'],
    text=True,
).splitlines())
extra = changed - allowed
assert not extra, sorted(extra)
PY

# Apply only the reviewed successor-specific tuning delta from 71b33. The
# restored execution tree intentionally rejects the stale hard-wave wording.
git diff "${SOURCE}^" "${SOURCE}" -- src/drpo/e8_multitask_exp_tuning.py > /tmp/e8_source.patch
if git apply --reject --whitespace=nowarn /tmp/e8_source.patch; then
  echo "ERROR: expected exactly one build_waves reject against restored cold-start" >&2
  exit 2
fi
REJECT="src/drpo/e8_multitask_exp_tuning.py.rej"
[[ -f "${REJECT}" ]]
[[ "$(grep -c '^@@' "${REJECT}")" -eq 1 ]]
grep -F 'def build_waves' "${REJECT}"
rm "${REJECT}"

python - <<'PY'
from pathlib import Path
path = Path('src/drpo/e8_multitask_exp_tuning.py')
text = path.read_text(encoding='utf-8')
old = '''        if len(waves) != 13 or any(len(wave) != 16 for wave in waves):
            raise AssertionError("Cold-start nominal geometry must be 13 exact 16-cell batches")
'''
new = '''        if (
            not waves
            or len(waves) != int(config["execution"]["expected_waves"])
            or any(len(wave) != capacity for wave in waves[:-1])
            or not 0 < len(waves[-1]) <= capacity
        ):
            raise AssertionError(
                "Cold-start nominal batches must fill capacity except possibly the final batch"
            )
'''
assert text.count(old) == 1, text.count(old)
text = text.replace(old, new, 1)

# PR #328's minimal lambda-only transport change is required in addition to
# 71b33: lambda identity must be accepted before optional rho provenance.
old = '''        if self.rho is None:
            raise AssertionError("Exponential cell requires rho")
        if self.lambda_value is not None:
            tag = f"{self.lambda_value:.12g}".replace(".", "p")
            return f"{self.task}__exp_lambda{tag}__seed{self.seed}"
'''
new = '''        if self.lambda_value is not None:
            tag = f"{self.lambda_value:.12g}".replace(".", "p")
            return f"{self.task}__exp_lambda{tag}__seed{self.seed}"
        if self.rho is None:
            raise AssertionError("Exponential cell requires rho or lambda")
'''
assert text.count(old) == 1, text.count(old)
path.write_text(text.replace(old, new, 1), encoding='utf-8')
PY

# Add the reviewed 199-cell matrix, but restore the two stale-main execution
# fields to the known cold-start execution behavior.
git show "${SOURCE}:configs/e8_multitask_exp_lambda_completion.yaml" > configs/e8_multitask_exp_lambda_completion.yaml
python - <<'PY'
from pathlib import Path
path = Path('configs/e8_multitask_exp_lambda_completion.yaml')
text = path.read_text(encoding='utf-8')
old = '    selection: evenly_spaced_reference_rank_including_extremes\n'
new = '    selection: source_p0_error_class_sequence_then_within_class_reference_rank_spread\n'
assert text.count(old) == 1
text = text.replace(old, new, 1)
old = '  wave_barriers: true\n'
new = '  wave_barriers: false\n'
assert text.count(old) == 1
path.write_text(text.replace(old, new, 1), encoding='utf-8')
PY

# Preserve the immutable historical curve anchor needed for later concatenation.
ANCHOR="experiments/results/e8_multitask_exp_coldstart_20260820_02/CURVE_ANCHOR.csv"
mkdir -p "$(dirname "${ANCHOR}")"
git show "${MAIN_WITH_ANCHOR}:${ANCHOR}" > "${ANCHOR}"

# Parameterize only experiment/config identity so the successor reuses the
# restored cold-start bootstrap, recovery, scheduler and task interface.
python - <<'PY'
from pathlib import Path

bootstrap = Path('scripts/bootstrap_e8_multitask_exp_coldstart.sh')
text = bootstrap.read_text(encoding='utf-8')
old = 'EXPERIMENT_ID="EXT-C-E8-MULTITASK-EXP-COLDSTART-01"\n'
new = 'EXPERIMENT_ID="${E8_COLDSTART_EXPERIMENT_ID:-EXT-C-E8-MULTITASK-EXP-COLDSTART-01}"\n'
assert text.count(old) == 1
bootstrap.write_text(text.replace(old, new, 1), encoding='utf-8')

runner = Path('scripts/run_e8_multitask_exp_coldstart.sh')
text = runner.read_text(encoding='utf-8')
assert text.count(old) == 1
text = text.replace(old, new, 1)

anchor = '''fail() {
  echo "ERROR: $*" >&2
  exit 2
}

case "${RUN_CLASS}" in
'''
replacement = '''fail() {
  echo "ERROR: $*" >&2
  exit 2
}

resolve_config_repo_path() {
  command -v python3 >/dev/null || fail "python3 is unavailable"
  local resolved
  resolved="$(python3 - "${ROOT_DIR}" "${CONFIG_PATH}" <<'PY_CONFIG'
from pathlib import Path
import sys
root = Path(sys.argv[1]).resolve()
candidate = Path(sys.argv[2])
if not candidate.is_absolute():
    candidate = root / candidate
path = candidate.resolve()
try:
    relative = path.relative_to(root)
except ValueError as exc:
    raise SystemExit(f"config path escapes repository: {path}") from exc
if not path.is_file():
    raise SystemExit(f"config file is missing: {path}")
print(relative.as_posix())
PY_CONFIG
)" || fail "config path must resolve to a tracked file inside the repository: ${CONFIG_PATH}"
  CONFIG_REPO_PATH="${resolved}"
  CONFIG_PATH="${ROOT_DIR}/${CONFIG_REPO_PATH}"
  export CONFIG_REPO_PATH CONFIG_PATH
}

resolve_config_repo_path

case "${RUN_CLASS}" in
'''
assert text.count(anchor) == 1, text.count(anchor)
text = text.replace(anchor, replacement, 1)

clean_anchor = '''  [[ -z "$(git -C "${ROOT_DIR}" status --porcelain=v1 --untracked-files=all)" ]] || \\
    fail "source checkout must be fully clean; keep runtime files outside the repository"
}
'''
clean_replacement = '''  [[ -z "$(git -C "${ROOT_DIR}" status --porcelain=v1 --untracked-files=all)" ]] || \\
    fail "source checkout must be fully clean; keep runtime files outside the repository"
  git -C "${ROOT_DIR}" cat-file -e "${EXPECTED_COMMIT}:${CONFIG_REPO_PATH}" 2>/dev/null || \\
    fail "runtime config is unavailable at launch commit ${EXPECTED_COMMIT}: ${CONFIG_REPO_PATH}"
}
'''
assert text.count(clean_anchor) == 1, text.count(clean_anchor)
text = text.replace(clean_anchor, clean_replacement, 1)

literal = '--source-file configs/e8_multitask_exp_coldstart.yaml'
assert text.count(literal) == 3, text.count(literal)
runner.write_text(text.replace(literal, '--source-file "${CONFIG_REPO_PATH}"'), encoding='utf-8')
PY

cat > scripts/run_e8_multitask_exp_lambda_completion.sh <<'SH'
#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export E8_COLDSTART_EXPERIMENT_ID="EXT-C-E8-MULTITASK-EXP-LAMBDA-COMPLETION-01"
export E8_COLDSTART_CONFIG="${ROOT_DIR}/configs/e8_multitask_exp_lambda_completion.yaml"
export E8_COLDSTART_RUN_ID="${E8_COLDSTART_RUN_ID:-E8_MULTITASK_EXP_LAMBDA_COMPLETION_01}"
export E8_COLDSTART_RUN_CLASS="${E8_COLDSTART_RUN_CLASS:-pilot}"
export E8_COLDSTART_REQUIRE_ORIGIN_MAIN="${E8_COLDSTART_REQUIRE_ORIGIN_MAIN:-0}"
if [[ -z "${E8_COLDSTART_TARGET_REF:-}" ]]; then
  branch="$(git -C "${ROOT_DIR}" rev-parse --abbrev-ref HEAD)"
  [[ "${branch}" != "HEAD" ]] || {
    echo "ERROR: set E8_COLDSTART_TARGET_REF from a detached checkout" >&2
    exit 2
  }
  export E8_COLDSTART_TARGET_REF="refs/heads/${branch}"
fi
export E8_COLDSTART_BOOTSTRAP_ROOT="${E8_COLDSTART_BOOTSTRAP_ROOT:-${ROOT_DIR}/../drpo-e8-lambda-completion-${1:-full}}"
exec bash "${ROOT_DIR}/scripts/bootstrap_e8_multitask_exp_coldstart.sh" "$@"
SH
chmod +x scripts/run_e8_multitask_exp_lambda_completion.sh

# Replace the stale superseded-runbook test, then append the successor tests.
python - <<'PY'
from pathlib import Path
path = Path('tests/test_e8_multitask_p0.py')
text = path.read_text(encoding='utf-8')
marker = 'def test_coldstart_runbook_embeds_bootstrap_and_current_protocol() -> None:'
start = text.find(marker)
assert start >= 0
replacement = '''def test_coldstart_runbook_points_to_current_v2_protocol() -> None:
    superseded = Path("docs/experiments/EXT-C-E8-MULTITASK-EXP-COLDSTART-01_RUNBOOK.md").read_text(
        encoding="utf-8"
    )
    runbook = Path("docs/experiments/EXT-C-E8-MULTITASK-EXP-COLDSTART-01_RUNBOOK_V2.md").read_text(
        encoding="utf-8"
    )
    assert "SUPERSEDED" in superseded
    assert "EXT-C-E8-MULTITASK-EXP-COLDSTART-01_RUNBOOK_V2.md" in superseded
    assert "E8_MULTITASK_EXP_COLDSTART_20260820_02" in superseded
    assert "208 cells" in runbook
    assert "13" in runbook and "16-cell" in runbook
    assert "Spiral Matrix" in runbook
    assert "Pass@64" in runbook
    assert "结果门禁" in runbook
    assert "task_results/<task>" in runbook
    assert "TASK_COMPLETE.json" in runbook
    assert "terminal valid rate" in runbook.lower()
    assert "0.002" not in runbook
    assert "峰值必须高于" not in runbook
    assert "run_experiment_guard_hardened.py" in Path(
        "scripts/run_e8_multitask_exp_coldstart.sh"
    ).read_text(encoding="utf-8")
'''
successor_tests = Path('.github/tmp_e8_lambda_tests.txt').read_text(encoding='utf-8')
assert 'test_lambda_completion_matrix_is_config_driven_and_lambda_only' in successor_tests
path.write_text(text[:start] + replacement.rstrip() + '\n' + successor_tests, encoding='utf-8')
PY

cat >> tests/test_e8_multitask_p0.py <<'PYTEST'


def test_lambda_completion_preserves_restored_coldstart_behavior() -> None:
    from drpo import e8_multitask_exp_tuning as exp_tuning

    cold = exp_tuning.load_config(Path("configs/e8_multitask_exp_coldstart.yaml"))
    successor = exp_tuning.load_config(Path("configs/e8_multitask_exp_lambda_completion.yaml"))
    frozen_sections = (
        "parent", "reference", "initialization", "model", "suite", "split",
        "training", "evaluation", "negative_sampling", "remoteness_calibration",
        "task_runtime", "selection", "canonical_coldstart", "execution",
    )
    for section in frozen_sections:
        assert successor[section] == cold[section], section

    historical_cells = exp_tuning.build_cells(cold)
    historical_waves = exp_tuning.build_waves(cold)
    assert len(historical_cells) == 208
    assert [len(wave) for wave in historical_waves] == [16] * 13
    assert successor["negative_sampling"]["reference_remoteness_bank"]["selection"] == (
        "source_p0_error_class_sequence_then_within_class_reference_rank_spread"
    )
    assert successor["execution"]["scheduler"] == "dynamic_slot_queue"
    assert successor["execution"]["wave_barriers"] is False
    launcher = Path("scripts/run_e8_multitask_exp_lambda_completion.sh").read_text(encoding="utf-8")
    assert "bootstrap_e8_multitask_exp_coldstart.sh" in launcher
PYTEST

# Science-surface audit against the restored cold-start tree.
python - <<'PY'
import ast
import subprocess
from pathlib import Path
import yaml
base_sha = '8bdd07590f155ad26bc8cfbd641d40647eab57d2'
path = 'src/drpo/e8_multitask_exp_tuning.py'
base = subprocess.check_output(['git', 'show', f'{base_sha}:{path}'], text=True)
current = Path(path).read_text(encoding='utf-8')

def units(text):
    tree = ast.parse(text)
    lines = text.splitlines(keepends=True)
    return {node.name: ''.join(lines[node.lineno - 1:node.end_lineno])
            for node in tree.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))}
a, b = units(base), units(current)
changed = {name for name in a.keys() & b.keys() if a[name] != b[name]}
allowed = {'Cell', 'validate_config', 'build_cells', 'build_waves',
           '_countdown_protocol_diagnostic', '_coldstart_completed_task_rows',
           '_write_coldstart_task_result', '_aggregate_coldstart'}
assert changed <= allowed, sorted(changed - allowed)
print('changed_top_level_units=', sorted(changed))

cold = yaml.safe_load(Path('configs/e8_multitask_exp_coldstart.yaml').read_text(encoding='utf-8'))
successor = yaml.safe_load(Path('configs/e8_multitask_exp_lambda_completion.yaml').read_text(encoding='utf-8'))
for section in ('parent', 'reference', 'initialization', 'model', 'suite', 'split',
                'training', 'evaluation', 'negative_sampling', 'remoteness_calibration',
                'task_runtime', 'selection', 'canonical_coldstart', 'execution'):
    assert successor[section] == cold[section], section
PY

grep -F 'TRANSFER_SYSTEM_PROMPT = "Answer with only the requested final output and no explanation."' src/drpo/e8_multitask_exp_tuning.py
grep -F 'arena.SYSTEM_PROMPT = TRANSFER_SYSTEM_PROMPT' src/drpo/e8_multitask_exp_tuning.py
grep -F 'arena.sequence_surprisal_only(model, batch)' src/drpo/e8_multitask_exp_tuning.py
grep -F 'def cmd_run_dynamic(' src/drpo/e8_multitask_exp_tuning.py

git diff --check
bash -n scripts/bootstrap_e8_multitask_exp_coldstart.sh
bash -n scripts/run_e8_multitask_exp_coldstart.sh
bash -n scripts/run_e8_multitask_exp_lambda_completion.sh
python -m py_compile src/drpo/e8_multitask_exp_tuning.py tests/test_e8_multitask_p0.py
python -m pip install -q 'numpy==1.26.4' 'PyYAML==6.0.2' pytest
PYTHONPATH=src pytest -q tests/test_e8_multitask_p0.py

# Final candidate must not contain one-shot repair machinery.
git rm .github/tmp_e8_lambda_repair_apply.sh
git rm .github/tmp_e8_lambda_tests.txt
git rm .github/workflows/tmp-e8-lambda-repair-run.yml

git diff --check
git config user.name "github-actions[bot]"
git config user.email "41898282+github-actions[bot]@users.noreply.github.com"
git add -A
git commit -m "E8: rebuild lambda completion on restored coldstart execution"
git push origin "HEAD:${BRANCH}"
