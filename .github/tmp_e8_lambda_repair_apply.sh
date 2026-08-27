#!/usr/bin/env bash
set -euo pipefail

BASE="8bdd07590f155ad26bc8cfbd641d40647eab57d2"
SOURCE="71b33ea5a783660008e495a34eda7757bee757d1"
MAIN_WITH_ANCHOR="5260ca58ecb67509838a1e5480f6dcaafdb7a773"
BRANCH="dev/e8-multitask-lambda-completion-repair-02"

# Ensure this executor only runs on the branch created from the restored
# cold-start execution tree. Temporary CI commits may sit above BASE.
git merge-base --is-ancestor "${BASE}" HEAD
git cat-file -e "${SOURCE}^{commit}"
git cat-file -e "${MAIN_WITH_ANCHOR}^{commit}"

# Only protocol/temporary CI files may precede the scientific repair.
python - <<'PY'
import subprocess
allowed = {
    'docs/experiments/E8_MULTITASK_LAMBDA_COMPLETION_PROTOCOL.md',
    '.github/tmp_e8_lambda_repair_apply.sh',
    '.github/workflows/tmp-e8-lambda-repair-run.yml',
}
changed = set(subprocess.check_output(
    ['git', 'diff', '--name-only', '8bdd07590f155ad26bc8cfbd641d40647eab57d2..HEAD'],
    text=True,
).splitlines())
extra = changed - allowed
assert not extra, sorted(extra)
PY

# Apply only the successor-specific tuning delta. The restored cold-start tree
# differs in exactly one build_waves hunk; all other hunks must apply cleanly.
git diff "${SOURCE}^" "${SOURCE}" -- src/drpo/e8_multitask_exp_tuning.py > /tmp/e8_source.patch
if git apply --reject --whitespace=nowarn /tmp/e8_source.patch; then
  echo "ERROR: expected the restored build_waves wording to require one manual hunk" >&2
  exit 2
fi
REJECT="src/drpo/e8_multitask_exp_tuning.py.rej"
[[ -f "${REJECT}" ]]
[[ "$(grep -c '^@@' "${REJECT}")" -eq 1 ]]
grep -F 'def build_waves' "${REJECT}"
grep -F 'Cold-start geometry must be 13 exact 16-cell scheduling waves' "${REJECT}"
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
path.write_text(text.replace(old, new, 1), encoding='utf-8')
PY

# Add the frozen 199-cell matrix from the reviewed successor delta, then restore
# the two scientific fields that were stale on main.
git show "${SOURCE}:configs/e8_multitask_exp_lambda_completion.yaml" > \
  configs/e8_multitask_exp_lambda_completion.yaml
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

# Keep the historical curve anchor locally available for later concatenation.
ANCHOR="experiments/results/e8_multitask_exp_coldstart_20260820_02/CURVE_ANCHOR.csv"
mkdir -p "$(dirname "${ANCHOR}")"
git show "${MAIN_WITH_ANCHOR}:${ANCHOR}" > "${ANCHOR}"
echo '36f0ac540edc5b399e9a30ea4bd0a30c030808aa79477cb6e3db28f5688d898d  '"${ANCHOR}" | sha256sum -c -

# Parameterize experiment/config identity only; preserve cold-start run-class,
# recovery, scheduler, task interface and scientific implementation.
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
old = 'EXPERIMENT_ID="EXT-C-E8-MULTITASK-EXP-COLDSTART-01"\n'
new = 'EXPERIMENT_ID="${E8_COLDSTART_EXPERIMENT_ID:-EXT-C-E8-MULTITASK-EXP-COLDSTART-01}"\n'
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
text = text.replace(literal, '--source-file "${CONFIG_REPO_PATH}"')
runner.write_text(text, encoding='utf-8')
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

# Add the reviewed successor tests from the same small delta; do not replace the
# historical test file.
git diff "${SOURCE}^" "${SOURCE}" -- tests/test_e8_multitask_p0.py > /tmp/e8_tests.patch
git apply /tmp/e8_tests.patch
cat >> tests/test_e8_multitask_p0.py <<'PYTEST'


def test_lambda_completion_preserves_restored_coldstart_behavior() -> None:
    from drpo import e8_multitask_exp_tuning as exp_tuning

    cold = exp_tuning.load_config(Path("configs/e8_multitask_exp_coldstart.yaml"))
    successor = exp_tuning.load_config(Path("configs/e8_multitask_exp_lambda_completion.yaml"))
    frozen_sections = (
        "parent",
        "reference",
        "initialization",
        "model",
        "suite",
        "split",
        "training",
        "evaluation",
        "negative_sampling",
        "remoteness_calibration",
        "task_runtime",
        "selection",
        "canonical_coldstart",
        "execution",
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
    assert 'E8_COLDSTART_RUN_CLASS="${E8_COLDSTART_RUN_CLASS:-pilot}"' in launcher
    assert 'E8_COLDSTART_REQUIRE_ORIGIN_MAIN="${E8_COLDSTART_REQUIRE_ORIGIN_MAIN:-0}"' in launcher
PYTEST

# Static science-surface audit against the restored cold-start implementation.
python - <<'PY'
import ast
import subprocess
from pathlib import Path
import yaml

base_sha = '8bdd07590f155ad26bc8cfbd641d40647eab57d2'
path = 'src/drpo/e8_multitask_exp_tuning.py'
base = subprocess.check_output(['git', 'show', f'{base_sha}:{path}'], text=True)
current = Path(path).read_text(encoding='utf-8')

def units(text: str) -> dict[str, str]:
    tree = ast.parse(text)
    lines = text.splitlines(keepends=True)
    return {
        node.name: ''.join(lines[node.lineno - 1:node.end_lineno])
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
    }

a = units(base)
b = units(current)
changed = {name for name in a.keys() & b.keys() if a[name] != b[name]}
allowed = {
    'Cell',
    'validate_config',
    'build_cells',
    'build_waves',
    '_countdown_protocol_diagnostic',
    '_coldstart_completed_task_rows',
    '_write_coldstart_task_result',
    '_aggregate_coldstart',
}
assert changed <= allowed, sorted(changed - allowed)
print('changed_top_level_units=', sorted(changed))

cold = yaml.safe_load(Path('configs/e8_multitask_exp_coldstart.yaml').read_text(encoding='utf-8'))
successor = yaml.safe_load(Path('configs/e8_multitask_exp_lambda_completion.yaml').read_text(encoding='utf-8'))
for section in (
    'parent', 'reference', 'initialization', 'model', 'suite', 'split',
    'training', 'evaluation', 'negative_sampling', 'remoteness_calibration',
    'task_runtime', 'selection', 'canonical_coldstart', 'execution',
):
    assert successor[section] == cold[section], section
PY

grep -F 'TRANSFER_SYSTEM_PROMPT = "Answer with only the requested final output and no explanation."' \
  src/drpo/e8_multitask_exp_tuning.py
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

# Remove one-shot CI machinery before the candidate commit.
git rm .github/tmp_e8_lambda_repair_apply.sh
git rm .github/workflows/tmp-e8-lambda-repair-run.yml

git diff --check
git config user.name "github-actions[bot]"
git config user.email "41898282+github-actions[bot]@users.noreply.github.com"
git add -A
git commit -m "E8: rebuild lambda completion on restored coldstart execution"
git push origin "HEAD:${BRANCH}"
