#!/usr/bin/env bash
set -euo pipefail

BASE=8bdd07590f155ad26bc8cfbd641d40647eab57d2
BRANCH=dev/e8-multitask-lambda-completion-repair-03-finalize
EXP_ID=EXT-C-E8-MULTITASK-EXP-LAMBDA-COMPLETION-01

git merge-base --is-ancestor "${BASE}" HEAD

python - <<'PY'
from pathlib import Path
import re
import textwrap
import yaml

EXP_ID = "EXT-C-E8-MULTITASK-EXP-LAMBDA-COMPLETION-01"

# Fix the successor config path: it must be resolved inside bootstrap's isolated checkout.
launcher = Path("scripts/run_e8_multitask_exp_lambda_completion.sh")
text = launcher.read_text(encoding="utf-8")
old = 'export E8_COLDSTART_CONFIG="${ROOT_DIR}/configs/e8_multitask_exp_lambda_completion.yaml"\n'
new = 'export E8_COLDSTART_CONFIG="configs/e8_multitask_exp_lambda_completion.yaml"\n'
assert text.count(old) == 1, text.count(old)
launcher.write_text(text.replace(old, new, 1), encoding="utf-8")

# Add successor-only source-provenance files without changing cold-start defaults.
runner = Path("scripts/run_e8_multitask_exp_coldstart.sh")
text = runner.read_text(encoding="utf-8")
anchor = 'resolve_config_repo_path\n\ncase "${RUN_CLASS}" in\n'
replacement = '''resolve_config_repo_path

SUCCESSOR_SOURCE_ARGS=()
if [[ "${EXPERIMENT_ID}" == "EXT-C-E8-MULTITASK-EXP-LAMBDA-COMPLETION-01" ]]; then
  SUCCESSOR_SOURCE_ARGS=(
    --source-file scripts/run_e8_multitask_exp_lambda_completion.sh
    --source-file docs/experiments/E8_MULTITASK_LAMBDA_COMPLETION_PROTOCOL.md
  )
fi

case "${RUN_CLASS}" in
'''
assert text.count(anchor) == 1, text.count(anchor)
text = text.replace(anchor, replacement, 1)
lines = text.splitlines()
rebuilt = []
config_source_count = 0
for line in lines:
    rebuilt.append(line)
    if '--source-file "${CONFIG_REPO_PATH}"' in line:
        indent = line[: len(line) - len(line.lstrip())]
        suffix = ' \\' if line.rstrip().endswith('\\') else ''
        rebuilt.append(f'{indent}"${{SUCCESSOR_SOURCE_ARGS[@]}}"{suffix}')
        config_source_count += 1
assert config_source_count == 3, config_source_count
runner.write_text("\n".join(rebuilt) + "\n", encoding="utf-8")

# Complete the current V2 runbook with facts that the predecessor regression test checks.
runbook = Path("docs/experiments/EXT-C-E8-MULTITASK-EXP-COLDSTART-01_RUNBOOK_V2.md")
text = runbook.read_text(encoding="utf-8")
section = '''

## 6. 可审计输出与实现身份

- transfer-task 的 reference-remoteness bank 仍只承担负例覆盖 provenance/diagnostic；训练权重继续使用每次 update 重算的 current-policy surprisal。
- 每个 task 完成后，确定性快照写入 `task_results/<task>/`，并以 `TASK_COMPLETE.json` 作为该 task 全部冻结 cells 已完成的原子完成标记；最终 aggregate/audit 仍是权威结果。
- terminal valid rate 继续只作结构/有效性诊断，不作为 Exp coefficient 选择资格门槛，也不新增任何结果门禁。
'''
if "## 6. 可审计输出与实现身份" not in text:
    text = text.rstrip() + section + "\n"
runbook.write_text(text, encoding="utf-8")

# Register the successor in the research master before experiment execution.
handoff = Path("docs/handoff.md")
text = handoff.read_text(encoding="utf-8")
if EXP_ID not in text:
    first_nl = text.find("\n")
    assert first_nl > 0
    first = text[:first_nl]
    if " v80" in first:
        first = first.replace(" v80", " v81", 1)
    block = f'''\n<!-- HANDOFF-DELTA-BLOCK:after_heading:v81-e8-lambda-completion-repair:START -->
> **v81 增量登记：`{EXP_ID}` 在已恢复 cold-start execution tree 上重建（不删除 v80 及更早内容）**
>
> - 新 successor `{EXP_ID}` 状态为 **not_run**，职责仅为八个非 Countdown 任务的高-lambda response-tail / curve-completion；不得用于方法排名、收敛或稳态结论。Countdown 新增科学 cells 为 0。
> - implementation base 为 `8bdd07590f155ad26bc8cfbd641d40647eab57d2`，其 repository tree 与公开 pre-successor execution commit `1723d0c507b2309a1a352c2459165b86b9625c9d` 完全一致。历史服务器记录的 `01471868...` 仍不可解析，因此不得声称 byte-for-byte 恢复该不可解析 source commit。
> - 新 workload 为 199 cells：183 Exp + 16 Positive-only；Spiral Matrix 10 cells，其他七个 transfer tasks 各 27 cells；Exp seed offset 4000，Positive-only seed offsets 8000/9000。13 组只作 nominal audit geometry，实际继续使用 16-slot shared dynamic refill，无 hard wave barrier。
> - successor 仅新增 lambda-only transport 与新 grid/config/launcher/provenance；negative-bank selector、16-negative consumer、transfer system prompt、current-policy sequence-surprisal、fresh LoRA、1200 updates、optimizer/LR、task runtime overrides、evaluation、recovery 与 terminal audit 继续继承 cold-start execution 行为。
> - protocol：`docs/experiments/E8_MULTITASK_LAMBDA_COMPLETION_PROTOCOL.md`；config：`configs/e8_multitask_exp_lambda_completion.yaml`；entrypoint：`scripts/run_e8_multitask_exp_lambda_completion.sh`。任何 scientific launch 前必须完成 canonical lambda transport 的 taper/loss/gradient exact-equivalence test。
<!-- HANDOFF-DELTA-BLOCK:after_heading:v81-e8-lambda-completion-repair:END -->\n'''
    text = first + "\n" + block + text[first_nl + 1:]
    handoff.write_text(text, encoding="utf-8")

# Register the successor in registry without reformatting historical entries.
registry = Path("experiments/registry.yaml")
text = registry.read_text(encoding="utf-8")
if EXP_ID not in text:
    lines = text.splitlines(keepends=True)
    start = next(i for i, line in enumerate(lines) if line.strip() == "experiments:")
    end = len(lines)
    root_key = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*:\s*(?:#.*)?$")
    for i in range(start + 1, len(lines)):
        if root_key.match(lines[i].rstrip("\n")):
            end = i
            break
    block = textwrap.dedent(f'''\
- id: {EXP_ID}
  environment: Countdown + eight frozen transfer tasks
  name: e8_multitask_exp_lambda_completion
  status: not_run
  claim: Complete the frozen high-lambda response tails for the eight non-Countdown tasks while preserving the closed 208-cell predecessor; no method-ranking, convergence, steady-state, or OOD claim is permitted.
  role: external_validity_response_shape
  execution_class: pilot
  result_status: not_run
  depends_on_delivered_experiment: EXT-C-E8-MULTITASK-EXP-COLDSTART-01
  code_entrypoint: src/drpo/e8_multitask_exp_tuning.py
  command:
  - bash
  - scripts/run_e8_multitask_exp_lambda_completion.sh
  config_entrypoint: configs/e8_multitask_exp_lambda_completion.yaml
  protocol_document: docs/experiments/E8_MULTITASK_LAMBDA_COMPLETION_PROTOCOL.md
  workload:
    expected_cells: 199
    exp_cells: 183
    positive_only_cells: 16
    countdown_new_scientific_cells: 0
    exp_seed_offset: 4000
    positive_only_seed_offsets: [8000, 9000]
    scheduler: dynamic_slot_queue
    wave_barriers: false
  scientific_invariants:
  - source_p0_error_class_sequence_then_within_class_reference_rank_spread
  - sixteen_negative_consumer
  - current_policy_sequence_surprisal_each_update
  - fresh_lora_zero_update_initialization
  - fixed_1200_optimizer_updates
  - coldstart_task_runtime_and_evaluation_contract
  - coldstart_recovery_and_terminal_audit
  formal_evidence_allowed: false
  execution:
    state: not_run
    launched: false
  preserved_history: true
''')
    lines[end:end] = [block + ("\n" if not block.endswith("\n") else "")]
    registry.write_text("".join(lines), encoding="utf-8")
parsed = yaml.safe_load(registry.read_text(encoding="utf-8"))
ids = [entry.get("id") for entry in parsed["experiments"]]
assert ids.count(EXP_ID) == 1, ids.count(EXP_ID)

# Record successor-specific provenance inventory in its protocol.
protocol = Path("docs/experiments/E8_MULTITASK_LAMBDA_COMPLETION_PROTOCOL.md")
text = protocol.read_text(encoding="utf-8")
needle = "7. No successor scientific hyperparameter is hard-coded into Python.\n"
replacement = needle + "8. Guarded source provenance includes the successor launcher and this protocol document in addition to the selected successor config and inherited cold-start scientific sources.\n"
assert text.count(needle) == 1, text.count(needle)
protocol.write_text(text.replace(needle, replacement, 1), encoding="utf-8")

# Restore predecessor coverage while keeping the successor-specific checks additive.
tests = Path("tests/test_e8_multitask_p0.py")
text = tests.read_text(encoding="utf-8")
start_marker = "def test_coldstart_runbook_points_to_current_v2_protocol() -> None:\n"
end_marker = "\ndef test_lambda_completion_matrix_is_config_driven_and_lambda_only"
start = text.index(start_marker)
end = text.index(end_marker, start)
replacement = '''def test_coldstart_runbook_points_to_current_v2_protocol() -> None:
    superseded = Path("docs/experiments/EXT-C-E8-MULTITASK-EXP-COLDSTART-01_RUNBOOK.md").read_text(
        encoding="utf-8"
    )
    runbook = Path("docs/experiments/EXT-C-E8-MULTITASK-EXP-COLDSTART-01_RUNBOOK_V2.md").read_text(
        encoding="utf-8"
    )
    bootstrap = Path("scripts/bootstrap_e8_multitask_exp_coldstart.sh").read_text(encoding="utf-8")
    assert "SUPERSEDED" in superseded
    assert "EXT-C-E8-MULTITASK-EXP-COLDSTART-01_RUNBOOK_V2.md" in superseded
    assert "E8_MULTITASK_EXP_COLDSTART_20260820_02" in superseded
    assert "208 cells" in runbook
    assert "13" in runbook and "16-cell" in runbook
    assert "Spiral Matrix" in runbook
    assert "Pass@64" in runbook
    assert "结果门禁" in runbook
    assert "reference-remoteness" in runbook
    assert "task_results/<task>" in runbook
    assert "TASK_COMPLETE.json" in runbook
    assert "terminal valid rate" in runbook.lower()
    assert "0.002" not in runbook
    assert "峰值必须高于" not in runbook
    historical_bootstrap = subprocess.check_output(
        ["git", "show", "8bdd07590f155ad26bc8cfbd641d40647eab57d2:scripts/bootstrap_e8_multitask_exp_coldstart.sh"],
        text=True,
    )
    normalized = bootstrap.replace(
        'EXPERIMENT_ID="${E8_COLDSTART_EXPERIMENT_ID:-EXT-C-E8-MULTITASK-EXP-COLDSTART-01}"',
        'EXPERIMENT_ID="EXT-C-E8-MULTITASK-EXP-COLDSTART-01"',
        1,
    )
    assert normalized == historical_bootstrap
    assert "run_experiment_guard_hardened.py" in Path(
        "scripts/run_e8_multitask_exp_coldstart.sh"
    ).read_text(encoding="utf-8")
'''
text = text[:start] + replacement.rstrip() + text[end:]
launcher_assert = '''    assert 'export E8_COLDSTART_EXPERIMENT_ID="EXT-C-E8-MULTITASK-EXP-LAMBDA-COMPLETION-01"' in successor_launcher
    assert 'EXPERIMENT_ID="${E8_COLDSTART_EXPERIMENT_ID:-EXT-C-E8-MULTITASK-EXP-COLDSTART-01}"' in historical_launcher
'''
launcher_replacement = launcher_assert + '''    assert 'export E8_COLDSTART_CONFIG="configs/e8_multitask_exp_lambda_completion.yaml"' in successor_launcher
    assert '${ROOT_DIR}/configs/e8_multitask_exp_lambda_completion.yaml' not in successor_launcher
    assert "EXT-C-E8-MULTITASK-EXP-LAMBDA-COMPLETION-01" in Path("docs/handoff.md").read_text(encoding="utf-8")
    assert "EXT-C-E8-MULTITASK-EXP-LAMBDA-COMPLETION-01" in Path("experiments/registry.yaml").read_text(encoding="utf-8")
    assert "SUCCESSOR_SOURCE_ARGS" in historical_launcher
    assert "scripts/run_e8_multitask_exp_lambda_completion.sh" in historical_launcher
    assert "docs/experiments/E8_MULTITASK_LAMBDA_COMPLETION_PROTOCOL.md" in historical_launcher
'''
assert text.count(launcher_assert) == 1, text.count(launcher_assert)
tests.write_text(text.replace(launcher_assert, launcher_replacement, 1), encoding="utf-8")
PY

git diff --check
bash -n scripts/bootstrap_e8_multitask_exp_coldstart.sh
bash -n scripts/run_e8_multitask_exp_coldstart.sh
bash -n scripts/run_e8_multitask_exp_lambda_completion.sh
python -m py_compile src/drpo/e8_multitask_exp_tuning.py tests/test_e8_multitask_p0.py

# Audit that the restored cold-start scientific surface stays untouched.
python - <<'PY'
import ast
import subprocess
from pathlib import Path
import yaml

BASE = "8bdd07590f155ad26bc8cfbd641d40647eab57d2"
changed = set(subprocess.check_output(["git", "diff", "--name-only", BASE], text=True).splitlines())
allowed = {
    "configs/e8_multitask_exp_lambda_completion.yaml",
    "docs/experiments/E8_MULTITASK_LAMBDA_COMPLETION_PROTOCOL.md",
    "docs/experiments/EXT-C-E8-MULTITASK-EXP-COLDSTART-01_RUNBOOK_V2.md",
    "docs/handoff.md",
    "experiments/registry.yaml",
    "experiments/results/e8_multitask_exp_coldstart_20260820_02/CURVE_ANCHOR.csv",
    "scripts/bootstrap_e8_multitask_exp_coldstart.sh",
    "scripts/run_e8_multitask_exp_coldstart.sh",
    "scripts/run_e8_multitask_exp_lambda_completion.sh",
    "src/drpo/e8_multitask_exp_tuning.py",
    "tests/test_e8_multitask_p0.py",
    ".github/tmp_e8_finalize_apply.sh",
    ".github/workflows/tmp-e8-lambda-finalize.yml",
}
assert changed <= allowed, sorted(changed - allowed)
for path in (
    "configs/e8_multitask_exp_coldstart.yaml",
    "src/drpo/countdown_qwen_arena_onefile.py",
    "src/drpo/countdown_e8_alpha1_c_scan_common.py",
    "src/drpo/countdown_e8_alpha1_c_scan_runtime.py",
    "src/drpo/countdown_e8_alpha1_c_scan_trainer.py",
    "src/drpo/countdown_e8_alpha1_highc_scan_common.py",
    "src/drpo/countdown_e8_alpha1_highc_scan_runtime.py",
    "src/drpo/e8_multitask_p0.py",
    "src/drpo/e8_multitask_tasks.py",
    "configs/e8_multitask_p0.yaml",
):
    subprocess.check_call(["git", "diff", "--exit-code", BASE, "--", path])

path = "src/drpo/e8_multitask_exp_tuning.py"
base = subprocess.check_output(["git", "show", f"{BASE}:{path}"], text=True)
current = Path(path).read_text(encoding="utf-8")
def units(src):
    tree = ast.parse(src)
    lines = src.splitlines(keepends=True)
    return {
        node.name: ''.join(lines[node.lineno - 1:node.end_lineno])
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
    }
a, b = units(base), units(current)
changed_units = {name for name in a.keys() & b.keys() if a[name] != b[name]}
allowed_units = {
    "Cell", "validate_config", "build_cells", "build_waves",
    "_countdown_protocol_diagnostic", "_coldstart_completed_task_rows",
    "_write_coldstart_task_result", "_aggregate_coldstart",
}
assert changed_units <= allowed_units, sorted(changed_units - allowed_units)

cold = yaml.safe_load(Path("configs/e8_multitask_exp_coldstart.yaml").read_text(encoding="utf-8"))
successor = yaml.safe_load(Path("configs/e8_multitask_exp_lambda_completion.yaml").read_text(encoding="utf-8"))
for section in (
    "parent", "reference", "initialization", "model", "suite", "split",
    "training", "evaluation", "negative_sampling", "remoteness_calibration",
    "task_runtime", "selection", "canonical_coldstart", "execution",
):
    assert successor[section] == cold[section], section

registry = yaml.safe_load(Path("experiments/registry.yaml").read_text(encoding="utf-8"))
ids = [entry.get("id") for entry in registry["experiments"]]
assert ids.count("EXT-C-E8-MULTITASK-EXP-LAMBDA-COMPLETION-01") == 1
handoff = Path("docs/handoff.md").read_text(encoding="utf-8")
assert handoff.count("EXT-C-E8-MULTITASK-EXP-LAMBDA-COMPLETION-01") >= 1
src = Path("src/drpo/e8_multitask_exp_tuning.py").read_text(encoding="utf-8")
for needle in (
    'TRANSFER_SYSTEM_PROMPT = "Answer with only the requested final output and no explanation."',
    'arena.SYSTEM_PROMPT = TRANSFER_SYSTEM_PROMPT',
    'arena.sequence_surprisal_only(model, batch)',
    'def cmd_run_dynamic(',
):
    assert needle in src, needle
PY

python -m pip install -q 'numpy==1.26.4' 'PyYAML==6.0.2' pytest
PYTHONPATH=src pytest -q tests/test_e8_multitask_p0.py

# Execute the previously skipped exact taper/loss/gradient equivalence test with real Torch.
python -m pip install -q --index-url https://download.pytorch.org/whl/cpu 'torch==2.7.1'
PYTHONPATH=src pytest -q tests/test_e8_multitask_p0.py -k lambda_only_canonical_transport_is_exactly_equivalent

git diff --check

git rm .github/tmp_e8_finalize_apply.sh
git rm .github/workflows/tmp-e8-lambda-finalize.yml
git config user.name "github-actions[bot]"
git config user.email "41898282+github-actions[bot]@users.noreply.github.com"
git add -A
git commit -m "E8: finalize lambda completion repair on restored coldstart"
git push origin "HEAD:${BRANCH}"
