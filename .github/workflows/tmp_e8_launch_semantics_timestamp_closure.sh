#!/usr/bin/env bash
set -euo pipefail

git config user.name 'github-actions[bot]'
git config user.email '41898282+github-actions[bot]@users.noreply.github.com'

MAIN_SHA=8a21fd16e63ea96a0cde2a477353d24562624655
test "$(git rev-parse origin/main)" = "$MAIN_SHA"

# 1. Durable scope first.
python - <<'PY'
from pathlib import Path
path = Path('docs/scopes/E8_MULTITASK_BASELINES_CAPABILITY_DEV.md')
text = path.read_text(encoding='utf-8')
heading = '## Owner-authorized launch-semantics and timestamp fail-closed closure (2026-09-10)'
if heading not in text:
    block = """

## Owner-authorized launch-semantics and timestamp fail-closed closure (2026-09-10)

The owner explicitly approved two final repairs after re-auditing the current launch policy and scheduler provenance. First, `GOV-EXPERIMENT-LAUNCH-SIMPLIFICATION-01` is authoritative for launch permission: registry/READY/schema-v3/formal-channel/lane activation bookkeeping must not by itself block an otherwise frozen, source/config-valid scientific workload. The existing E8 scientific runner and hardened shell execution path are implemented. Any still-planned `GOV-FORMAL-ENTRYPOINT-01` / hardened-v1 registration completeness is retained only as optional legacy operational bookkeeping and has no launch-veto or launch-permission role. This clarification does not authorize a scientific launch; the experiment remains **not_run** until an explicit later launch decision from a reviewed merged commit, and exact-commit, clean-checkout, frozen-science, provenance, recovery, supervision, terminal-audit, artifact, and repository-merge requirements remain unchanged. No new wrapper or Python path is authorized or required by this correction.

Second, the already-approved `scientific_execution_provenance` contract is made fail-closed at the write boundary. A newly executed scientific cell may persist authoritative chronology only from the actual `_run_subprocess_cell()` result. `started_unix` and `finished_unix` must both be present, numeric-convertible, finite, and ordered with `finished_unix >= started_unix`; a missing, malformed, non-finite, or reversed timestamp must raise and must never be replaced by a newly fabricated `time.time()` value. This is a robustness repair of the already-approved actual-subprocess-time evidence contract, not a new scientific threshold or experiment gate.

These repairs do not change the 176-cell matrix, eight transfer tasks, Countdown reuse, seeds `[4000,5000]`, hard 88+88 seed barrier, AsymRE/TOPR/DPO grids, 1,200-update horizon, evaluation protocol, model initialization, optimizer, canonical AsymRE/TOPR mathematics, or historical Countdown PR #268 DPO semantics. No scientific run is authorized or started by this closure.
"""
    path.write_text(text.rstrip() + block + '\n', encoding='utf-8')
PY
git add docs/scopes/E8_MULTITASK_BASELINES_CAPABILITY_DEV.md
git commit -m 'docs: authorize E8 launch semantics and timestamp closure'

# 2. Fail-closed timestamp evidence and focused regressions.
python - <<'PY'
from pathlib import Path
src = Path('src/drpo/e8_multitask_exp_tuning.py')
text = src.read_text(encoding='utf-8')
anchor = 'def _scientific_execution_provenance_valid(value: Mapping[str, Any], cell: Cell) -> bool:\n'
if '_require_scientific_execution_timestamps' not in text:
    if text.count(anchor) != 1:
        raise SystemExit(f'provenance validator anchor count={text.count(anchor)}')
    helper = '''def _require_scientific_execution_timestamps(\n    result: Mapping[str, Any],\n) -> tuple[float, float]:\n    try:\n        started = float(result["started_unix"])\n        finished = float(result["finished_unix"])\n    except (KeyError, TypeError, ValueError) as exc:\n        raise RuntimeError(\n            "subprocess result must contain numeric scientific execution timestamps"\n        ) from exc\n    if not math.isfinite(started) or not math.isfinite(finished):\n        raise RuntimeError("scientific execution timestamps must be finite")\n    if finished < started:\n        raise RuntimeError("scientific execution finish timestamp precedes start timestamp")\n    return started, finished\n\n\n'''
    text = text.replace(anchor, helper + anchor, 1)
old = '''                    if execution_origin == "executed_current_run":\n                        started_unix = float(result.get("started_unix", time.time()))\n                        finished_unix = float(result.get("finished_unix", time.time()))\n                        completed_manifest["scientific_execution_provenance"] = {\n'''
new = '''                    if execution_origin == "executed_current_run":\n                        started_unix, finished_unix = _require_scientific_execution_timestamps(\n                            result\n                        )\n                        completed_manifest["scientific_execution_provenance"] = {\n'''
if old in text:
    text = text.replace(old, new, 1)
elif 'result.get("started_unix", time.time())' in text or 'result.get("finished_unix", time.time())' in text:
    raise SystemExit('unexpected timestamp fallback shape')
src.write_text(text, encoding='utf-8')

test = Path('tests/test_e8_multitask_p0.py')
t = test.read_text(encoding='utf-8')
start = t.index('def test_baseline_matrix_tail_pipeline_scheduler_aggregate_and_resume_identity(')
end = t.index('\ndef test_formal_baseline_matrix_config_matches_september7_runbook()', start)
segment = t[start:end]
if '"started_unix": 0.0' not in segment:
    old_return = '''            "seed": cell.seed,\n            "returncode": 0,\n        }\n'''
    new_return = '''            "seed": cell.seed,\n            "returncode": 0,\n            "started_unix": 0.0,\n            "finished_unix": 1.0,\n        }\n'''
    if segment.count(old_return) != 1:
        raise SystemExit(f'tail fake return anchor count={segment.count(old_return)}')
    segment = segment.replace(old_return, new_return, 1)
    t = t[:start] + segment + t[end:]
regression_name = 'def test_scientific_execution_timestamps_fail_closed(result) -> None:'
if regression_name not in t:
    test_anchor = '\ndef test_formal_baseline_matrix_config_matches_september7_runbook() -> None:\n'
    regression = '''\n\n@pytest.mark.parametrize(\n    "result",\n    [\n        {},\n        {"started_unix": 1.0},\n        {"finished_unix": 2.0},\n        {"started_unix": "bad", "finished_unix": 2.0},\n        {"started_unix": 1.0, "finished_unix": "bad"},\n        {"started_unix": float("nan"), "finished_unix": 2.0},\n        {"started_unix": 1.0, "finished_unix": float("inf")},\n        {"started_unix": 2.0, "finished_unix": 1.0},\n    ],\n)\ndef test_scientific_execution_timestamps_fail_closed(result) -> None:\n    from drpo import e8_multitask_exp_tuning as exp_tuning\n\n    with pytest.raises(RuntimeError):\n        exp_tuning._require_scientific_execution_timestamps(result)\n\n\ndef test_scientific_execution_timestamps_accept_actual_ordered_values() -> None:\n    from drpo import e8_multitask_exp_tuning as exp_tuning\n\n    assert exp_tuning._require_scientific_execution_timestamps(\n        {"started_unix": "1.25", "finished_unix": 2}\n    ) == (1.25, 2.0)\n'''
    if t.count(test_anchor) != 1:
        raise SystemExit(f'test insertion anchor count={t.count(test_anchor)}')
    t = t.replace(test_anchor, regression + test_anchor, 1)
test.write_text(t, encoding='utf-8')
PY
git add src/drpo/e8_multitask_exp_tuning.py tests/test_e8_multitask_p0.py
git commit -m 'fix: fail closed on missing E8 execution timestamps'

# 3. Successor schema-v3 delta; prior accepted deltas remain untouched.
ROOT="$PWD"
BASE_HEAD="$(git rev-parse HEAD)"
TRUSTED=/tmp/e8-launch-trusted
SOURCE=/tmp/e8-launch-source
STAGE=/tmp/e8-launch-stage
rm -rf "$TRUSTED" "$SOURCE" "$STAGE"
git worktree add --detach "$TRUSTED" "$MAIN_SHA"
python "$TRUSTED/scripts/handoff_authority.py" verify --repo-root "$ROOT" --json | tee /tmp/e8-launch-predecessor-verify.json
git worktree add --detach "$SOURCE" "$BASE_HEAD"
SOURCE="$SOURCE" TRUSTED="$TRUSTED" BASE_HEAD="$BASE_HEAD" python - <<'PY'
import hashlib, os, sys
from pathlib import Path
import yaml
source = Path(os.environ['SOURCE'])
trusted = Path(os.environ['TRUSTED'])
base = os.environ['BASE_HEAD']
sys.path.insert(0, str(trusted / 'scripts'))
import handoff_delta_shadow as shadow
exp = 'EXT-C-E8-MULTITASK-BASELINE-MATRIX-01'
uid = exp + '-LAUNCH-SEMANTICS-TIMESTAMP-CORRECTION-2026-09-10'
ddir = source / 'docs/handoff_deltas' / uid
if ddir.exists():
    raise SystemExit('successor delta already exists')
rp = source / 'experiments/registry.yaml'
hp = source / 'docs/handoff.md'
r0 = rp.read_text(encoding='utf-8')
h0 = hp.read_text(encoding='utf-8')
start = r0.index(f'- id: {exp}\n')
end = r0.index('\n- id: ', start + 1)
block = r0[start:end]
old_state = '  implementation_state: not_implemented\n'
if block.count(old_state) != 1:
    raise SystemExit(f'implementation_state anchor count={block.count(old_state)}')
block = block.replace(old_state, '  implementation_state: implemented\n  scientific_runner_state: implemented\n  launch_blocked_by_formal_channel: false\n', 1)
b0 = block.index('  blocking_reason: >-\n')
b1 = block.index('  formal_execution:\n', b0)
block = block[:b0] + '  blocking_reason: null\n' + block[b1:]
marker = '    artifact_protocol: docs/formal_experiment_artifact_protocol.md\n'
if block.count(marker) != 1:
    raise SystemExit(f'formal execution marker count={block.count(marker)}')
block = block.replace(marker, marker + '    bookkeeping_role: optional_legacy_compatibility_only\n    launch_permission: false\n    launch_blocking: false\n', 1)
r1 = r0[:start] + block + r0[end:]
rp.write_text(r1, encoding='utf-8')
old_reg = yaml.safe_load(r0)
new_reg = yaml.safe_load(r1)
old_ent = next(x for x in old_reg['experiments'] if x['id'] == exp)
ent = next(x for x in new_reg['experiments'] if x['id'] == exp)
content = (
    '- **E8 multitask baseline launch-semantics / timestamp correction (`EXT-C-E8-MULTITASK-BASELINE-MATRIX-01`)：**'
    '本条覆盖前一 provenance correction 中仅关于“canonical hardened-v1 planned + blocked 因而 176-cell scientific run blocked”的表述。'
    '按照当前 `GOV-EXPERIMENT-LAUNCH-SIMPLIFICATION-01`，registry/READY/schema-v3/formal-channel/lane activation 不是 workload launch permission，'
    '也不得仅因其缺失、stale 或未激活而阻塞已经冻结且 source/config-valid 的科学 workload。现有 E8 scientific runner 与 hardened shell execution path '
    '记为 **implemented**；仍计划中的 hardened-v1 formal-channel registration completeness 只保留为 optional legacy operational bookkeeping，'
    '`launch_blocked_by_formal_channel=false`，既不授权 launch，也没有 launch veto。科学状态仍为 **not_run**；实际启动仍须满足 exact commit、clean checkout、'
    'frozen science、provenance、recovery、supervision、terminal audit、artifact 与现行 merge/launch 决策要求。另将已批准的 scientific execution chronology '
    '写入端改为 fail-closed：`_run_subprocess_cell()` 必须实际返回 present、numeric-convertible、finite 且 `finished_unix >= started_unix` 的 '
    '`started_unix/finished_unix`，缺失、非法、非有限或倒序都必须失败，禁止使用 `time.time()` 现场补造 authoritative evidence。'
    '本修正不改变 176-cell 矩阵、tasks、seeds、AsymRE/TOPR/DPO 网格、1200-update horizon、optimizer 或 historical Countdown PR #268 DPO 语义。'
)
op = {
    'operation_id': 'append-e8-baseline-launch-semantics-timestamp-correction',
    'op': 'append_to_section',
    'heading_path': ['0. 研究与执行原则（每次新会话首先阅读）', '0.1 当前执行门禁'],
    'block_id': 'e8-multitask-baseline-launch-semantics-timestamp-correction-2026-09-10',
    'content': content,
}
cand = shadow.render(h0, [op]).text
evidence = [
    'AGENTS.md',
    'docs/scopes/GOV-EXPERIMENT-LAUNCH-SIMPLIFICATION-01.md',
    'docs/scopes/E8_MULTITASK_BASELINES_CAPABILITY_DEV.md',
    'experiments/registry.yaml',
    'scripts/run_e8_multitask_exp_coldstart.sh',
    'src/drpo/e8_multitask_exp_tuning.py',
    'tests/test_e8_multitask_p0.py',
]
def change(cid, path, before, after, reason):
    return {'change_id': cid, 'kind': 'update_field', 'entity_id': exp, 'field_path': path, 'from': before, 'to': after, 'reason': reason, 'evidence': evidence}
changes = [
    change('correct-e8-implementation-state', ['implementation_state'], old_ent.get('implementation_state'), ent.get('implementation_state'), 'Scientific runner is implemented; legacy channel completeness is not the experiment implementation state.'),
    change('add-e8-scientific-runner-state', ['scientific_runner_state'], old_ent.get('scientific_runner_state'), ent.get('scientific_runner_state'), 'Make scientific runner implementation explicit.'),
    change('add-e8-formal-channel-launch-blocking', ['launch_blocked_by_formal_channel'], old_ent.get('launch_blocked_by_formal_channel'), ent.get('launch_blocked_by_formal_channel'), 'Current launch simplification forbids formal-channel activation bookkeeping from acting as launch veto.'),
    change('clear-e8-formal-channel-blocking-reason', ['blocking_reason'], old_ent.get('blocking_reason'), ent.get('blocking_reason'), 'Remove stale launch-blocking semantics while preserving not_run scientific status.'),
    change('add-e8-formal-bookkeeping-role', ['formal_execution', 'bookkeeping_role'], old_ent['formal_execution'].get('bookkeeping_role'), ent['formal_execution'].get('bookkeeping_role'), 'Classify remaining hardened-v1 completeness metadata as optional legacy operational bookkeeping.'),
    change('add-e8-formal-launch-permission-role', ['formal_execution', 'launch_permission'], old_ent['formal_execution'].get('launch_permission'), ent['formal_execution'].get('launch_permission'), 'Formal-channel bookkeeping does not grant scientific launch permission.'),
    change('add-e8-formal-launch-blocking-role', ['formal_execution', 'launch_blocking'], old_ent['formal_execution'].get('launch_blocking'), ent['formal_execution'].get('launch_blocking'), 'Formal-channel bookkeeping does not block an otherwise valid frozen launch.'),
]
sha = lambda s: hashlib.sha256(s.encode()).hexdigest()
delta = {
    'schema_version': 3,
    'update_id': uid,
    'mode': 'authoritative',
    'base': {'commit': base, 'handoff_sha256': sha(h0), 'registry_sha256': sha(r0)},
    'renderer_version': 1,
    'operations': [op],
    'registry': {'mode': 'expected_after', 'exact_base_after_sha256': sha(r1), 'changes': changes},
    'expected': {'exact_base_candidate_sha256': sha(cand)},
}
ddir.mkdir(parents=True)
(ddir / 'HANDOFF_DELTA.yaml').write_text(yaml.safe_dump(delta, sort_keys=False, allow_unicode=True), encoding='utf-8')
print('successor_delta', uid)
print('base_head', base)
print('candidate_handoff_sha256', sha(cand))
print('registry_after_sha256', sha(r1))
PY
git -C "$SOURCE" add experiments/registry.yaml docs/handoff_deltas/EXT-C-E8-MULTITASK-BASELINE-MATRIX-01-LAUNCH-SEMANTICS-TIMESTAMP-CORRECTION-2026-09-10/HANDOFF_DELTA.yaml
git -C "$SOURCE" commit -m 'source: append E8 launch semantics correction delta'
SOURCE_COMMIT="$(git -C "$SOURCE" rev-parse HEAD)"
git worktree add --detach "$STAGE" "$SOURCE_COMMIT"
python "$TRUSTED/scripts/handoff_authority.py" normalize --repo-root "$STAGE" --trusted-repo-root "$TRUSTED" --current-before "$BASE_HEAD" --source-base "$BASE_HEAD" --source-patch-commit "$SOURCE_COMMIT" --json | tee /tmp/e8-launch-normalize.json
git -C "$STAGE" add -A
git -C "$STAGE" commit --amend --no-edit
python "$TRUSTED/scripts/handoff_authority.py" verify --repo-root "$STAGE" --json | tee /tmp/e8-launch-stage-verify.json
python "$TRUSTED/scripts/validate_governance_pipeline_stage_status.py" --repo-root "$STAGE"
git -C "$STAGE" diff --check "$BASE_HEAD"..HEAD
while IFS= read -r -d '' f; do
  if test -e "$STAGE/$f"; then
    mkdir -p "$ROOT/$(dirname "$f")"
    cp "$STAGE/$f" "$ROOT/$f"
  else
    rm -f "$ROOT/$f"
  fi
done < <(git -C "$STAGE" diff --name-only -z "$BASE_HEAD"..HEAD)
git add -A -- ':!.github/workflows/tmp_e8_launch_semantics_timestamp_closure.yml' ':!.github/workflows/tmp_e8_launch_semantics_timestamp_closure.sh' ':!.github/workflows/tmp_e8_launch_semantics_timestamp_closure_v2.yml'
test -n "$(git diff --cached --name-only)"
git commit -m 'docs: materialize E8 launch semantics correction'
python "$TRUSTED/scripts/handoff_authority.py" verify --repo-root "$ROOT" --json | tee /tmp/e8-launch-branch-verify.json
python "$TRUSTED/scripts/validate_governance_pipeline_stage_status.py" --repo-root "$ROOT"
git diff --check "$MAIN_SHA"..HEAD

# 4. Regressions/static checks. The legacy formal-channel validator is intentionally not a launch gate.
PYTHONPATH=.:src python -m pytest -q tests/test_e8_multitask_p0.py
ruff check src/drpo/e8_experiment_config.py tests/test_e8_multitask_p0.py
ruff check --ignore BLE001,B023 src/drpo/e8_multitask_exp_tuning.py
python -m py_compile src/drpo/e8_experiment_config.py src/drpo/e8_multitask_exp_tuning.py
python - <<'PY'
from pathlib import Path
text = Path('src/drpo/e8_multitask_exp_tuning.py').read_text(encoding='utf-8')
assert 'result.get("started_unix", time.time())' not in text
assert 'result.get("finished_unix", time.time())' not in text
assert 'historical_PR_268_protected_implementation_cc0ead2be00c89a3c35296b7adc1ddeae8d14759' in text
print('E8_TIMESTAMP_FALLBACK_REMOVED_AND_DPO_PR268_MARKER_PRESERVED')
PY

# 5. Remove all temporary helpers in one commit and push the reviewed closure.
git rm .github/workflows/tmp_e8_launch_semantics_timestamp_closure.yml .github/workflows/tmp_e8_launch_semantics_timestamp_closure.sh .github/workflows/tmp_e8_launch_semantics_timestamp_closure_v2.yml
git commit -m 'chore: remove temporary E8 closure helpers'
git push origin HEAD:dev/e8-multitask-baselines-capability-01
