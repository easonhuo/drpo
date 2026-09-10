#!/usr/bin/env bash
set -euo pipefail

python - <<'PY'
from pathlib import Path
path = Path('.github/workflows/tmp_e8_terminal_evidence_patch.sh')
text = path.read_text(encoding='utf-8')
start_marker = '# Seal the exact pre-final identity payload. This preserves the existing final identity_hash bytes.\n'
end_marker = '# Add terminal identity and true-execution provenance validators before cmd_audit.\n'
start = text.index(start_marker)
end = text.index(end_marker, start)
replacement = r'''# Seal the exact pre-final identity payload. This preserves the existing final identity_hash bytes.
def replace_identity_hash_in_function(function_name: str) -> None:
    global text
    function_start = text.index(f"def {function_name}(")
    next_function = text.find("\ndef ", function_start + 1)
    function_end = len(text) if next_function == -1 else next_function
    segment = text[function_start:function_end]
    old = '    identity["identity_hash"] = stable_hash(identity)\n'
    if segment.count(old) != 1:
        raise SystemExit(
            f"{function_name}: expected one final identity-hash assignment, "
            f"found {segment.count(old)}"
        )
    new = (
        '    identity_hash_payload = copy.deepcopy(identity)\n'
        '    identity["identity_hash"] = stable_hash(identity_hash_payload)\n'
        '    identity["identity_hash_payload"] = identity_hash_payload\n'
    )
    text = text[:function_start] + segment.replace(old, new, 1) + text[function_end:]

replace_identity_hash_in_function("_train_canonical_dpo_transfer_cell")
replace_identity_hash_in_function("_train_canonical_cold_cell")

'''
text = text[:start] + replacement + text[end:]
# Make scheduler_run_id collision-safe for sequential invocations within one process/second.
marker = '# Scheduler: classify actual execution vs reuse and persist actual successful execution times.\n'
insert = '''replace_once(\n    '    scheduler_run_id = f"queue-{int(time.time())}-{os.getpid()}"\\n',\n    '    scheduler_run_id = f"queue-{time.time_ns()}-{os.getpid()}"\\n',\n)\n'''
if text.count(marker) != 1:
    raise SystemExit(f'scheduler-run-id marker mismatch: {text.count(marker)}')
text = text.replace(marker, marker + insert, 1)
# The reuse regression targets queue/provenance behavior only.  Do not make it
# depend on unrelated task-result metric fixture fields.
old = '    monkeypatch.setattr(exp_tuning, "_require_liveness_gate", lambda *args, **kwargs: None)\n\n    def fake_subprocess_cell(**kwargs):\n'
new = '    monkeypatch.setattr(exp_tuning, "_require_liveness_gate", lambda *args, **kwargs: None)\n    monkeypatch.setattr(exp_tuning, "_coldstart_completed_task_rows", lambda *args, **kwargs: None)\n\n    def fake_subprocess_cell(**kwargs):\n'
if text.count(old) != 1:
    raise SystemExit(f'reuse-regression isolation anchor mismatch: {text.count(old)}')
text = text.replace(old, new, 1)
path.write_text(text, encoding='utf-8')
PY

bash .github/workflows/tmp_e8_terminal_evidence_patch.sh

git config user.name "github-actions[bot]"
git config user.email "41898282+github-actions[bot]@users.noreply.github.com"
if git ls-files --error-unmatch .github/workflows/tmp_e8_terminal_evidence_retry.sh >/dev/null 2>&1; then
  git rm .github/workflows/tmp_e8_terminal_evidence_retry.sh
  git commit -m "chore: remove temporary E8 terminal evidence retry helper"
  git push origin HEAD:dev/e8-multitask-baselines-capability-01
fi
