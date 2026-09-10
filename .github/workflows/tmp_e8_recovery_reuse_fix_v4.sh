#!/usr/bin/env bash
set -euo pipefail
python - <<'PY'
from pathlib import Path
path = Path('.github/workflows/tmp_e8_recovery_reuse_fix_v2.sh')
text = path.read_text(encoding='utf-8')
replacements = {
    "    scope.write_text(text + block + '\\n', encoding='utf-8')": "    scope.write_text((text + block).rstrip() + '\\n', encoding='utf-8')",
    "    test_path.write_text(tests + '\\n', encoding='utf-8')": "    test_path.write_text(tests.rstrip() + '\\n', encoding='utf-8')",
}
for old, new in replacements.items():
    if text.count(old) != 1:
        raise SystemExit(f'normalization anchor count={text.count(old)} for {old}')
    text = text.replace(old, new, 1)
old_cleanup = '''  .github/workflows/tmp_e8_recovery_reuse_fix_v2.sh \\
  .github/workflows/tmp_e8_recovery_reuse_fix_v2.yml || true
'''
new_cleanup = '''  .github/workflows/tmp_e8_recovery_reuse_fix_v2.sh \\
  .github/workflows/tmp_e8_recovery_reuse_fix_v2.yml \\
  .github/workflows/tmp_e8_recovery_reuse_fix_v3.sh \\
  .github/workflows/tmp_e8_recovery_reuse_fix_v3.yml \\
  .github/workflows/tmp_e8_recovery_reuse_fix_v4.sh \\
  .github/workflows/tmp_e8_recovery_reuse_fix_v4.yml || true
'''
if text.count(old_cleanup) != 1:
    raise SystemExit(f'cleanup anchor count={text.count(old_cleanup)}')
path.write_text(text.replace(old_cleanup, new_cleanup, 1), encoding='utf-8')
PY
bash .github/workflows/tmp_e8_recovery_reuse_fix_v2.sh
