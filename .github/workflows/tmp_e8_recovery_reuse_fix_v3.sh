#!/usr/bin/env bash
set -euo pipefail
python - <<'PY'
from pathlib import Path
path = Path('.github/workflows/tmp_e8_recovery_reuse_fix_v2.sh')
text = path.read_text(encoding='utf-8')
old = "    scope.write_text(text + block + '\\n', encoding='utf-8')"
new = "    scope.write_text((text + block).rstrip() + '\\n', encoding='utf-8')"
if text.count(old) != 1:
    raise SystemExit(f'EOF anchor count={text.count(old)}')
text = text.replace(old, new, 1)
old_cleanup = '''  .github/workflows/tmp_e8_recovery_reuse_fix_v2.sh \\
  .github/workflows/tmp_e8_recovery_reuse_fix_v2.yml || true
'''
new_cleanup = '''  .github/workflows/tmp_e8_recovery_reuse_fix_v2.sh \\
  .github/workflows/tmp_e8_recovery_reuse_fix_v2.yml \\
  .github/workflows/tmp_e8_recovery_reuse_fix_v3.sh \\
  .github/workflows/tmp_e8_recovery_reuse_fix_v3.yml || true
'''
if text.count(old_cleanup) != 1:
    raise SystemExit(f'cleanup anchor count={text.count(old_cleanup)}')
path.write_text(text.replace(old_cleanup, new_cleanup, 1), encoding='utf-8')
PY
bash .github/workflows/tmp_e8_recovery_reuse_fix_v2.sh
