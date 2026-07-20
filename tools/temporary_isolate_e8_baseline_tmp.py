from pathlib import Path

path = Path('.github/workflows/temporary_e8_reciprocal_current_main_finalize.yml')
text = path.read_text(encoding='utf-8')
old = '''          BASELINE_PYTEST_RC=$?
          (
            cd /tmp/e8-target
'''
new = '''          BASELINE_PYTEST_RC=$?
          # One legacy E7 test writes a fixed /tmp/digest-a path instead of using tmp_path.
          # Remove that baseline-run residue before executing the target tree so the
          # differential gate compares code, not cross-suite filesystem contamination.
          rm -rf /tmp/digest-a
          (
            cd /tmp/e8-target
'''
if text.count(old) != 1:
    raise SystemExit('baseline-target boundary not found exactly once')
path.write_text(text.replace(old, new), encoding='utf-8')
