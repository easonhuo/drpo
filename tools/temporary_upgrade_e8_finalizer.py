from pathlib import Path

path = Path('.github/workflows/temporary_e8_reciprocal_current_main_finalize.yml')
text = path.read_text(encoding='utf-8')

needle = """          registry_path.write_text(registry_after, encoding='utf-8')

          delta_path = source_root / delta_rel
"""
replacement = """          registry_path.write_text(registry_after, encoding='utf-8')

          formal_test_path = source_root / 'tests/test_formal_execution_channel.py'
          formal_test_text = formal_test_path.read_text(encoding='utf-8')
          expected_pilot_count = sum(
              entry.get('execution_class') == 'pilot'
              for entry in parsed_after['experiments']
          )
          old_pilot_line = '        "pilot": 11,\n'
          new_pilot_line = f'        "pilot": {expected_pilot_count},\n'
          if formal_test_text.count(old_pilot_line) != 1:
              raise SystemExit('formal-channel pilot-count snapshot is not the expected current-main form')
          formal_test_path.write_text(
              formal_test_text.replace(old_pilot_line, new_pilot_line),
              encoding='utf-8',
          )

          delta_path = source_root / delta_rel
"""
if text.count(needle) != 1:
    raise SystemExit('registry insertion point not found exactly once')
text = text.replace(needle, replacement)

needle = """            tests/test_countdown_e8_oracle_offline_v2_alpha1_highc_scan.py \\
            docs/experiments/E8_RECIPROCAL_JOINT_CLOSURE.md \\
"""
replacement = """            tests/test_countdown_e8_oracle_offline_v2_alpha1_highc_scan.py \\
            tests/test_formal_execution_channel.py \\
            docs/experiments/E8_RECIPROCAL_JOINT_CLOSURE.md \\
"""
if text.count(needle) != 1:
    raise SystemExit('git-add insertion point not found exactly once')
text = text.replace(needle, replacement)

old = """          python -m pytest -q
          ruff check .
          git diff --check "$CURRENT_MAIN".."$TARGET_COMMIT"
"""
new = """          python -m pytest -q tests/test_countdown_e8_oracle_offline_v2_alpha1_highc_scan.py
          python -m pytest -q tests/test_formal_execution_channel.py::test_current_registry_uses_canonical_channel
          set +e
          (
            cd /tmp/e8-trusted
            PYTHONPATH=/tmp/e8-trusted/src:/tmp/e8-trusted/scripts python -m pytest -q
          ) > /tmp/e8-baseline-pytest.log 2>&1
          BASELINE_PYTEST_RC=$?
          (
            cd /tmp/e8-target
            PYTHONPATH=/tmp/e8-target/src:/tmp/e8-target/scripts python -m pytest -q
          ) > /tmp/e8-target-pytest.log 2>&1
          TARGET_PYTEST_RC=$?
          set -e
          BASELINE_PYTEST_RC="$BASELINE_PYTEST_RC" TARGET_PYTEST_RC="$TARGET_PYTEST_RC" python - <<'PY'
          import json
          import os
          import re
          from pathlib import Path

          def failures(path: str) -> set[str]:
              rows: set[str] = set()
              for line in Path(path).read_text(encoding='utf-8', errors='replace').splitlines():
                  match = re.match(r'(?:FAILED|ERROR)\\s+(\\S+)', line)
                  if match:
                      rows.add(match.group(1))
              return rows

          baseline = failures('/tmp/e8-baseline-pytest.log')
          target = failures('/tmp/e8-target-pytest.log')
          new_failures = sorted(target - baseline)
          report = {
              'schema_version': 1,
              'base_commit': os.environ['CURRENT_MAIN'],
              'baseline_exit_code': int(os.environ['BASELINE_PYTEST_RC']),
              'target_exit_code': int(os.environ['TARGET_PYTEST_RC']),
              'baseline_failures': sorted(baseline),
              'target_failures': sorted(target),
              'new_failures': new_failures,
          }
          print(json.dumps(report, indent=2, sort_keys=True))
          if new_failures:
              raise SystemExit(f'target introduced new pytest failures: {new_failures}')
          if int(os.environ['BASELINE_PYTEST_RC']) == 0 and int(os.environ['TARGET_PYTEST_RC']) != 0:
              raise SystemExit('current main is pytest-clean but target is not')
          PY
          ruff check \\
            src/drpo/countdown_e8_alpha1_highc_scan_common.py \\
            src/drpo/countdown_e8_alpha1_highc_scan_runtime.py \\
            tests/test_countdown_e8_oracle_offline_v2_alpha1_highc_scan.py \\
            tests/test_formal_execution_channel.py
          git diff --check "$CURRENT_MAIN".."$TARGET_COMMIT"
"""
if text.count(old) != 1:
    raise SystemExit('acceptance block not found exactly once')
path.write_text(text.replace(old, new), encoding='utf-8')
