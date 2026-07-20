from pathlib import Path

path = Path('.github/workflows/temporary_e8_reciprocal_current_main_finalize.yml')
text = path.read_text(encoding='utf-8')
for rel in (
    'src/drpo/countdown_e8_alpha1_highc_scan_common.py',
    'src/drpo/countdown_e8_alpha1_highc_scan_runtime.py',
    'tests/test_countdown_e8_oracle_offline_v2_alpha1_highc_scan.py',
):
    line = f"              '{rel}',\n"
    if text.count(line) != 1:
        raise SystemExit(f'unexpected path count for {rel}')
    text = text.replace(line, '')
needle = """          for rel in copied:
              data = subprocess.check_output(['git', 'show', f'{trigger}:{rel}'])
              path = source_root / rel
              path.parent.mkdir(parents=True, exist_ok=True)
              path.write_bytes(data)

          provenance_path = source_root / 'docs/experiments/E8_RECIPROCAL_JOINT_PROVENANCE_AUDIT.json'
"""
insert = """          for rel in copied:
              data = subprocess.check_output(['git', 'show', f'{trigger}:{rel}'])
              path = source_root / rel
              path.parent.mkdir(parents=True, exist_ok=True)
              path.write_bytes(data)

          shared_paths = [
              'src/drpo/countdown_e8_alpha1_highc_scan_common.py',
              'src/drpo/countdown_e8_alpha1_highc_scan_runtime.py',
              'tests/test_countdown_e8_oracle_offline_v2_alpha1_highc_scan.py',
          ]
          shared_patch = subprocess.check_output([
              'git', 'diff', '--binary',
              'dd46727c1efefd2e6d4cdf6f3b204ec1fc58fca3',
              trigger, '--', *shared_paths,
          ])
          subprocess.run(
              ['git', '-C', str(source_root), 'apply', '--3way', '--index'],
              input=shared_patch,
              check=True,
          )

          provenance_path = source_root / 'docs/experiments/E8_RECIPROCAL_JOINT_PROVENANCE_AUDIT.json'
"""
if text.count(needle) != 1:
    raise SystemExit('insertion point not found exactly once')
path.write_text(text.replace(needle, insert), encoding='utf-8')
