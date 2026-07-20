from pathlib import Path

path = Path('.github/workflows/temporary_e8_reciprocal_current_main_finalize.yml')
text = path.read_text(encoding='utf-8')
old = '''          old_pilot_line = '        "pilot": 11,
'
          new_pilot_line = f'        "pilot": {expected_pilot_count},
'
'''
new = '''          old_pilot_line = '        "pilot": 11,' + chr(10)
          new_pilot_line = f'        "pilot": {expected_pilot_count},' + chr(10)
'''
if text.count(old) != 1:
    raise SystemExit('malformed pilot-count block not found exactly once')
path.write_text(text.replace(old, new), encoding='utf-8')
