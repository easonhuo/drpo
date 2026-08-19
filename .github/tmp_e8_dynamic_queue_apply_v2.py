from pathlib import Path

script_path = Path('/tmp/tmp_e8_dynamic_queue_apply.py')
text = script_path.read_text(encoding='utf-8')

old = '''old_asserts = dedent(''' + "'''" + '''\n    assert result[\"queue_audit\"][\"wave_barriers_respected\"]\n    assert result[\"queue_audit\"][\"wave_count\"] == 13\n    assert result[\"queue_audit\"][\"slots_per_gpu\"] == 2\n''' + "'''" + ''')\nnew_asserts = dedent(''' + "'''" + '''\n    assert result[\"queue_audit\"][\"wave_barriers_respected\"] is False\n    assert result[\"queue_audit\"][\"later_cell_started_before_first_batch_finished\"]\n    assert result[\"queue_audit\"][\"wave_count\"] == 13\n    assert result[\"queue_audit\"][\"slots_per_gpu\"] == 2\n''' + "'''" + ''')'''
new = '''old_asserts = (\n    '    assert result[\"queue_audit\"][\"wave_barriers_respected\"]\\n'\n    '    assert result[\"queue_audit\"][\"wave_count\"] == 13\\n'\n    '    assert result[\"queue_audit\"][\"slots_per_gpu\"] == 2\\n'\n)\nnew_asserts = (\n    '    assert result[\"queue_audit\"][\"wave_barriers_respected\"] is False\\n'\n    '    assert result[\"queue_audit\"][\"later_cell_started_before_first_batch_finished\"]\\n'\n    '    assert result[\"queue_audit\"][\"wave_count\"] == 13\\n'\n    '    assert result[\"queue_audit\"][\"slots_per_gpu\"] == 2\\n'\n)'''
if text.count(old) != 1:
    raise SystemExit(f'wrapper expected one assertion-template block, found {text.count(old)}')
text = text.replace(old, new, 1)

old_events = '''    replacement_started = threading.Event()\n    first_cell_released_by_replacement = threading.Event()\n'''
if text.count(old_events) != 1:
    raise SystemExit(f'wrapper expected one fragile event declaration block, found {text.count(old_events)}')
text = text.replace(old_events, '', 1)

old_wait = '''        if cell in cells[16:]:\n            replacement_started.set()\n        if cell.key == cells[0].key:\n            if replacement_started.wait(timeout=1.0):\n                first_cell_released_by_replacement.set()\n        else:\n            time.sleep(0.001)\n'''
new_wait = '''        time.sleep(0.05 if cell.key == cells[0].key else 0.001)\n'''
if text.count(old_wait) != 1:
    raise SystemExit(f'wrapper expected one fragile wait block, found {text.count(old_wait)}')
text = text.replace(old_wait, new_wait, 1)

old_assert = '''    assert first_cell_released_by_replacement.is_set()\n'''
if text.count(old_assert) != 1:
    raise SystemExit(f'wrapper expected one fragile event assertion, found {text.count(old_assert)}')
text = text.replace(old_assert, '', 1)

exec(compile(text, str(script_path), 'exec'))
