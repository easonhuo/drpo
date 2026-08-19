from pathlib import Path

script_path = Path('/tmp/tmp_e8_dynamic_queue_apply.py')
text = script_path.read_text(encoding='utf-8')
old = '''old_asserts = dedent(''' + "'''" + '''\n    assert result[\"queue_audit\"][\"wave_barriers_respected\"]\n    assert result[\"queue_audit\"][\"wave_count\"] == 13\n    assert result[\"queue_audit\"][\"slots_per_gpu\"] == 2\n''' + "'''" + ''')\nnew_asserts = dedent(''' + "'''" + '''\n    assert result[\"queue_audit\"][\"wave_barriers_respected\"] is False\n    assert result[\"queue_audit\"][\"later_cell_started_before_first_batch_finished\"]\n    assert result[\"queue_audit\"][\"wave_count\"] == 13\n    assert result[\"queue_audit\"][\"slots_per_gpu\"] == 2\n''' + "'''" + ''')'''
new = '''old_asserts = (\n    '    assert result[\"queue_audit\"][\"wave_barriers_respected\"]\\n'\n    '    assert result[\"queue_audit\"][\"wave_count\"] == 13\\n'\n    '    assert result[\"queue_audit\"][\"slots_per_gpu\"] == 2\\n'\n)\nnew_asserts = (\n    '    assert result[\"queue_audit\"][\"wave_barriers_respected\"] is False\\n'\n    '    assert result[\"queue_audit\"][\"later_cell_started_before_first_batch_finished\"]\\n'\n    '    assert result[\"queue_audit\"][\"wave_count\"] == 13\\n'\n    '    assert result[\"queue_audit\"][\"slots_per_gpu\"] == 2\\n'\n)'''
if text.count(old) != 1:
    raise SystemExit(f'wrapper expected one assertion-template block, found {text.count(old)}')
text = text.replace(old, new, 1)
exec(compile(text, str(script_path), 'exec'))
