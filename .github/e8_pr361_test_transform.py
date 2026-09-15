from pathlib import Path

path = Path("tests/test_e8_multitask_p0.py")
source = path.read_text(encoding="utf-8")
old = '''    reusable = inspect.getsource(exp_tuning._reusable_cell_manifests)\n    assert 'value.get("canonical_summary"' in reusable\n    assert 'value.get("canonical_summary_sha256"' in reusable\n'''
new = '''    from drpo import e8_multitask_runtime as runtime\n\n    reusable = inspect.getsource(runtime.reusable_cell_manifests)\n    assert 'value.get("canonical_summary"' in reusable\n    assert 'value.get("canonical_summary_sha256"' in reusable\n'''
if source.count(old) != 1:
    raise RuntimeError("Expected exactly one DPO recovery-source assertion block")
path.write_text(source.replace(old, new, 1), encoding="utf-8")
