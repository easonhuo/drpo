#!/usr/bin/env bash
set -euo pipefail

python - <<'PY'
from pathlib import Path

path = Path('src/drpo/e8_multitask_exp_tuning.py')
text = path.read_text()
old = '''def load_config(path: str | Path = DEFAULT_CONFIG) -> dict[str, Any]:
    source = Path(path)
    repo_root = Path(__file__).resolve().parents[2]
    if not source.is_absolute():
        source = repo_root / source
    value = yaml.safe_load(source.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError("Configuration root must be a mapping")
    validate_config(value)
    experiment_config.validate_historical_config_identity(source, value, repo_root=repo_root)
    return value
'''
new = '''def load_config(path: str | Path = DEFAULT_CONFIG) -> dict[str, Any]:
    """Load one externally selected, repository-tracked E8 config fail-closed."""

    repo_root = Path(__file__).resolve().parents[2]
    source, _, _ = experiment_config.require_tracked_config(path, repo_root)
    value = experiment_config.load_strict_yaml(source)
    validate_config(value)
    experiment_config.validate_historical_config_identity(source, value, repo_root=repo_root)
    return value


def _load_internal_config(path: str | Path) -> dict[str, Any]:
    """Load one runtime-generated internal config through the same strict validator."""

    value = experiment_config.load_strict_yaml(path)
    validate_config(value)
    if not experiment_config.is_engineering_self_test_config(value):
        raise ValueError("internal E8 config loading is restricted to engineering self-test configs")
    return value
'''
if old not in text:
    raise SystemExit('load_config anchor not found')
text = text.replace(old, new, 1)
text = text.replace('recovered = load_config(engineering_config)', 'recovered = _load_internal_config(engineering_config)', 1)
text = text.replace('recovered_config = load_config(config_path)', 'recovered_config = _load_internal_config(config_path)', 1)
path.write_text(text)

path = Path('src/drpo/e8_experiment_config.py')
text = path.read_text()
old = '''    historical = is_historical_coldstart_config(config) and not is_engineering_self_test_config(config)
'''
new = '''    # Internal engineering self-tests derived from an immutable historical config
    # retain that historical scientific matrix. The engineering marker changes the
    # backend/evidence class, not the reviewed scientific matrix semantics.
    historical = is_historical_coldstart_config(config)
'''
if old not in text:
    raise SystemExit('historical sweep anchor not found')
text = text.replace(old, new, 1)
text = text.replace(
    '''    if not historical and not countdown_seeds:\n        if countdown_values or sentinel_values or countdown_positive:\n            raise ValueError(\n                "generic inactive Countdown requires empty lambda/sentinel grids and countdown_include_positive_only=false"\n            )\n''',
    '''    if (\n        not historical\n        and not countdown_seeds\n        and (countdown_values or sentinel_values or countdown_positive)\n    ):\n        raise ValueError(\n            "generic inactive Countdown requires empty lambda/sentinel grids and countdown_include_positive_only=false"\n        )\n''',
    1,
)
path.write_text(text)
PY

python - <<'PY'
from pathlib import Path

path = Path('tests/test_e8_multitask_p0.py')
text = path.read_text()
anchor = '''def test_new_coldstart_config_controls_materialized_runtime_without_core_edits(\n    tmp_path: Path,\n) -> None:\n'''
helper = '''def _generic_coldstart_test_config():\n    from drpo import e8_multitask_exp_tuning as exp_tuning\n\n    config = exp_tuning.load_config(\n        Path("configs/e8_multitask_exp_lambda_curve_completion.yaml")\n    )\n    # Frozen predecessor metadata is not part of the generic reviewed schema.\n    config.pop("historical_curve_anchor", None)\n    sweep = config["sweep"]\n    sweep["countdown_seed_offsets"] = []\n    sweep["countdown_include_positive_only"] = False\n    sweep["include_global_endpoint"] = False\n    sweep["task_lambda"]["countdown"] = []\n    sweep["countdown_sentinel_coefficients"] = []\n    return config\n\n\n'''
if anchor not in text:
    raise SystemExit('generic helper anchor not found')
text = text.replace(anchor, helper + anchor, 1)
needle = 'config = exp_tuning.load_config(Path("configs/e8_multitask_exp_lambda_curve_completion.yaml"))'
head, tail = text.split(helper, 1)
count = tail.count(needle)
if count < 5:
    raise SystemExit(f'expected at least five generic test anchors, found {count}')
tail = tail.replace(needle, 'config = _generic_coldstart_test_config()', 5)
text = head + helper + tail
old = '''    try:\n        with pytest.raises(ValueError, match="canonical config path"):\n            exp_tuning.load_config(copied)\n    finally:\n        copied.unlink(missing_ok=True)\n'''
new = '''    try:\n        with pytest.raises(ValueError, match="not Git-tracked"):\n            exp_tuning.load_config(copied)\n        from drpo import e8_experiment_config as experiment_config\n\n        frozen = exp_tuning.load_config(source)\n        with pytest.raises(ValueError, match="canonical config path"):\n            experiment_config.validate_historical_config_identity(\n                copied, frozen, repo_root=Path.cwd()\n            )\n    finally:\n        copied.unlink(missing_ok=True)\n'''
if old not in text:
    raise SystemExit('historical identity test anchor not found')
text = text.replace(old, new, 1)
text += '''\n\ndef test_e8_strict_yaml_rejects_duplicate_nested_keys(tmp_path: Path) -> None:\n    from drpo import e8_experiment_config as experiment_config\n\n    path = tmp_path / "duplicate.yaml"\n    path.write_text("schema_version: 1\\nsweep:\\n  method: exponential\\n  method: linear\\n")\n    with pytest.raises(ValueError, match="duplicate key"):\n        experiment_config.load_strict_yaml(path)\n\n\ndef test_e8_internal_config_loader_is_explicit(tmp_path: Path) -> None:\n    from drpo import e8_multitask_exp_tuning as tuning\n\n    path = tmp_path / "historical-copy.yaml"\n    path.write_text(\n        Path("configs/e8_multitask_exp_coldstart.yaml").read_text(encoding="utf-8"),\n        encoding="utf-8",\n    )\n    with pytest.raises(ValueError, match="restricted to engineering self-test"):\n        tuning._load_internal_config(path)\n'''
path.write_text(text)
PY

python -m py_compile src/drpo/e8_experiment_config.py scripts/preflight_e8_multitask_config.py src/drpo/e8_multitask_exp_tuning.py tests/test_e8_multitask_p0.py
bash -n scripts/run_e8_multitask_exp_coldstart.sh
bash -n scripts/bootstrap_e8_multitask_exp_coldstart.sh
bash -n scripts/run_e8_multitask_exp_lambda_completion.sh
python -m pytest -q tests/test_e8_multitask_p0.py
ruff check src/drpo/e8_experiment_config.py scripts/preflight_e8_multitask_config.py tests/test_e8_multitask_p0.py
ruff format --check src/drpo/e8_experiment_config.py scripts/preflight_e8_multitask_config.py tests/test_e8_multitask_p0.py
git diff --check

git rm .github/workflows/e8-third-audit-loader-fix-once.yml scripts/.tmp_e8_third_audit_loader_fix.sh
git add src/drpo/e8_experiment_config.py src/drpo/e8_multitask_exp_tuning.py tests/test_e8_multitask_p0.py
git diff --cached --check
git config user.name github-actions[bot]
git config user.email 41898282+github-actions[bot]@users.noreply.github.com
git commit -m 'fix(e8): close strict config loader boundary'
git push origin HEAD:dev/e8-config-driven-sweep-01
