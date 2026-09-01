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
PY

cat >> tests/test_e8_multitask_p0.py <<'PYTEST'


def test_e8_strict_yaml_rejects_duplicate_nested_keys(tmp_path: Path) -> None:
    from drpo import e8_experiment_config as experiment_config

    path = tmp_path / "duplicate.yaml"
    path.write_text("schema_version: 1\nsweep:\n  method: exponential\n  method: linear\n")
    with pytest.raises(ValueError, match="duplicate key"):
        experiment_config.load_strict_yaml(path)


def test_e8_internal_config_loader_is_explicit_and_rejects_scientific_config(tmp_path: Path) -> None:
    from drpo import e8_multitask_exp_tuning as tuning

    path = tmp_path / "ordinary.yaml"
    path.write_text("schema_version: 1\nexperiment_id: X\n")
    with pytest.raises((KeyError, ValueError)):
        tuning._load_internal_config(path)
PYTEST

python -m py_compile src/drpo/e8_experiment_config.py scripts/preflight_e8_multitask_config.py src/drpo/e8_multitask_exp_tuning.py tests/test_e8_multitask_p0.py
bash -n scripts/run_e8_multitask_exp_coldstart.sh
bash -n scripts/bootstrap_e8_multitask_exp_coldstart.sh
bash -n scripts/run_e8_multitask_exp_lambda_completion.sh
python -m pytest -q tests/test_e8_multitask_p0.py
ruff check src/drpo/e8_experiment_config.py scripts/preflight_e8_multitask_config.py tests/test_e8_multitask_p0.py
ruff format --check src/drpo/e8_experiment_config.py scripts/preflight_e8_multitask_config.py tests/test_e8_multitask_p0.py
git diff --check

git rm .github/workflows/e8-third-audit-loader-fix-once.yml scripts/.tmp_e8_third_audit_loader_fix.sh
git add src/drpo/e8_multitask_exp_tuning.py tests/test_e8_multitask_p0.py scripts/bootstrap_e8_multitask_exp_coldstart.sh
git diff --cached --check
git config user.name github-actions[bot]
git config user.email 41898282+github-actions[bot]@users.noreply.github.com
git commit -m 'fix(e8): close strict config loader boundary'
git push origin HEAD:dev/e8-config-driven-sweep-01
