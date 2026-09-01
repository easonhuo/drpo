#!/usr/bin/env bash
set -euo pipefail

python - <<'PY'
from pathlib import Path

path = Path('src/drpo/e8_experiment_config.py')
text = path.read_text(encoding='utf-8')

old = '''    if not 0.0 <= float(config["training"]["warmup_ratio"]) < 1.0:\n        raise ValueError("training.warmup_ratio must be in [0, 1)")\n'''
new = '''    if not 0.0 < float(config["training"]["warmup_ratio"]) < 1.0:\n        raise ValueError(\n            "training.warmup_ratio must be in (0, 1); the canonical trainer always uses at least one warmup step"\n        )\n'''
assert old in text
text = text.replace(old, new, 1)

old = '''    if current_id in (RHO_EXPERIMENT_ID, DENSE_EXPERIMENT_ID):\n        raise ValueError("RHO/DENSE historical IDs may not be reused for cold-start")\n'''
new = '''    if current_id in (P0_EXPERIMENT_ID, RHO_EXPERIMENT_ID, DENSE_EXPERIMENT_ID):\n        raise ValueError("P0/RHO/DENSE experiment IDs may not be reused for cold-start")\n'''
assert old in text
text = text.replace(old, new, 1)

old = '''    countdown_values = task_lambdas(config, "countdown")\n    if len(set(countdown_values)) != len(countdown_values):\n        raise ValueError("Countdown coefficient grid contains duplicates")\n'''
new = '''    countdown_values = task_lambdas(config, "countdown")\n    sentinel_values = tuple(\n        _number(value, "sweep.countdown_sentinel_coefficients entry")\n        for value in _sequence(\n            sweep.get("countdown_sentinel_coefficients"),\n            "sweep.countdown_sentinel_coefficients",\n        )\n    )\n    if sentinel_values != countdown_values:\n        raise ValueError(\n            "sweep.countdown_sentinel_coefficients must equal sweep.task_lambda.countdown"\n        )\n    if len(set(countdown_values)) != len(countdown_values):\n        raise ValueError("Countdown coefficient grid contains duplicates")\n'''
assert old in text
text = text.replace(old, new, 1)

old = '''    _integer(sweep.get("task_transfer_seed_offset"), "task_transfer_seed_offset")\n    _integer(sweep.get("tuning_seed"), "tuning_seed")\n\n    transfer_tasks = set(tasks) - {"countdown"}\n'''
new = '''    transfer_seed = _integer(\n        sweep.get("task_transfer_seed_offset"), "task_transfer_seed_offset"\n    )\n    tuning_seed = _integer(sweep.get("tuning_seed"), "tuning_seed")\n    if tuning_seed != transfer_seed:\n        raise ValueError(\n            "sweep.tuning_seed must equal sweep.task_transfer_seed_offset; cold-start has one transfer Exp seed authority"\n        )\n\n    transfer_tasks = set(tasks) - {"countdown"}\n'''
assert old in text
text = text.replace(old, new, 1)

marker = '''def _validate_canonical_and_execution(config: Mapping[str, Any]) -> None:\n'''
assert marker in text
helper = '''def _validate_data_volume_and_evaluation_capacity(\n    config: Mapping[str, Any], tasks: tuple[str, ...]\n) -> None:\n    """Reject values that the locked source pipelines/evaluators cannot consume exactly."""\n\n    split = config["split"]\n    p0_train = int(split["p0_train_rows"])\n    p0_validation = int(split["p0_validation_rows"])\n    p0_test = int(split["p0_test_rows"])\n    if p0_train <= 0:\n        raise ValueError("split.p0_train_rows must be positive for cold-start preparation")\n    if p0_train + p0_validation + p0_test != 6000:\n        raise ValueError(\n            "Cold-start P0 split sizes must consume exactly the canonical 6000-row bank per task"\n        )\n    if (\n        int(split["countdown_train_rows"]) != 6000\n        or int(split["countdown_validation_rows"]) != 500\n    ):\n        raise ValueError(\n            "Cold-start Countdown must consume the canonical 6000 train / 500 validation rows because wrapper subsampling is forbidden"\n        )\n\n    runtime = config["task_runtime"]\n    sweep = config["sweep"]\n    if tuple(sweep["countdown_seed_offsets"]):\n        countdown = runtime["countdown"]\n        if max(\n            int(countdown["greedy_prompt_rows"]),\n            int(countdown["passk_prompt_rows"]),\n        ) > int(split["countdown_validation_rows"]):\n            raise ValueError(\n                "Countdown evaluation prompt budget exceeds the configured validation partition"\n            )\n\n    for task in tasks:\n        if task == "countdown" or not task_lambdas(config, task):\n            continue\n        task_runtime = runtime[task]\n        if max(\n            int(task_runtime["greedy_prompt_rows"]),\n            int(task_runtime["passk_prompt_rows"]),\n        ) > p0_validation:\n            raise ValueError(\n                f"task_runtime.{task} evaluation prompt budget exceeds split.p0_validation_rows"\n            )\n\n\n'''
text = text.replace(marker, helper + marker, 1)

old = '''    execution = _mapping(config.get("execution"), "execution")\n    if _integer(execution.get("max_concurrent_cells"), "execution.max_concurrent_cells") != 16:\n        raise ValueError("Cold-start scheduler currently implements exactly 16 slots")\n'''
new = '''    execution = _mapping(config.get("execution"), "execution")\n    capacity = _integer(\n        execution.get("max_concurrent_cells"), "execution.max_concurrent_cells"\n    )\n    if capacity != 16:\n        raise ValueError("Cold-start scheduler currently implements exactly 16 slots")\n    expected_waves = math.ceil(int(config["sweep"]["expected_cells"]) / capacity)\n    if _integer(execution.get("expected_waves"), "execution.expected_waves", positive=True) != expected_waves:\n        raise ValueError(\n            "execution.expected_waves must match the nominal wave count derived from the expanded matrix"\n        )\n'''
assert old in text
text = text.replace(old, new, 1)

old = '''    _validate_runtime_authority_consistency(config, tasks)\n    _validate_sweep(config, tasks)\n    _validate_canonical_and_execution(config)\n'''
new = '''    _validate_runtime_authority_consistency(config, tasks)\n    _validate_sweep(config, tasks)\n    _validate_data_volume_and_evaluation_capacity(config, tasks)\n    _validate_canonical_and_execution(config)\n'''
assert old in text
text = text.replace(old, new, 1)

path.write_text(text, encoding='utf-8')

path = Path('scripts/run_e8_multitask_exp_coldstart.sh')
text = path.read_text(encoding='utf-8')
old = '''setup() {\n  check_source\n  command -v python3 >/dev/null || fail "python3 is unavailable"\n  mkdir -p "${RUNTIME_ROOT}"\n  python3 - <<'PY'\nimport sys\nassert sys.version_info >= (3, 10), sys.version\ntry:\n    import torch\nexcept ImportError as exc:\n    raise SystemExit("Install a CUDA-compatible PyTorch build before running setup") from exc\nassert torch.cuda.is_available(), "The system PyTorch build cannot see CUDA"\nPY\n  python3 -m venv --system-site-packages "${VENV_DIR}"\n  # shellcheck disable=SC1091\n  source "${VENV_DIR}/bin/activate"\n  python -m pip install --upgrade "pip==24.3.1" "setuptools==75.6.0" "wheel==0.45.1"\n  python -m pip install -r "${ROOT_DIR}/requirements/e8_multitask_exp_coldstart.txt"\n  python -m pip install --no-deps -e "${ROOT_DIR}"\n  mkdir -p "${RECOVERY_ROOT}"\n  config_preflight | tee "${RECOVERY_ROOT}/CONFIG_PREFLIGHT.json"\n  python - <<PY\nfrom huggingface_hub import snapshot_download\n'''
new = '''setup() {\n  check_source\n  command -v python3 >/dev/null || fail "python3 is unavailable"\n  mkdir -p "${RUNTIME_ROOT}"\n  python3 - <<'PY'\nimport sys\nassert sys.version_info >= (3, 10), sys.version\nPY\n  if [[ ! -x "${VENV_DIR}/bin/python" ]]; then\n    python3 -m venv --system-site-packages "${VENV_DIR}"\n  fi\n  # shellcheck disable=SC1091\n  source "${VENV_DIR}/bin/activate"\n  if ! python - <<'PY'\nimport numpy\nimport yaml\nPY\n  then\n    python -m pip install "numpy==1.26.4" "PyYAML==6.0.2"\n  fi\n  mkdir -p "${RECOVERY_ROOT}"\n  config_preflight | tee "${RECOVERY_ROOT}/CONFIG_PREFLIGHT.json"\n  python - <<'PY'\ntry:\n    import torch\nexcept ImportError as exc:\n    raise SystemExit("Install a CUDA-compatible PyTorch build before running setup") from exc\nassert torch.cuda.is_available(), "The system PyTorch build cannot see CUDA"\nPY\n  python -m pip install --upgrade "pip==24.3.1" "setuptools==75.6.0" "wheel==0.45.1"\n  python -m pip install -r "${ROOT_DIR}/requirements/e8_multitask_exp_coldstart.txt"\n  python -m pip install --no-deps -e "${ROOT_DIR}"\n  python - <<PY\nfrom huggingface_hub import snapshot_download\n'''
assert old in text
text = text.replace(old, new, 1)

old = '''    --source-file scripts/bootstrap_e8_multitask_exp_coldstart.sh \\\n    --source-file src/drpo/e8_multitask_exp_tuning.py \\\n    --source-file "${CONFIG_REPO_PATH}" \\\n'''
new = '''    --source-file scripts/bootstrap_e8_multitask_exp_coldstart.sh \\\n    --source-file src/drpo/e8_multitask_exp_tuning.py \\\n    --source-file src/drpo/e8_experiment_config.py \\\n    --source-file scripts/preflight_e8_multitask_config.py \\\n    --source-file "${CONFIG_REPO_PATH}" \\\n'''
# There are multiple similar blocks. Replace only the final formal-guard block that still lacks them.
formal_start = text.index('run_formal_guard_attempt() {')
formal_end = text.index('report_formal_success() {', formal_start)
formal = text[formal_start:formal_end]
assert old in formal
formal = formal.replace(old, new, 1)
text = text[:formal_start] + formal + text[formal_end:]
path.write_text(text, encoding='utf-8')

path = Path('tests/test_e8_multitask_p0.py')
text = path.read_text(encoding='utf-8')
append = r'''


def _generic_second_audit_config():
    from drpo import e8_multitask_exp_tuning as exp_tuning

    config = exp_tuning.load_config(Path("configs/e8_multitask_exp_lambda_curve_completion.yaml"))
    config["experiment_id"] = "EXT-C-E8-MULTITASK-EXP-SECOND-AUDIT-TEST"
    return exp_tuning, config


def test_generic_coldstart_cannot_reuse_p0_experiment_id() -> None:
    exp_tuning, config = _generic_second_audit_config()
    config["experiment_id"] = exp_tuning.P0_EXPERIMENT_ID
    with pytest.raises(ValueError, match="P0/RHO/DENSE"):
        exp_tuning.validate_config(config)


def test_generic_coldstart_redundant_authority_fields_must_agree() -> None:
    exp_tuning, config = _generic_second_audit_config()

    bad = copy.deepcopy(config)
    bad["sweep"]["tuning_seed"] += 1
    with pytest.raises(ValueError, match="one transfer Exp seed authority"):
        exp_tuning.validate_config(bad)

    bad = copy.deepcopy(config)
    bad["sweep"]["countdown_sentinel_coefficients"] = list(
        bad["sweep"]["countdown_sentinel_coefficients"]
    )[:-1]
    with pytest.raises(ValueError, match="must equal sweep.task_lambda.countdown"):
        exp_tuning.validate_config(bad)

    bad = copy.deepcopy(config)
    bad["execution"]["expected_waves"] += 1
    with pytest.raises(ValueError, match="nominal wave count"):
        exp_tuning.validate_config(bad)


def test_generic_coldstart_rejects_unconsumable_data_volumes_and_eval_budgets() -> None:
    exp_tuning, config = _generic_second_audit_config()

    bad = copy.deepcopy(config)
    bad["split"]["p0_train_rows"] -= 1
    with pytest.raises(ValueError, match="canonical 6000-row bank"):
        exp_tuning.validate_config(bad)

    bad = copy.deepcopy(config)
    bad["split"]["countdown_train_rows"] -= 1
    with pytest.raises(ValueError, match="6000 train / 500 validation"):
        exp_tuning.validate_config(bad)

    bad = copy.deepcopy(config)
    bad["task_runtime"]["word_sorting"]["greedy_prompt_rows"] = (
        int(bad["split"]["p0_validation_rows"]) + 1
    )
    with pytest.raises(ValueError, match="word_sorting evaluation prompt budget"):
        exp_tuning.validate_config(bad)


def test_generic_coldstart_rejects_zero_warmup_that_legacy_trainer_cannot_express() -> None:
    exp_tuning, config = _generic_second_audit_config()
    config["training"]["warmup_ratio"] = 0.0
    with pytest.raises(ValueError, match="at least one warmup step"):
        exp_tuning.validate_config(config)


def test_coldstart_setup_preflights_before_expensive_runtime_setup() -> None:
    runner = Path("scripts/run_e8_multitask_exp_coldstart.sh").read_text(encoding="utf-8")
    setup = runner[runner.index("setup() {") : runner.index("runtime_ready() {")]
    assert setup.index("config_preflight | tee") < setup.index(
        'pip install -r "${ROOT_DIR}/requirements/e8_multitask_exp_coldstart.txt"'
    )
    assert setup.index("config_preflight | tee") < setup.index("torch.cuda.is_available()")
    assert setup.index("config_preflight | tee") < setup.index("snapshot_download")


def test_formal_guard_provenance_includes_config_authority_paths() -> None:
    runner = Path("scripts/run_e8_multitask_exp_coldstart.sh").read_text(encoding="utf-8")
    formal = runner[
        runner.index("run_formal_guard_attempt() {") : runner.index("report_formal_success() {")
    ]
    assert "--source-file src/drpo/e8_experiment_config.py" in formal
    assert "--source-file scripts/preflight_e8_multitask_config.py" in formal
'''
if 'def test_generic_coldstart_cannot_reuse_p0_experiment_id()' not in text:
    text += append
path.write_text(text, encoding='utf-8')
PY

ruff format src/drpo/e8_experiment_config.py tests/test_e8_multitask_p0.py
python -m py_compile \
  src/drpo/e8_experiment_config.py \
  scripts/preflight_e8_multitask_config.py \
  src/drpo/e8_multitask_exp_tuning.py \
  tests/test_e8_multitask_p0.py
bash -n scripts/run_e8_multitask_exp_coldstart.sh
bash -n scripts/bootstrap_e8_multitask_exp_coldstart.sh
bash -n scripts/run_e8_multitask_exp_lambda_completion.sh
ruff check src/drpo/e8_experiment_config.py scripts/preflight_e8_multitask_config.py tests/test_e8_multitask_p0.py
python -m pytest -q tests/test_e8_multitask_p0.py
git diff --check

rm -f .github/workflows/e8-second-audit-fix-once.yml scripts/.tmp_e8_second_audit_fix.sh

git config user.name github-actions[bot]
git config user.email 41898282+github-actions[bot]@users.noreply.github.com
git add -A
git diff --cached --check
git commit -m "fix: close E8 second config authority audit"
git push origin HEAD:dev/e8-config-driven-sweep-01
