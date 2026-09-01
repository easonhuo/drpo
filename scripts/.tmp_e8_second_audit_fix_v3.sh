#!/usr/bin/env bash
set -euo pipefail

python - <<'PY'
from pathlib import Path

path = Path('src/drpo/e8_experiment_config.py')
text = path.read_text(encoding='utf-8')

replacements = [
(
'''    if not 0.0 <= float(config["training"]["warmup_ratio"]) < 1.0:
        raise ValueError("training.warmup_ratio must be in [0, 1)")
''',
'''    if not 0.0 < float(config["training"]["warmup_ratio"]) < 1.0:
        raise ValueError(
            "training.warmup_ratio must be in (0, 1); the canonical trainer always uses at least one warmup step"
        )
'''
),
(
'''    if current_id in (RHO_EXPERIMENT_ID, DENSE_EXPERIMENT_ID):
        raise ValueError("RHO/DENSE historical IDs may not be reused for cold-start")
''',
'''    if current_id in (P0_EXPERIMENT_ID, RHO_EXPERIMENT_ID, DENSE_EXPERIMENT_ID):
        raise ValueError("P0/RHO/DENSE experiment IDs may not be reused for cold-start")
'''
),
(
'''    countdown_values = task_lambdas(config, "countdown")
    if len(set(countdown_values)) != len(countdown_values):
        raise ValueError("Countdown coefficient grid contains duplicates")
''',
'''    countdown_values = task_lambdas(config, "countdown")
    sentinel_values = tuple(
        _number(value, "sweep.countdown_sentinel_coefficients entry")
        for value in _sequence(
            sweep.get("countdown_sentinel_coefficients"),
            "sweep.countdown_sentinel_coefficients",
        )
    )
    if sentinel_values != countdown_values:
        raise ValueError(
            "sweep.countdown_sentinel_coefficients must equal sweep.task_lambda.countdown"
        )
    if len(set(countdown_values)) != len(countdown_values):
        raise ValueError("Countdown coefficient grid contains duplicates")
'''
),
(
'''    _integer(sweep.get("task_transfer_seed_offset"), "task_transfer_seed_offset")
    _integer(sweep.get("tuning_seed"), "tuning_seed")

    transfer_tasks = set(tasks) - {"countdown"}
''',
'''    transfer_seed = _integer(
        sweep.get("task_transfer_seed_offset"), "task_transfer_seed_offset"
    )
    tuning_seed = _integer(sweep.get("tuning_seed"), "tuning_seed")
    if tuning_seed != transfer_seed:
        raise ValueError(
            "sweep.tuning_seed must equal sweep.task_transfer_seed_offset; cold-start has one transfer Exp seed authority"
        )

    transfer_tasks = set(tasks) - {"countdown"}
'''
),
(
'''    execution = _mapping(config.get("execution"), "execution")
    if _integer(execution.get("max_concurrent_cells"), "execution.max_concurrent_cells") != 16:
        raise ValueError("Cold-start scheduler currently implements exactly 16 slots")
''',
'''    execution = _mapping(config.get("execution"), "execution")
    capacity = _integer(
        execution.get("max_concurrent_cells"), "execution.max_concurrent_cells"
    )
    if capacity != 16:
        raise ValueError("Cold-start scheduler currently implements exactly 16 slots")
    expected_waves = math.ceil(int(config["sweep"]["expected_cells"]) / capacity)
    if (
        _integer(
            execution.get("expected_waves"),
            "execution.expected_waves",
            positive=True,
        )
        != expected_waves
    ):
        raise ValueError(
            "execution.expected_waves must match the nominal wave count derived from the expanded matrix"
        )
'''
),
(
'''    _validate_runtime_authority_consistency(config, tasks)
    _validate_sweep(config, tasks)
    _validate_canonical_and_execution(config)
''',
'''    _validate_runtime_authority_consistency(config, tasks)
    _validate_sweep(config, tasks)
    _validate_data_volume_and_evaluation_capacity(config, tasks)
    _validate_canonical_and_execution(config)
'''
),
]
for old, new in replacements:
    if old not in text:
        raise SystemExit(f'missing expected source block:\n{old}')
    text = text.replace(old, new, 1)

marker = 'def _validate_canonical_and_execution(config: Mapping[str, Any]) -> None:\n'
if marker not in text:
    raise SystemExit('missing canonical/execution validator marker')
helper = '''def _validate_data_volume_and_evaluation_capacity(
    config: Mapping[str, Any], tasks: tuple[str, ...]
) -> None:
    """Reject formal values the locked source/evaluator pipeline cannot consume exactly."""

    engineering = config.get("engineering_self_test")
    if isinstance(engineering, Mapping) and engineering.get("placeholder_backend") is True:
        return

    split = config["split"]
    p0_train = int(split["p0_train_rows"])
    p0_validation = int(split["p0_validation_rows"])
    p0_test = int(split["p0_test_rows"])
    if p0_train <= 0:
        raise ValueError("split.p0_train_rows must be positive for cold-start preparation")
    if p0_train + p0_validation + p0_test != 6000:
        raise ValueError(
            "Cold-start P0 split sizes must consume exactly the canonical 6000-row bank per task"
        )
    if (
        int(split["countdown_train_rows"]) != 6000
        or int(split["countdown_validation_rows"]) != 500
    ):
        raise ValueError(
            "Cold-start Countdown must consume the canonical 6000 train / 500 validation rows because wrapper subsampling is forbidden"
        )

    runtime = config["task_runtime"]
    sweep = config["sweep"]
    if tuple(sweep["countdown_seed_offsets"]):
        countdown = runtime["countdown"]
        if max(
            int(countdown["greedy_prompt_rows"]),
            int(countdown["passk_prompt_rows"]),
        ) > int(split["countdown_validation_rows"]):
            raise ValueError(
                "Countdown evaluation prompt budget exceeds the configured validation partition"
            )

    for task in tasks:
        if task == "countdown" or not task_lambdas(config, task):
            continue
        task_runtime = runtime[task]
        if max(
            int(task_runtime["greedy_prompt_rows"]),
            int(task_runtime["passk_prompt_rows"]),
        ) > p0_validation:
            raise ValueError(
                f"task_runtime.{task} evaluation prompt budget exceeds split.p0_validation_rows"
            )


'''
text = text.replace(marker, helper + marker, 1)
path.write_text(text, encoding='utf-8')

path = Path('scripts/run_e8_multitask_exp_coldstart.sh')
text = path.read_text(encoding='utf-8')
old_setup = '''setup() {
  check_source
  command -v python3 >/dev/null || fail "python3 is unavailable"
  mkdir -p "${RUNTIME_ROOT}"
  python3 - <<'PY'
import sys
assert sys.version_info >= (3, 10), sys.version
try:
    import torch
except ImportError as exc:
    raise SystemExit("Install a CUDA-compatible PyTorch build before running setup") from exc
assert torch.cuda.is_available(), "The system PyTorch build cannot see CUDA"
PY
  python3 -m venv --system-site-packages "${VENV_DIR}"
  # shellcheck disable=SC1091
  source "${VENV_DIR}/bin/activate"
  python -m pip install --upgrade "pip==24.3.1" "setuptools==75.6.0" "wheel==0.45.1"
  python -m pip install -r "${ROOT_DIR}/requirements/e8_multitask_exp_coldstart.txt"
  python -m pip install --no-deps -e "${ROOT_DIR}"
  mkdir -p "${RECOVERY_ROOT}"
  config_preflight | tee "${RECOVERY_ROOT}/CONFIG_PREFLIGHT.json"
  python - <<PY
from huggingface_hub import snapshot_download
'''
new_setup = '''setup() {
  check_source
  command -v python3 >/dev/null || fail "python3 is unavailable"
  mkdir -p "${RUNTIME_ROOT}"
  python3 - <<'PY'
import sys
assert sys.version_info >= (3, 10), sys.version
PY
  if [[ ! -x "${VENV_DIR}/bin/python" ]]; then
    python3 -m venv --system-site-packages "${VENV_DIR}"
  fi
  # shellcheck disable=SC1091
  source "${VENV_DIR}/bin/activate"
  if ! python - <<'PY'
import numpy
import yaml
PY
  then
    python -m pip install "numpy==1.26.4" "PyYAML==6.0.2"
  fi
  mkdir -p "${RECOVERY_ROOT}"
  config_preflight | tee "${RECOVERY_ROOT}/CONFIG_PREFLIGHT.json"
  python - <<'PY'
try:
    import torch
except ImportError as exc:
    raise SystemExit("Install a CUDA-compatible PyTorch build before running setup") from exc
assert torch.cuda.is_available(), "The system PyTorch build cannot see CUDA"
PY
  python -m pip install --upgrade "pip==24.3.1" "setuptools==75.6.0" "wheel==0.45.1"
  python -m pip install -r "${ROOT_DIR}/requirements/e8_multitask_exp_coldstart.txt"
  python -m pip install --no-deps -e "${ROOT_DIR}"
  python - <<PY
from huggingface_hub import snapshot_download
'''
if old_setup not in text:
    raise SystemExit('missing expected setup block')
text = text.replace(old_setup, new_setup, 1)

formal_start = text.index('run_formal_guard_attempt() {')
formal_end = text.index('report_formal_success() {', formal_start)
formal = text[formal_start:formal_end]
old_sources = '''    --source-file scripts/bootstrap_e8_multitask_exp_coldstart.sh \\
    --source-file src/drpo/e8_multitask_exp_tuning.py \\
    --source-file "${CONFIG_REPO_PATH}" \\
'''
new_sources = '''    --source-file scripts/bootstrap_e8_multitask_exp_coldstart.sh \\
    --source-file src/drpo/e8_multitask_exp_tuning.py \\
    --source-file src/drpo/e8_experiment_config.py \\
    --source-file scripts/preflight_e8_multitask_config.py \\
    --source-file "${CONFIG_REPO_PATH}" \\
'''
if old_sources not in formal:
    raise SystemExit('formal guard source block already differs unexpectedly')
formal = formal.replace(old_sources, new_sources, 1)
text = text[:formal_start] + formal + text[formal_end:]
path.write_text(text, encoding='utf-8')

path = Path('tests/test_e8_multitask_p0.py')
text = path.read_text(encoding='utf-8')
if 'def test_generic_coldstart_cannot_reuse_p0_experiment_id()' in text:
    raise SystemExit('second-audit tests unexpectedly already present in branch source')
text += r'''


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

rm -f \
  .github/workflows/e8-second-audit-fix-once.yml \
  .github/workflows/e8-second-audit-fix-v2-once.yml \
  .github/workflows/e8-second-audit-fix-v3-once.yml \
  scripts/.tmp_e8_second_audit_fix.sh \
  scripts/.tmp_e8_second_audit_fix_v3.sh

git config user.name github-actions[bot]
git config user.email 41898282+github-actions[bot]@users.noreply.github.com
git add -A
git diff --cached --check
git commit -m "fix: close E8 second adversarial audit"
git push origin HEAD:dev/e8-config-driven-sweep-01
