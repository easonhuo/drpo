#!/usr/bin/env bash
set -euo pipefail

python - <<'PY'
from __future__ import annotations

import re
import subprocess
from pathlib import Path

BASE_SHA = "6ed153d5ad7361a4e52348610b86b51b71e25e47"


def base_text(path: str) -> str:
    return subprocess.check_output(["git", "show", f"{BASE_SHA}:{path}"], text=True)


# Core: preserve current dynamic-cell/liveness fixes, but restore the base
# constant/validator regions and delegate new cold-start config policy.
core_path = Path("src/drpo/e8_multitask_exp_tuning.py")
core = core_path.read_text(encoding="utf-8")
base_core = base_text(core_path.as_posix())

if "from drpo import e8_experiment_config as experiment_config" not in core:
    marker = "import yaml\n\n"
    if marker not in core:
        raise SystemExit("cannot find yaml import marker")
    core = core.replace(
        marker,
        marker + "from drpo import e8_experiment_config as experiment_config\n\n",
        1,
    )

start = 'EXPERIMENT_ID = "EXT-C-E8-MULTITASK-EXP-TUNING-01"'
end = "@dataclass(frozen=True)\nclass Cell:"
b0 = base_core.index(start)
b1 = base_core.index(end)
constants = base_core[b0:b1]
constants = re.sub(r"^SUPPORTED_EXPERIMENT_IDS = .*?\n", "", constants, flags=re.M)
seed_line = "PAPER_SEED_OFFSETS = (4000, 5000)\n"
if seed_line not in constants:
    raise SystemExit("cannot find paper seed marker")
constants = constants.replace(
    seed_line,
    seed_line
    + "COUNTDOWN_LIVENESS_COEFFICIENT = 0.693147181\n"
    + "COUNTDOWN_LIVENESS_SEED_OFFSET = 4000\n",
    1,
)
c0 = core.index(start)
c1 = core.index(end)
core = core[:c0] + constants + core[c1:]

helper_start = "def load_config("
helper_end = "def coefficient_from_rho("
hb0 = base_core.index(helper_start)
hb1 = base_core.index(helper_end)
helpers = base_core[hb0:hb1]

old_load = '''def load_config(path: str | Path = DEFAULT_CONFIG) -> dict[str, Any]:
    value = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError("Configuration root must be a mapping")
    validate_config(value)
    return value
'''
new_load = '''def load_config(path: str | Path = DEFAULT_CONFIG) -> dict[str, Any]:
    source = Path(path)
    repo_root = Path(__file__).resolve().parents[2]
    if not source.is_absolute():
        source = repo_root / source
    value = yaml.safe_load(source.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError("Configuration root must be a mapping")
    validate_config(value)
    experiment_config.validate_historical_config_identity(
        source, value, repo_root=repo_root
    )
    return value
'''
if old_load not in helpers:
    raise SystemExit("base load_config block not found")
helpers = helpers.replace(old_load, new_load, 1)

old_id = '''def experiment_id(config: Mapping[str, Any]) -> str:
    value = str(config.get("experiment_id", ""))
    if value not in SUPPORTED_EXPERIMENT_IDS:
        raise ValueError(f"Unsupported experiment_id: {value}")
    return value
'''
new_id = '''def experiment_id(config: Mapping[str, Any]) -> str:
    return experiment_config.experiment_id(config)
'''
if old_id not in helpers:
    raise SystemExit("base experiment_id block not found")
helpers = helpers.replace(old_id, new_id, 1)

old_lambdas = '''def _task_lambdas(config: Mapping[str, Any], task: str) -> tuple[float, ...]:
    if not _uses_task_lambdas(config):
        raise ValueError("Task-local lambdas are not defined for this profile")
    values = _tuple_floats(config["sweep"]["task_lambda"][task])
    if any(not math.isfinite(value) or value <= 0.0 for value in values):
        raise ValueError(f"{task} lambda values must be finite and positive")
    return values
'''
new_lambdas = '''def _task_lambdas(config: Mapping[str, Any], task: str) -> tuple[float, ...]:
    if not _uses_task_lambdas(config):
        raise ValueError("Task-local lambdas are not defined for this profile")
    return experiment_config.task_lambdas(config, task)
'''
if old_lambdas not in helpers:
    raise SystemExit("base _task_lambdas block not found")
helpers = helpers.replace(old_lambdas, new_lambdas, 1)

profile_gate = '''    if profile not in (SWEEP_PROFILE_RHO, SWEEP_PROFILE_DENSE, SWEEP_PROFILE_COLDSTART):
        raise ValueError(f"Unsupported sweep profile: {profile}")
'''
profile_gate_new = profile_gate + '''    experiment_config.validate_profile_experiment_id(config)
    if profile == SWEEP_PROFILE_COLDSTART:
        return
'''
if profile_gate not in helpers:
    raise SystemExit("base profile gate not found")
helpers = helpers.replace(profile_gate, profile_gate_new, 1)

ch0 = core.index(helper_start)
ch1 = core.index(helper_end)
core = core[:ch0] + helpers + core[ch1:]
core_path.write_text(core, encoding="utf-8")

# Runner: actual tracked-file check + standalone preflight before model/GPU cost.
runner_path = Path("scripts/run_e8_multitask_exp_coldstart.sh")
runner = runner_path.read_text(encoding="utf-8")
tracked_marker = '''  CONFIG_REPO_PATH="${resolved}"
  CONFIG_PATH="${ROOT_DIR}/${CONFIG_REPO_PATH}"
'''
tracked_replacement = '''  CONFIG_REPO_PATH="${resolved}"
  git -C "${ROOT_DIR}" ls-files --error-unmatch -- "${CONFIG_REPO_PATH}" >/dev/null 2>&1 ||
    fail "config is not Git-tracked: ${CONFIG_REPO_PATH}"
  CONFIG_PATH="${ROOT_DIR}/${CONFIG_REPO_PATH}"
'''
if tracked_marker not in runner:
    raise SystemExit("runner tracked marker not found")
runner = runner.replace(tracked_marker, tracked_replacement, 1)

run_module_marker = '''run_module() {
  python -m drpo.e8_multitask_exp_tuning \\
    --config "${CONFIG_PATH}" \\
    --output-root "${OUTPUT_ROOT}" \\
    "$@"
}
'''
preflight_block = run_module_marker + '''
config_preflight() {
  [[ -x "${VENV_DIR}/bin/python" ]] || fail "runtime Python is unavailable for config preflight"
  "${VENV_DIR}/bin/python" "${ROOT_DIR}/scripts/preflight_e8_multitask_config.py" \\
    --repo-root "${ROOT_DIR}" \\
    --config "${CONFIG_PATH}"
}
'''
if run_module_marker not in runner:
    raise SystemExit("runner run_module marker not found")
runner = runner.replace(run_module_marker, preflight_block, 1)

install_marker = '''  python -m pip install --no-deps -e "${ROOT_DIR}"
  python - <<PY
'''
install_replacement = '''  python -m pip install --no-deps -e "${ROOT_DIR}"
  mkdir -p "${RECOVERY_ROOT}"
  config_preflight | tee "${RECOVERY_ROOT}/CONFIG_PREFLIGHT.json"
  python - <<PY
'''
if install_marker not in runner:
    raise SystemExit("runner setup install marker not found")
runner = runner.replace(install_marker, install_replacement, 1)

ensure_old = '''ensure_setup() {
  if runtime_ready; then
    echo "Reusing verified runtime and pinned model at ${RUNTIME_ROOT}"
  else
    setup
  fi
}
'''
ensure_new = '''ensure_setup() {
  if [[ -x "${VENV_DIR}/bin/python" ]] &&
     "${VENV_DIR}/bin/python" -c 'import numpy, yaml' >/dev/null 2>&1; then
    config_preflight
  fi
  if runtime_ready; then
    echo "Reusing verified runtime and pinned model at ${RUNTIME_ROOT}"
  else
    setup
  fi
}
'''
if ensure_old not in runner:
    raise SystemExit("runner ensure_setup block not found")
runner = runner.replace(ensure_old, ensure_new, 1)

slash = chr(92)
guard_marker = f"    --source-file src/drpo/e8_multitask_exp_tuning.py {slash}\n"
if guard_marker not in runner:
    raise SystemExit("runner guard source marker not found")
runner = runner.replace(
    guard_marker,
    guard_marker
    + f"    --source-file src/drpo/e8_experiment_config.py {slash}\n"
    + f"    --source-file scripts/preflight_e8_multitask_config.py {slash}\n",
    1,
)

delivery_marker = "    --source-file src/drpo/e8_multitask_exp_tuning.py\n"
if delivery_marker not in runner:
    raise SystemExit("runner delivery source marker not found")
runner = runner.replace(
    delivery_marker,
    delivery_marker
    + "    --source-file src/drpo/e8_experiment_config.py\n"
    + "    --source-file scripts/preflight_e8_multitask_config.py\n",
    1,
)

case_marker = "  plan) check_source; activate_runtime; run_module plan ;;\n"
if case_marker not in runner:
    raise SystemExit("runner case plan marker not found")
runner = runner.replace(
    case_marker,
    "  preflight) check_source; activate_runtime; config_preflight ;;\n" + case_marker,
    1,
)
runner = runner.replace(
    "{self-test|setup|prepare|plan|calibrate|liveness|run|resume|finish|full}",
    "{self-test|setup|preflight|prepare|plan|calibrate|liveness|run|resume|finish|full}",
    1,
)
runner_path.write_text(runner, encoding="utf-8")

# Tests: restore main's suite, patch expectations for the already-approved shell
# identity changes, then add only architecture-level regressions.
test_path = Path("tests/test_e8_multitask_p0.py")
tests = base_text(test_path.as_posix())
tests = tests.replace(
    "from __future__ import annotations\n\n",
    "from __future__ import annotations\n\nimport copy\n",
    1,
)

runbook_pattern = re.compile(
    r'''    historical_bootstrap = subprocess\.check_output\(\n.*?    assert normalized == historical_bootstrap\n''',
    flags=re.S,
)
tests, count = runbook_pattern.subn(
    '''    assert 'EXPERIMENT_ID_OVERRIDE="${E8_COLDSTART_EXPERIMENT_ID:-}"' in bootstrap
    assert 'CONFIG_EXPERIMENT_ID="$(read_config_experiment_id "${CONFIG_PATH}")"' in bootstrap
    assert "refs/pull/309/head" not in bootstrap
''',
    tests,
    count=1,
)
if count != 1:
    raise SystemExit("runbook test replacement failed")

tests = tests.replace(
    '''    assert 'EXPERIMENT_ID="${E8_COLDSTART_EXPERIMENT_ID:-EXT-C-E8-MULTITASK-EXP-COLDSTART-01}"' in historical_launcher
''',
    '''    assert 'EXPERIMENT_ID_OVERRIDE="${E8_COLDSTART_EXPERIMENT_ID:-}"' in historical_launcher
    assert 'EXPERIMENT_ID="${config_experiment_id}"' in historical_launcher
''',
    1,
)
tests = tests.replace(
    '''    assert "SUCCESSOR_SOURCE_ARGS" in historical_launcher
    assert "scripts/run_e8_multitask_exp_lambda_completion.sh" in historical_launcher
    assert "docs/experiments/E8_MULTITASK_LAMBDA_COMPLETION_PROTOCOL.md" in historical_launcher
''',
    '''    assert "SUCCESSOR_SOURCE_ARGS" not in historical_launcher
    assert "CONFIG_SOURCE_ARGS" in historical_launcher
''',
    1,
)
tests = tests.replace(
    '    config["sweep"]["parameterization"] = "paper_coefficient_c"\n',
    '    config["sweep"]["parameterization"] = "unsupported_parameterization"\n',
    1,
)

tests += r'''


def test_new_coldstart_config_controls_scientific_scalars_without_core_edits() -> None:
    from drpo import e8_multitask_exp_tuning as exp_tuning

    config = exp_tuning.load_config(
        Path("configs/e8_multitask_exp_lambda_curve_completion.yaml")
    )
    config["experiment_id"] = "EXT-C-E8-MULTITASK-EXP-CONFIG-AUTHORITY-TEST"
    config["training"]["optimizer_updates"] = 1500
    config["evaluation"].update(
        {"sampling_temperature": 0.7, "top_p": 0.9, "generation_seed": 2026090101}
    )
    config["split"]["hash_seed"] = 2026090102
    config["initialization"]["seed"] = 2026090103
    config["model"]["lora_rank"] = 16
    config["task_runtime"]["word_sorting"]["max_length"] = 640

    exp_tuning.validate_config(config)
    cells = exp_tuning.build_cells(config)
    assert len(cells) == 140
    assert config["training"]["optimizer_updates"] == 1500
    assert config["evaluation"]["sampling_temperature"] == 0.7
    assert config["model"]["lora_rank"] == 16


def test_profile_experiment_id_scope_is_fail_closed() -> None:
    from drpo import e8_multitask_exp_tuning as exp_tuning

    rho = exp_tuning.load_config(Path("configs/e8_multitask_exp_tuning.yaml"))
    rho["experiment_id"] = "GENERIC-RHO-ID"
    with pytest.raises(ValueError, match="requires experiment_id"):
        exp_tuning.validate_config(rho)

    dense = exp_tuning.load_config(Path("configs/e8_multitask_exp_lambda_dense.yaml"))
    dense["experiment_id"] = "GENERIC-DENSE-ID"
    with pytest.raises(ValueError, match="requires experiment_id"):
        exp_tuning.validate_config(dense)


def test_historical_coldstart_id_requires_canonical_config_identity() -> None:
    from drpo import e8_multitask_exp_tuning as exp_tuning

    source = Path("configs/e8_multitask_exp_coldstart.yaml")
    copied = Path("configs/.e8_historical_identity_test.yaml")
    copied.write_text(source.read_text(encoding="utf-8"), encoding="utf-8")
    try:
        with pytest.raises(ValueError, match="canonical config path"):
            exp_tuning.load_config(copied)
    finally:
        copied.unlink(missing_ok=True)


def test_generic_coldstart_matrix_is_dynamic_but_self_consistent() -> None:
    from drpo import e8_multitask_exp_tuning as exp_tuning

    config = exp_tuning.load_config(
        Path("configs/e8_multitask_exp_lambda_curve_completion.yaml")
    )
    config["experiment_id"] = "EXT-C-E8-MULTITASK-EXP-DYNAMIC-MATRIX-TEST"
    sweep = config["sweep"]
    sweep["include_global_endpoint"] = True
    sweep["transfer_positive_only_seed_offsets"] = [8000, 9000]
    config["reporting"]["positive_only_seed_count_per_transfer_task"] = 2
    sweep["task_lambda"]["maze"] = []
    sweep["task_grid_hashes"]["maze"] = exp_tuning.stable_hash([])
    active = [
        task for task in config["suite"]["p0_tasks"] if sweep["task_lambda"][task]
    ]
    sweep["expected_cells"] = sum(
        2 + 1 + len(sweep["task_lambda"][task]) for task in active
    )

    exp_tuning.validate_config(config)
    cells = exp_tuning.build_cells(config)
    assert len(cells) == sweep["expected_cells"]
    assert not any(cell.task == "maze" for cell in cells)
    for task in active:
        global_cells = [
            cell
            for cell in cells
            if cell.task == task and cell.method == exp_tuning.METHOD_GLOBAL
        ]
        assert len(global_cells) == 1
        assert global_cells[0].lambda_value == 0.0

    bad = copy.deepcopy(config)
    bad["sweep"]["expected_cells"] += 1
    with pytest.raises(ValueError, match="expected_cells"):
        exp_tuning.validate_config(bad)


def test_generic_coldstart_rejects_malformed_config_not_old_scientific_values() -> None:
    from drpo import e8_multitask_exp_tuning as exp_tuning

    config = exp_tuning.load_config(
        Path("configs/e8_multitask_exp_lambda_curve_completion.yaml")
    )
    config["experiment_id"] = "EXT-C-E8-MULTITASK-EXP-TYPE-CHECK-TEST"

    bad = copy.deepcopy(config)
    bad["sweep"]["task_transfer_seed_offset"] = 4000.5
    with pytest.raises(ValueError, match="integer"):
        exp_tuning.validate_config(bad)

    bad = copy.deepcopy(config)
    bad["sweep"]["task_lambda"]["word_sorting"][0] = True
    with pytest.raises(ValueError, match="finite numeric scalar"):
        exp_tuning.validate_config(bad)

    bad = copy.deepcopy(config)
    bad["sweep"]["method"] = "quadratic"
    with pytest.raises(ValueError, match="sweep.method"):
        exp_tuning.validate_config(bad)


def test_generic_coldstart_rejects_zero_scientific_cells() -> None:
    from drpo import e8_multitask_exp_tuning as exp_tuning

    config = exp_tuning.load_config(
        Path("configs/e8_multitask_exp_lambda_curve_completion.yaml")
    )
    config["experiment_id"] = "EXT-C-E8-MULTITASK-EXP-ZERO-CELLS-TEST"
    sweep = config["sweep"]
    sweep["countdown_seed_offsets"] = []
    sweep["transfer_positive_only_seed_offsets"] = []
    for task in config["suite"]["p0_tasks"]:
        sweep["task_lambda"][task] = []
        sweep["task_grid_hashes"][task] = exp_tuning.stable_hash([])
    sweep["expected_cells"] = 0
    config["reporting"]["positive_only_seed_count_per_transfer_task"] = 0
    with pytest.raises(ValueError, match="at least one scientific cell"):
        exp_tuning.validate_config(config)


def test_canonical_liveness_identity_is_grid_independent_and_stale_safe() -> None:
    from drpo import e8_multitask_exp_tuning as exp_tuning

    cell = exp_tuning._canonical_cold_liveness_cell()
    assert cell.task == "countdown"
    assert cell.method == exp_tuning.METHOD_EXPONENTIAL
    assert cell.lambda_value == 0.693147181
    assert cell.seed == 4000

    manifest = {
        "canonical_dispatch": "countdown_e8_alpha1_highc_scan_runtime.smoke",
        "cell": {
            "task": cell.task,
            "method": cell.method,
            "rho": cell.rho,
            "lambda": cell.lambda_value,
            "seed": cell.seed,
            "stage": cell.stage,
        },
    }
    assert exp_tuning._matches_canonical_cold_liveness_manifest(manifest)
    stale = copy.deepcopy(manifest)
    stale["cell"]["seed"] = 5000
    assert not exp_tuning._matches_canonical_cold_liveness_manifest(stale)


def test_e8_config_preflight_is_tracked_and_non_scientific() -> None:
    from drpo import e8_experiment_config as experiment_config
    from scripts.preflight_e8_multitask_config import build_summary

    repo = Path.cwd()
    summary = build_summary(Path("configs/e8_multitask_exp_coldstart.yaml"), repo)
    assert summary["experiment_id"] == "EXT-C-E8-MULTITASK-EXP-COLDSTART-01"
    assert summary["cell_count"] == 208
    assert summary["wave_sizes"] == [16] * 13
    assert summary["scientific_status"] == "not_run"

    untracked = Path("configs/.e8_untracked_preflight_test.yaml")
    untracked.write_text("schema_version: 1\n", encoding="utf-8")
    try:
        with pytest.raises(ValueError, match="not Git-tracked"):
            experiment_config.require_tracked_config(untracked, repo)
    finally:
        untracked.unlink(missing_ok=True)


def test_runner_uses_real_tracked_config_check_and_standalone_preflight() -> None:
    runner = Path("scripts/run_e8_multitask_exp_coldstart.sh").read_text(
        encoding="utf-8"
    )
    assert (
        'git -C "${ROOT_DIR}" ls-files --error-unmatch -- "${CONFIG_REPO_PATH}"'
        in runner
    )
    assert "scripts/preflight_e8_multitask_config.py" in runner
    assert "config_preflight | tee" in runner
    assert "E8_COLDSTART_RUN_ID must match [A-Za-z0-9]" in runner

    bootstrap = Path("scripts/bootstrap_e8_multitask_exp_coldstart.sh").read_text(
        encoding="utf-8"
    )
    assert "refs/pull/309/head" not in bootstrap
    assert "self-test requires E8_COLDSTART_TARGET_REF" in bootstrap
'''

test_path.write_text(tests, encoding="utf-8")
PY
