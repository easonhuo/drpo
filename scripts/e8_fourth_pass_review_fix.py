#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path


def replace_once(path: Path, old: str, new: str) -> None:
    text = path.read_text(encoding="utf-8")
    count = text.count(old)
    if count != 1:
        raise SystemExit(
            f"{path}: expected one target, found {count}: {old[:120]!r}"
        )
    path.write_text(text.replace(old, new, 1), encoding="utf-8")


runner = Path("scripts/run_e8_multitask_exp_coldstart.sh")
tests = Path("tests/test_e8_multitask_p0.py")

old = '''ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EXPERIMENT_ID_OVERRIDE="${E8_COLDSTART_EXPERIMENT_ID:-}"
EXPERIMENT_ID=""
CONFIG_PATH="${E8_COLDSTART_CONFIG:-${ROOT_DIR}/configs/e8_multitask_exp_coldstart.yaml}"
P0_CONFIG_PATH="${E8_COLDSTART_P0_CONFIG:-${ROOT_DIR}/configs/e8_multitask_p0.yaml}"
RUN_ID="${E8_COLDSTART_RUN_ID:-E8_MULTITASK_EXP_COLDSTART_20260820_02}"
RUNTIME_ROOT="${E8_COLDSTART_RUNTIME_ROOT:-${ROOT_DIR}/../drpo-e8-coldstart-runtime}"
ATTEMPTS_ROOT="${E8_COLDSTART_GUARD_ROOT:-${RUNTIME_ROOT}/guard/${RUN_ID}}"
GUARD_ROOT="${ATTEMPTS_ROOT}/attempt-001"
OUTPUT_ROOT="${E8_COLDSTART_OUTPUT_ROOT:-${GUARD_ROOT}/workload}"
P0_WORK_DIR="${E8_COLDSTART_P0_WORK_DIR:-${OUTPUT_ROOT}/p0_inputs}"
COUNTDOWN_WORK_DIR="${E8_COLDSTART_COUNTDOWN_WORK_DIR:-${OUTPUT_ROOT}/countdown_inputs}"
VENV_DIR="${E8_COLDSTART_VENV_DIR:-${RUNTIME_ROOT}/venv}"
SELFTEST_VENV_DIR="${E8_COLDSTART_SELFTEST_VENV_DIR:-${RUNTIME_ROOT}/selftest-venv}"
MODEL_DIR="${E8_COLDSTART_MODEL_DIR:-${RUNTIME_ROOT}/models/Qwen2.5-0.5B-Instruct-7ae5576}"
GUARD_ARTIFACT="${E8_COLDSTART_GUARD_ARTIFACT:-${RUNTIME_ROOT}/packages/${RUN_ID}_attempt-001_guarded.zip}"
RECOVERY_ROOT="${E8_COLDSTART_RECOVERY_ROOT:-${RUNTIME_ROOT}/recovery/${RUN_ID}}"
RECOVERY_PACKAGE="${RECOVERY_ROOT}/latest_checkpoint.zip"
DELIVERY_PREFLIGHT_PACKAGE="${RECOVERY_ROOT}/delivery_preflight.zip"
'''
new = '''ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EXPERIMENT_ID_OVERRIDE="${E8_COLDSTART_EXPERIMENT_ID:-}"
EXPERIMENT_ID=""
CONFIG_PATH="${E8_COLDSTART_CONFIG:-${ROOT_DIR}/configs/e8_multitask_exp_coldstart.yaml}"
P0_CONFIG_PATH="${E8_COLDSTART_P0_CONFIG:-${ROOT_DIR}/configs/e8_multitask_p0.yaml}"
RUN_ID_OVERRIDE="${E8_COLDSTART_RUN_ID:-}"
RUN_ID=""
RUNTIME_ROOT="${E8_COLDSTART_RUNTIME_ROOT:-${ROOT_DIR}/../drpo-e8-coldstart-runtime}"
VENV_DIR="${E8_COLDSTART_VENV_DIR:-${RUNTIME_ROOT}/venv}"
SELFTEST_VENV_DIR="${E8_COLDSTART_SELFTEST_VENV_DIR:-${RUNTIME_ROOT}/selftest-venv}"
MODEL_DIR="${E8_COLDSTART_MODEL_DIR:-${RUNTIME_ROOT}/models/Qwen2.5-0.5B-Instruct-7ae5576}"
'''
replace_once(runner, old, new)

marker = '''resolve_experiment_id

# Source provenance follows the selected config, not an experiment-ID branch.
'''
insert = '''resolve_experiment_id

resolve_run_identity() {
  if [[ -n "${RUN_ID_OVERRIDE}" ]]; then
    RUN_ID="${RUN_ID_OVERRIDE}"
  elif [[ "${CONFIG_REPO_PATH}" == "configs/e8_multitask_exp_coldstart.yaml" ]]; then
    RUN_ID="E8_MULTITASK_EXP_COLDSTART_20260820_02"
  else
    fail "set E8_COLDSTART_RUN_ID for non-default config: ${CONFIG_REPO_PATH}"
  fi
  [[ "${RUN_ID}" =~ ^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$ ]] || \
    fail "E8_COLDSTART_RUN_ID must match [A-Za-z0-9][A-Za-z0-9._-]{0,127}"

  ATTEMPTS_ROOT="${E8_COLDSTART_GUARD_ROOT:-${RUNTIME_ROOT}/guard/${RUN_ID}}"
  GUARD_ROOT="${ATTEMPTS_ROOT}/attempt-001"
  OUTPUT_ROOT="${E8_COLDSTART_OUTPUT_ROOT:-${GUARD_ROOT}/workload}"
  P0_WORK_DIR="${E8_COLDSTART_P0_WORK_DIR:-${OUTPUT_ROOT}/p0_inputs}"
  COUNTDOWN_WORK_DIR="${E8_COLDSTART_COUNTDOWN_WORK_DIR:-${OUTPUT_ROOT}/countdown_inputs}"
  GUARD_ARTIFACT="${E8_COLDSTART_GUARD_ARTIFACT:-${RUNTIME_ROOT}/packages/${RUN_ID}_attempt-001_guarded.zip}"
  RECOVERY_ROOT="${E8_COLDSTART_RECOVERY_ROOT:-${RUNTIME_ROOT}/recovery/${RUN_ID}}"
  RECOVERY_PACKAGE="${RECOVERY_ROOT}/latest_checkpoint.zip"
  DELIVERY_PREFLIGHT_PACKAGE="${RECOVERY_ROOT}/delivery_preflight.zip"
  export RUN_ID ATTEMPTS_ROOT GUARD_ROOT OUTPUT_ROOT P0_WORK_DIR COUNTDOWN_WORK_DIR
  export GUARD_ARTIFACT RECOVERY_ROOT RECOVERY_PACKAGE DELIVERY_PREFLIGHT_PACKAGE
}

resolve_run_identity

# Source provenance follows the selected config, not an experiment-ID branch.
'''
replace_once(runner, marker, insert)

text = tests.read_text(encoding="utf-8")
marker = '''def test_runner_and_bootstrap_derive_experiment_id_from_config() -> None:
'''
extra = r'''def test_coldstart_runner_run_identity_is_config_safe() -> None:
    import re

    runner = Path("scripts/run_e8_multitask_exp_coldstart.sh").read_text(
        encoding="utf-8"
    )
    match = re.search(
        r"(resolve_run_identity\(\) \{.*?\n\})\n\nresolve_run_identity",
        runner,
        flags=re.S,
    )
    assert match is not None
    function = match.group(1)

    def execute(*, config: str, override: str) -> subprocess.CompletedProcess[str]:
        script = f'''set -u
fail() {{ echo "ERROR: $*" >&2; exit 2; }}
RUN_ID_OVERRIDE={override!r}
RUN_ID=""
CONFIG_REPO_PATH={config!r}
RUNTIME_ROOT=/tmp/e8-runtime
{function}
resolve_run_identity
printf '%s\n' "$RUN_ID" "$ATTEMPTS_ROOT" "$RECOVERY_ROOT" "$GUARD_ARTIFACT"
'''
        return subprocess.run(
            ["bash", "-c", script],
            check=False,
            capture_output=True,
            text=True,
        )

    base = execute(
        config="configs/e8_multitask_exp_coldstart.yaml",
        override="",
    )
    assert base.returncode == 0, base.stderr
    base_lines = base.stdout.splitlines()
    assert base_lines[0] == "E8_MULTITASK_EXP_COLDSTART_20260820_02"
    assert base_lines[1].endswith("/guard/E8_MULTITASK_EXP_COLDSTART_20260820_02")
    assert base_lines[2].endswith("/recovery/E8_MULTITASK_EXP_COLDSTART_20260820_02")

    generic_missing = execute(
        config="configs/e8_new_generic_sweep.yaml",
        override="",
    )
    assert generic_missing.returncode == 2
    assert "set E8_COLDSTART_RUN_ID for non-default config" in generic_missing.stderr

    generic = execute(
        config="configs/e8_new_generic_sweep.yaml",
        override="E8_GENERIC_SWEEP_01",
    )
    assert generic.returncode == 0, generic.stderr
    generic_lines = generic.stdout.splitlines()
    assert generic_lines[0] == "E8_GENERIC_SWEEP_01"
    assert generic_lines[1].endswith("/guard/E8_GENERIC_SWEEP_01")
    assert generic_lines[2].endswith("/recovery/E8_GENERIC_SWEEP_01")
    assert generic_lines[3].endswith("/E8_GENERIC_SWEEP_01_attempt-001_guarded.zip")

    for unsafe in ("../escape", "nested/run", " bad", "bad id", "-bad"):
        rejected = execute(
            config="configs/e8_new_generic_sweep.yaml",
            override=unsafe,
        )
        assert rejected.returncode == 2
        assert "must match [A-Za-z0-9]" in rejected.stderr

    completion_wrapper = Path(
        "scripts/run_e8_multitask_exp_lambda_completion.sh"
    ).read_text(encoding="utf-8")
    curve_protocol = Path(
        "docs/experiments/E8_MULTITASK_LAMBDA_CURVE_COMPLETION_PROTOCOL.md"
    ).read_text(encoding="utf-8")
    assert "E8_MULTITASK_EXP_LAMBDA_COMPLETION_01" in completion_wrapper
    assert 'E8_COLDSTART_RUN_ID="E8_MULTITASK_EXP_LAMBDA_CURVE_COMPLETION_02"' in curve_protocol


'''
if text.count(marker) != 1 or "test_coldstart_runner_run_identity_is_config_safe" in text:
    raise SystemExit("cannot insert Run ID regression test")
text = text.replace(marker, extra + marker, 1)
tests.write_text(text, encoding="utf-8")
