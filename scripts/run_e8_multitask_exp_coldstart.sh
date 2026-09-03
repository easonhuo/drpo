#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
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
EXPECTED_COMMIT="${E8_COLDSTART_EXPECTED_COMMIT:-}"
RUN_CLASS="${E8_COLDSTART_RUN_CLASS:-formal}"
REQUIRE_ORIGIN_MAIN="${E8_COLDSTART_REQUIRE_ORIGIN_MAIN:-1}"
MODE="${1:-full}"

export PYTHONPATH="${ROOT_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}"
export TOKENIZERS_PARALLELISM=false
export PYTHONDONTWRITEBYTECODE=1

fail() {
  echo "ERROR: $*" >&2
  exit 2
}

resolve_config_repo_path() {
  command -v python3 >/dev/null || fail "python3 is unavailable"
  local resolved
  resolved="$(python3 - "${ROOT_DIR}" "${CONFIG_PATH}" <<'PY_CONFIG'
from pathlib import Path
import sys
root = Path(sys.argv[1]).resolve()
candidate = Path(sys.argv[2])
if not candidate.is_absolute():
    candidate = root / candidate
path = candidate.resolve()
try:
    relative = path.relative_to(root)
except ValueError as exc:
    raise SystemExit(f"config path escapes repository: {path}") from exc
if not path.is_file():
    raise SystemExit(f"config file is missing: {path}")
print(relative.as_posix())
PY_CONFIG
)" || fail "config path must resolve to a file inside the repository: ${CONFIG_PATH}"
  CONFIG_REPO_PATH="${resolved}"
  CONFIG_PATH="${ROOT_DIR}/${CONFIG_REPO_PATH}"
  export CONFIG_REPO_PATH CONFIG_PATH
}

resolve_config_repo_path

read_config_experiment_id() {
  command -v python3 >/dev/null || return 127
  python3 - "$1" <<'PY_EXPERIMENT_ID'
from pathlib import Path
import re
import sys

safe = r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}"
plain = re.compile(rf"^({safe})(?:[ \t]+#.*)?$")
single = re.compile(rf"^'({safe})'(?:[ \t]+#.*)?$")
double = re.compile(rf'^"({safe})"(?:[ \t]+#.*)?$')
values = []
malformed = False
for raw in Path(sys.argv[1]).read_text(encoding="utf-8").splitlines():
    if raw[:1].isspace() or not raw.startswith("experiment_id:"):
        continue
    rhs = raw.split(":", 1)[1].strip()
    match = plain.fullmatch(rhs) or single.fullmatch(rhs) or double.fullmatch(rhs)
    if match is None:
        malformed = True
    else:
        values.append(match.group(1))
if malformed or len(values) != 1:
    raise SystemExit(2)
print(values[0])
PY_EXPERIMENT_ID
}

resolve_experiment_id() {
  local config_experiment_id
  config_experiment_id="$(read_config_experiment_id "${CONFIG_PATH}")" || \
    fail "config must contain exactly one well-formed top-level experiment_id: ${CONFIG_REPO_PATH}"
  if [[ -n "${EXPERIMENT_ID_OVERRIDE}" && "${EXPERIMENT_ID_OVERRIDE}" != "${config_experiment_id}" ]]; then
    fail "experiment_id mismatch: env=${EXPERIMENT_ID_OVERRIDE} config=${config_experiment_id}"
  fi
  EXPERIMENT_ID="${config_experiment_id}"
  export E8_COLDSTART_EXPERIMENT_ID="${EXPERIMENT_ID}"
}

resolve_experiment_id

resolve_run_identity() {
  if [[ -n "${RUN_ID_OVERRIDE}" ]]; then
    RUN_ID="${RUN_ID_OVERRIDE}"
  elif [[ "${CONFIG_REPO_PATH}" == "configs/e8_multitask_exp_coldstart.yaml" ]]; then
    RUN_ID="E8_MULTITASK_EXP_COLDSTART_20260820_02"
  else
    RUN_ID="${EXPERIMENT_ID}"
  fi
  [[ "${RUN_ID}" =~ ^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$ ]] ||     fail "E8_COLDSTART_RUN_ID must match [A-Za-z0-9][A-Za-z0-9._-]{0,127}"

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
CONFIG_SOURCE_ARGS=()
while IFS= read -r source_rel; do
  [[ -n "${source_rel}" ]] || continue
  [[ "${source_rel}" != "scripts/run_e8_multitask_exp_coldstart.sh" ]] || continue
  source_path="${ROOT_DIR}/${source_rel}"
  if grep -Fq -- "${CONFIG_REPO_PATH}" "${source_path}"; then
    CONFIG_SOURCE_ARGS+=(--source-file "${source_rel}")
  fi
done < <(
  git -C "${ROOT_DIR}" ls-files \
    'docs/experiments/*.md' \
    'scripts/run_e8_multitask_exp_*.sh' | sort
)

case "${RUN_CLASS}" in
  formal|pilot) ;;
  *) fail "E8_COLDSTART_RUN_CLASS must be formal or pilot" ;;
esac
case "${REQUIRE_ORIGIN_MAIN}" in
  0|1) ;;
  *) fail "E8_COLDSTART_REQUIRE_ORIGIN_MAIN must be 0 or 1" ;;
esac
if [[ "${RUN_CLASS}" == "formal" && "${REQUIRE_ORIGIN_MAIN}" != "1" ]]; then
  fail "formal cold-start execution must retain origin/main matching"
fi

is_expected_origin() {
  local url="$1"
  [[ "${url}" =~ ^https://([^/@]+(:[^/@]*)?@)?github\.com/easonhuo/drpo(\.git)?/?$ ]] ||
    [[ "${url}" =~ ^git@github\.com:easonhuo/drpo(\.git)?$ ]] ||
    [[ "${url}" =~ ^ssh://git@github\.com/easonhuo/drpo(\.git)?/?$ ]]
}

check_source() {
  [[ -n "${EXPECTED_COMMIT}" ]] || fail "set E8_COLDSTART_EXPECTED_COMMIT to the reviewed implementation commit"
  local current_commit
  current_commit="$(git -C "${ROOT_DIR}" rev-parse HEAD)"
  [[ "${current_commit}" == "${EXPECTED_COMMIT}" ]] || \
    fail "source commit mismatch: expected ${EXPECTED_COMMIT}, found ${current_commit}"
  [[ "${EXPECTED_COMMIT}" =~ ^[0-9a-f]{40}$ ]] || fail "expected commit must be a full lowercase SHA"
  local origin_url
  origin_url="$(git -C "${ROOT_DIR}" remote get-url origin)"
  is_expected_origin "${origin_url}" || fail "origin is not the canonical easonhuo/drpo repository"
  [[ -z "$(git -C "${ROOT_DIR}" status --porcelain=v1 --untracked-files=all)" ]] || \
    fail "source checkout must be fully clean; keep runtime files outside the repository"
  git -C "${ROOT_DIR}" cat-file -e "${EXPECTED_COMMIT}:${CONFIG_REPO_PATH}" 2>/dev/null || \
    fail "runtime config is unavailable at launch commit ${EXPECTED_COMMIT}: ${CONFIG_REPO_PATH}"
}

check_authoritative_main_at_invocation() {
  [[ "${RUN_CLASS}" == "formal" && "${REQUIRE_ORIGIN_MAIN}" == "1" ]] || return 0
  local remote_commit
  remote_commit="$(
    git -C "${ROOT_DIR}" ls-remote origin refs/heads/main |
      awk '$2 == "refs/heads/main" {print $1}'
  )"
  [[ "${remote_commit}" =~ ^[0-9a-f]{40}$ ]] || \
    fail "could not resolve authoritative origin/main for formal invocation"
  [[ "${remote_commit}" == "${EXPECTED_COMMIT}" ]] || \
    fail "formal invocation is stale: expected ${EXPECTED_COMMIT}, origin/main is ${remote_commit}"
}

runtime_setup_lock() {
  command -v flock >/dev/null || fail "flock is required for shared runtime setup safety"
  mkdir -p "${RUNTIME_ROOT}"
  exec 7>"${RUNTIME_ROOT}/setup.lock"
  flock 7
}

runtime_setup_unlock() {
  flock -u 7
  exec 7>&-
}

activate_runtime() {
  [[ -x "${VENV_DIR}/bin/python" ]] || fail "runtime is absent; run '$0 setup' first"
  # shellcheck disable=SC1091
  source "${VENV_DIR}/bin/activate"
}

preflight_gpu() {
  command -v nvidia-smi >/dev/null || fail "nvidia-smi is unavailable"
  local gpu_count
  gpu_count="$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)"
  [[ "${gpu_count}" -ge 8 ]] || fail "eight visible GPUs are required; found ${gpu_count}"
  local small_gpu
  small_gpu="$(nvidia-smi --query-gpu=index,memory.total --format=csv,noheader,nounits | awk -F, '$2+0 < 12000 {print $1":"$2}')"
  [[ -z "${small_gpu}" ]] || fail "each GPU needs at least 12 GiB; undersized: ${small_gpu}"
  python - <<'PY'
import torch
assert torch.cuda.is_available(), "PyTorch cannot see CUDA"
assert torch.cuda.device_count() >= 8, f"PyTorch sees {torch.cuda.device_count()} GPUs"
print({"torch": torch.__version__, "cuda": torch.version.cuda, "gpus": torch.cuda.device_count()})
PY
  local free_kb
  free_kb="$(df -Pk "${RUNTIME_ROOT}" | awk 'NR==2 {print $4}')"
  [[ "${free_kb}" -ge 83886080 ]] || fail "at least 80 GiB free disk is required"
}

run_module() {
  python -m drpo.e8_multitask_exp_tuning \
    --config "${CONFIG_PATH}" \
    --output-root "${OUTPUT_ROOT}" \
    "$@"
}

config_preflight() {
  [[ -x "${VENV_DIR}/bin/python" ]] || fail "runtime Python is unavailable for config preflight"
  "${VENV_DIR}/bin/python" "${ROOT_DIR}/scripts/preflight_e8_multitask_config.py" \
    --repo-root "${ROOT_DIR}" \
    --config "${CONFIG_PATH}"
}

bootstrap_config_preflight() {
  command -v python3 >/dev/null || fail "python3 is unavailable"
  if [[ ! -x "${VENV_DIR}/bin/python" ]]; then
    python3 -m venv --system-site-packages "${VENV_DIR}"
  fi
  # shellcheck disable=SC1091
  source "${VENV_DIR}/bin/activate"
  if ! python - <<'PY_PREFLIGHT_DEPS'
import numpy
import yaml
PY_PREFLIGHT_DEPS
  then
    python -m pip install --disable-pip-version-check "numpy==1.26.4" "PyYAML==6.0.2"
  fi
  mkdir -p "${RECOVERY_ROOT}"
  config_preflight | tee "${RECOVERY_ROOT}/CONFIG_PREFLIGHT.json"
}

setup() {
  check_source
  command -v python3 >/dev/null || fail "python3 is unavailable"
  mkdir -p "${RUNTIME_ROOT}"
  bootstrap_config_preflight
  python3 - <<'PY'
import sys
assert sys.version_info >= (3, 10), sys.version
try:
    import torch
except ImportError as exc:
    raise SystemExit("Install a CUDA-compatible PyTorch build before running setup") from exc
assert torch.cuda.is_available(), "The system PyTorch build cannot see CUDA"
PY
  # shellcheck disable=SC1091
  source "${VENV_DIR}/bin/activate"
  python -m pip install --upgrade "pip==24.3.1" "setuptools==75.6.0" "wheel==0.45.1"
  python -m pip install -r "${ROOT_DIR}/requirements/e8_multitask_exp_coldstart.txt"
  python -m pip install --no-deps -e "${ROOT_DIR}"
  python - <<PY
import json
from pathlib import Path
from huggingface_hub import snapshot_download
preflight = json.loads((Path("${RECOVERY_ROOT}") / "CONFIG_PREFLIGHT.json").read_text())
model = preflight["model"]
snapshot_download(
    repo_id=model["base_model"],
    revision=model["revision"],
    local_dir="${MODEL_DIR}",
)
PY
  preflight_gpu
  python -m pytest -q "${ROOT_DIR}/tests/test_e8_multitask_p0.py"
  # This cold-start family uses the paper Linear/extension path; Reciprocal and AsymRE are separate experiments.
  python -m pytest -q \
    "${ROOT_DIR}/tests/test_countdown_e8_oracle_offline_v2_alpha1_highc_scan.py" \
    -k "not reciprocal and not asymre"
  python - <<PY
from pathlib import Path
from drpo.e8_multitask_exp_tuning import (
    audit_canonical_coldstart_sources,
    load_config,
)
config = load_config(Path("${CONFIG_PATH}"))
audit = audit_canonical_coldstart_sources(config)
assert audit["verified"], audit
print(audit)
PY
  mkdir -p "${RECOVERY_ROOT}"
  python - <<PY
import json
from pathlib import Path
preflight = json.loads((Path("${RECOVERY_ROOT}") / "CONFIG_PREFLIGHT.json").read_text())
path = Path("${RECOVERY_ROOT}") / "SETUP_COMPLETE.json"
path.write_text(json.dumps({
    "schema_version": 1,
    "experiment_id": "${EXPERIMENT_ID}",
    "source_commit": "${EXPECTED_COMMIT}",
    "model_revision": preflight["model"]["revision"],
    "venv": str(Path("${VENV_DIR}").resolve()),
    "model": str(Path("${MODEL_DIR}").resolve()),
    "complete": True,
}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY
}

runtime_ready() {
  [[ -x "${VENV_DIR}/bin/python" ]] || return 1
  [[ -f "${MODEL_DIR}/config.json" ]] || return 1
  [[ -f "${RECOVERY_ROOT}/SETUP_COMPLETE.json" ]] || return 1
  # shellcheck disable=SC1091
  source "${VENV_DIR}/bin/activate"
  python - <<PY >/dev/null
import json
from pathlib import Path
import numpy
import torch
import transformers
import yaml
from drpo.e8_multitask_exp_tuning import load_config
value = json.loads((Path("${RECOVERY_ROOT}") / "SETUP_COMPLETE.json").read_text())
config = load_config(Path("${CONFIG_PATH}"))
assert value["experiment_id"] == "${EXPERIMENT_ID}"
assert value["source_commit"] == "${EXPECTED_COMMIT}"
assert value["model_revision"] == config["model"]["revision"]
assert torch.cuda.is_available()
assert torch.cuda.device_count() >= 8
PY
  preflight_gpu
}

ensure_setup() {
  runtime_setup_lock
  if [[ -x "${VENV_DIR}/bin/python" ]] &&
     "${VENV_DIR}/bin/python" -c 'import numpy, yaml' >/dev/null 2>&1; then
    config_preflight
  fi
  if runtime_ready; then
    echo "Reusing verified runtime and pinned model at ${RUNTIME_ROOT}"
  else
    setup
  fi
  runtime_setup_unlock
}

self_test_setup() {
  check_source
  runtime_setup_lock
  command -v python3 >/dev/null || fail "python3 is unavailable"
  mkdir -p "${RUNTIME_ROOT}"
  python3 - <<'PY'
import sys
assert sys.version_info >= (3, 10), sys.version
PY
  python3 -m venv --system-site-packages "${SELFTEST_VENV_DIR}"
  # shellcheck disable=SC1091
  source "${SELFTEST_VENV_DIR}/bin/activate"
  if ! python - <<'PY'
import numpy
import yaml
PY
  then
    python -m pip install --upgrade "pip==24.3.1" "setuptools==75.6.0" "wheel==0.45.1"
    python -m pip install "numpy==1.26.4" "PyYAML==6.0.2"
  fi
  runtime_setup_unlock
}

attempt_number() {
  local attempt_dir="$1"
  basename "${attempt_dir}" | sed -E 's/^attempt-0*([0-9]+)$/\1/'
}

latest_attempt_root() {
  find "${ATTEMPTS_ROOT}" -mindepth 1 -maxdepth 1 -type d -name 'attempt-[0-9][0-9][0-9]' \
    -print 2>/dev/null | sort | tail -n 1
}

latest_recoverable_output() {
  local attempt
  while IFS= read -r attempt; do
    [[ -f "${attempt}/workload/prepare_manifest.json" ]] || continue
    [[ -f "${attempt}/workload/split_manifest.json" ]] || continue
    [[ -f "${attempt}/workload/source_provenance.json" ]] || continue
    printf '%s\n' "${attempt}/workload"
    return 0
  done < <(
    find "${ATTEMPTS_ROOT}" -mindepth 1 -maxdepth 1 -type d \
      -name 'attempt-[0-9][0-9][0-9]' -print 2>/dev/null | sort -r
  )
  return 1
}

reuse_successful_attempt() {
  local latest
  latest="$(latest_attempt_root)"
  [[ -n "${latest}" && -f "${latest}/RUN_RAW_COMPLETE.json" ]] || return 1
  local number
  number="$(attempt_number "${latest}")"
  local artifact="${RUNTIME_ROOT}/packages/${RUN_ID}_attempt-$(printf '%03d' "${number}")_guarded.zip"
  [[ -f "${artifact}" && -f "${latest}/workload/aggregate/plot_curve_points.csv" ]] || return 1
  python - "${latest}/workload" "${CONFIG_PATH}" "${EXPECTED_COMMIT}" "${MODE}" "${artifact}" <<'PY_REUSE' >/dev/null || return 1
from pathlib import Path
import sys

from drpo.e8_multitask_exp_tuning import (
    _engineering_self_test_config,
    _successful_attempt_matches_current_identity,
    load_config,
)

workload_root = Path(sys.argv[1])
config = load_config(Path(sys.argv[2]))
mode = sys.argv[4]
if mode == "self-test":
    config = _engineering_self_test_config(config)
elif mode != "full":
    raise SystemExit(1)
raise SystemExit(
    0
    if _successful_attempt_matches_current_identity(
        config,
        workload_root,
        source_commit=sys.argv[3],
        artifact_path=Path(sys.argv[5]),
    )
    else 1
)
PY_REUSE
  python "${ROOT_DIR}/scripts/verify_experiment_package_hardened.py" \
    --repo-root "${ROOT_DIR}" "${artifact}" >/dev/null || return 1
  GUARD_ROOT="${latest}"
  OUTPUT_ROOT="${latest}/workload"
  GUARD_ARTIFACT="${artifact}"
  RECOVERY_SOURCE_OUTPUT=""
  export GUARD_ROOT OUTPUT_ROOT GUARD_ARTIFACT RECOVERY_SOURCE_OUTPUT
  return 0
}

select_next_attempt() {
  mkdir -p "${ATTEMPTS_ROOT}" "${RUNTIME_ROOT}/packages" "${RECOVERY_ROOT}"
  local latest
  local next=1
  latest="$(latest_attempt_root)"
  if [[ -n "${latest}" ]]; then
    next=$(( $(attempt_number "${latest}") + 1 ))
  fi
  local maximum="${E8_COLDSTART_MAX_TOTAL_ATTEMPTS:-20}"
  [[ "${maximum}" =~ ^[1-9][0-9]*$ ]] || fail "maximum total attempts must be positive"
  [[ "${next}" -le "${maximum}" ]] || \
    fail "recovery attempt limit reached (${maximum}); local AI review is required"
  GUARD_ROOT="${ATTEMPTS_ROOT}/attempt-$(printf '%03d' "${next}")"
  OUTPUT_ROOT="${GUARD_ROOT}/workload"
  GUARD_ARTIFACT="${RUNTIME_ROOT}/packages/${RUN_ID}_attempt-$(printf '%03d' "${next}")_guarded.zip"
  RECOVERY_SOURCE_OUTPUT=""
  RECOVERY_SOURCE_OUTPUT="$(latest_recoverable_output || true)"
  export GUARD_ROOT OUTPUT_ROOT GUARD_ARTIFACT RECOVERY_SOURCE_OUTPUT
  export E8_COLDSTART_OUTPUT_ROOT="${OUTPUT_ROOT}"
  export E8_COLDSTART_RECOVERY_ROOT="${RECOVERY_ROOT}"
  export E8_COLDSTART_EXPECTED_COMMIT="${EXPECTED_COMMIT}"
}

write_attempt_state() {
  local status="$1"
  mkdir -p "${RECOVERY_ROOT}"
  {
    printf 'experiment_id=%q\n' "${EXPERIMENT_ID}"
    printf 'status=%q\n' "${status}"
    printf 'source_commit=%q\n' "${EXPECTED_COMMIT}"
    printf 'guard_root=%q\n' "${GUARD_ROOT}"
    printf 'output_root=%q\n' "${OUTPUT_ROOT}"
    printf 'artifact=%q\n' "${GUARD_ARTIFACT}"
    printf 'recovery_source_output=%q\n' "${RECOVERY_SOURCE_OUTPUT:-}"
    printf 'recovery_checkpoint=%q\n' "${RECOVERY_PACKAGE}"
  } >"${RECOVERY_ROOT}/ATTEMPT_STATE.env"
}

write_local_ai_recovery() {
  local failure_stage="$1"
  local prompt="${RECOVERY_ROOT}/LOCAL_AI_RECOVERY.md"
  local bundle="${RECOVERY_ROOT}/LOCAL_AI_RECOVERY_BUNDLE.zip"
  mkdir -p "${RECOVERY_ROOT}"
  python3 - "${prompt}" "${bundle}" "${failure_stage}" "${ROOT_DIR}" \
    "${RUNTIME_ROOT}" "${EXPECTED_COMMIT}" "${EXPERIMENT_ID}" <<'PY'
import sys
import zipfile
from pathlib import Path

prompt = Path(sys.argv[1])
bundle = Path(sys.argv[2])
stage = sys.argv[3]
repo = Path(sys.argv[4]).resolve()
runtime = Path(sys.argv[5]).resolve()
commit = sys.argv[6]
experiment_id = sys.argv[7]
text = f"""# Local AI recovery handoff

Experiment: `{experiment_id}`

Source commit: `{commit}`

Failed stage: `{stage}`

Repository: `{repo}`

Runtime: `{runtime}`

## Required local-AI procedure

1. Read `ATTEMPT_STATE.env`, every available `RECOVERY_PLAN.json`, the latest
   `scheduler/dynamic_run.json`, `recovery_package_status.json`, and the supervised log tail.
2. Confirm no original guard/scheduler/cell process still holds the runtime lock. Never start a
   second queue beside a live queue.
3. Do not change tasks, lambdas, seeds, optimizer/training formulas, thresholds, adapters, or the
   frozen scientific configuration. Do not edit the locked canonical cold-start kernels.
4. First rerun the same reviewed one-click bootstrap. It automatically creates a fresh guard
   attempt, hard-links only identity-checked completed cells, and resumes from the first incomplete
   stage.
5. If automatic recovery fails again, classify the first failure as source, runtime, storage, cell,
   aggregation/audit, or packaging. Repair only the causal engineering fault. Preserve every prior
   attempt directory and artifact.
6. A partially trained cell has no scientifically exact intra-cell optimizer/RNG resume. Rerun only
   that cell; never manufacture a completion manifest.
7. If the authorized source ref advanced, identity/hash validation fails, disk is damaged, the
   hard-link filesystem boundary is crossed, or the exact fix would alter a frozen scientific
   variable, stop and report the blocker to the reviewer instead of bypassing a gate.

Primary recovery checkpoint: `{runtime / 'recovery'}`
"""
prompt.write_text(text, encoding="utf-8")

patterns = (
    "**/ATTEMPT_STATE.env",
    "**/BOOTSTRAP_STATE.env",
    "**/RECOVERY_PLAN.json",
    "**/IMPORT_MANIFEST.json",
    "**/RECOVERY_CHECKPOINT_STATUS.json",
    "**/dynamic_run.json",
    "**/recovery_package_status.json",
    "**/supervised_run.log",
    "**/LOCAL_AI_ACTION.log",
    "**/LOCAL_AI_INVOCATION.env",
)
candidates = {prompt}
for pattern in patterns:
    candidates.update(path for path in runtime.glob(pattern) if path.is_file())
temporary = bundle.with_name(f".{bundle.name}.tmp")
with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED) as archive:
    archive.write(prompt, "LOCAL_AI_RECOVERY.md")
    bootstrap_state_value = __import__("os").environ.get("E8_COLDSTART_BOOTSTRAP_STATE", "")
    bootstrap_state = Path(bootstrap_state_value) if bootstrap_state_value else None
    if bootstrap_state is not None and bootstrap_state.is_file():
        archive.write(bootstrap_state, "bootstrap/BOOTSTRAP_STATE.env")
    for path in sorted(candidates - {prompt}):
        try:
            relative = path.relative_to(runtime).as_posix()
        except ValueError:
            continue
        data = path.read_bytes()
        if len(data) > 2 * 1024 * 1024:
            data = data[-2 * 1024 * 1024 :]
            relative += ".tail"
        archive.writestr(relative, data)
temporary.replace(bundle)
print(f"LOCAL_AI_RECOVERY_PROMPT={prompt}")
print(f"LOCAL_AI_RECOVERY_BUNDLE={bundle}")
PY
}

invoke_local_ai_recovery() {
  local failure_stage="$1"
  local hook="${E8_COLDSTART_LOCAL_AI_COMMAND:-}"
  [[ -n "${hook}" ]] || {
    echo "No E8_COLDSTART_LOCAL_AI_COMMAND configured; use ${RECOVERY_ROOT}/LOCAL_AI_RECOVERY_BUNDLE.zip" >&2
    return 1
  }
  [[ "${hook}" = /* && -f "${hook}" && -x "${hook}" ]] || {
    echo "Configured local-AI command must be one absolute executable file: ${hook}" >&2
    return 1
  }
  command -v timeout >/dev/null || {
    echo "timeout is required to invoke the local-AI recovery hook safely" >&2
    return 1
  }
  local timeout_seconds="${E8_COLDSTART_LOCAL_AI_TIMEOUT_SECONDS:-900}"
  local maximum_invocations="${E8_COLDSTART_LOCAL_AI_MAX_INVOCATIONS:-3}"
  [[ "${timeout_seconds}" =~ ^[1-9][0-9]*$ ]] || {
    echo "local-AI timeout must be a positive integer" >&2
    return 1
  }
  [[ "${maximum_invocations}" =~ ^[1-9][0-9]*$ ]] || {
    echo "local-AI maximum invocation count must be positive" >&2
    return 1
  }
  local count_file="${RECOVERY_ROOT}/LOCAL_AI_INVOCATION_COUNT"
  local invocation_count=0
  if [[ -f "${count_file}" ]]; then
    read -r invocation_count <"${count_file}"
    [[ "${invocation_count}" =~ ^[0-9]+$ ]] || {
      echo "local-AI invocation counter is corrupt" >&2
      return 1
    }
  fi
  if [[ "${invocation_count}" -ge "${maximum_invocations}" ]]; then
    echo "local-AI invocation limit reached (${maximum_invocations})" >&2
    return 1
  fi
  invocation_count=$((invocation_count + 1))
  printf '%s\n' "${invocation_count}" >"${count_file}.tmp"
  mv "${count_file}.tmp" "${count_file}"
  local prompt="${RECOVERY_ROOT}/LOCAL_AI_RECOVERY.md"
  local bundle="${RECOVERY_ROOT}/LOCAL_AI_RECOVERY_BUNDLE.zip"
  local action_log="${RECOVERY_ROOT}/LOCAL_AI_ACTION.log"
  local state="${RECOVERY_ROOT}/LOCAL_AI_INVOCATION.env"
  {
    printf 'experiment_id=%q\n' "${EXPERIMENT_ID}"
    printf 'source_commit=%q\n' "${EXPECTED_COMMIT}"
    printf 'failure_stage=%q\n' "${failure_stage}"
    printf 'invocation=%q\n' "${invocation_count}"
    printf 'hook=%q\n' "${hook}"
    printf 'status=%q\n' running
  } >"${state}"
  if DRPO_LOCAL_AI_PROMPT="${prompt}" \
    DRPO_LOCAL_AI_BUNDLE="${bundle}" \
    DRPO_LOCAL_AI_REPOSITORY="${ROOT_DIR}" \
    DRPO_LOCAL_AI_RUNTIME="${RUNTIME_ROOT}" \
    DRPO_LOCAL_AI_FAILURE_STAGE="${failure_stage}" \
    timeout --signal=TERM "${timeout_seconds}" "${hook}" "${prompt}" "${bundle}" \
      >"${action_log}" 2>&1
  then
    printf 'status=%q\n' completed >>"${state}"
    check_source
    echo "Local-AI hook completed; allowing one new guarded recovery attempt."
    return 0
  else
    local hook_status="$?"
    printf 'status=%q\n' failed >>"${state}"
    printf 'returncode=%q\n' "${hook_status}" >>"${state}"
    echo "Local-AI hook failed or timed out; inspect ${action_log}" >&2
    return 1
  fi
}

recover_import_if_requested() {
  [[ -n "${RECOVERY_SOURCE_OUTPUT:-}" ]] || return 0
  [[ -d "${RECOVERY_SOURCE_OUTPUT}" ]] || fail "recovery source vanished: ${RECOVERY_SOURCE_OUTPUT}"
  local recovery_source_commit
  if ! recovery_source_commit="$(
    python - "${RECOVERY_SOURCE_OUTPUT}/source_provenance.json" <<'PY_RECOVERY_SOURCE'
import json
import sys
from pathlib import Path

value = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
if not isinstance(value, dict) or not isinstance(value.get("source_commit"), str):
    raise SystemExit(2)
print(value["source_commit"])
PY_RECOVERY_SOURCE
  )"; then
    echo "Recovery source provenance is unreadable; starting fresh instead of retrying stale import." >&2
    RECOVERY_SOURCE_OUTPUT=""
    export RECOVERY_SOURCE_OUTPUT
    return 0
  fi
  if [[ "${recovery_source_commit}" != "${EXPECTED_COMMIT}" ]]; then
    echo "Recovery source commit ${recovery_source_commit} != ${EXPECTED_COMMIT}; starting fresh instead of retrying stale import." >&2
    RECOVERY_SOURCE_OUTPUT=""
    export RECOVERY_SOURCE_OUTPUT
    return 0
  fi
  local recovery_model="${MODEL_DIR}"
  if [[ "${MODE}" == "self-test" || "${MODE}" == "engineering-self-test-internal" ]]; then
    recovery_model="${RECOVERY_SOURCE_OUTPUT}/engineering_fixtures/placeholder_model"
  fi
  run_module import-recovery \
    --source-output-root "${RECOVERY_SOURCE_OUTPUT}" \
    --base-model-path "${recovery_model}" \
    --source-commit "${EXPECTED_COMMIT}"
}

engineering_self_test_internal() {
  recover_import_if_requested
  export E8_COLDSTART_RECOVERY_PACKAGE="${RECOVERY_PACKAGE}"
  export E8_COLDSTART_RECOVERY_INTERVAL_CELLS="${E8_COLDSTART_RECOVERY_INTERVAL_CELLS:-40}"
  run_module engineering-self-test --source-commit "${EXPECTED_COMMIT}"
}

engineering_self_test() {
  self_test_setup
  ATTEMPTS_ROOT="${E8_COLDSTART_SELFTEST_OUTPUT_ROOT:-${RUNTIME_ROOT}/self-test-guard/${RUN_ID}}"
  RECOVERY_ROOT="${RUNTIME_ROOT}/self-test-recovery/${RUN_ID}"
  RECOVERY_PACKAGE="${RECOVERY_ROOT}/latest_checkpoint.zip"
  command -v flock >/dev/null || fail "flock is required for single-writer recovery safety"
  mkdir -p "${RECOVERY_ROOT}"
  exec 8>"${RECOVERY_ROOT}/runtime.lock"
  flock -n 8 || fail "another engineering self-test still holds the recovery lock"
  # Re-bind the fast-path decision to the checkout after setup has finished.
  check_source
  if reuse_successful_attempt; then
    echo "ENGINEERING_SELF_TEST_ROOT=${OUTPUT_ROOT}"
    echo "ENGINEERING_SELF_TEST_GUARD_ARTIFACT=${GUARD_ARTIFACT}"
    return 0
  fi
  local automatic_attempts="${E8_COLDSTART_AUTO_RECOVERY_ATTEMPTS:-3}"
  local local_attempt=1
  while [[ "${local_attempt}" -le "${automatic_attempts}" ]]; do
    select_next_attempt
    write_attempt_state running
    if python "${ROOT_DIR}/scripts/run_experiment_guard_hardened.py" \
      --experiment-id "${EXPERIMENT_ID}" \
      --repo-root "${ROOT_DIR}" \
      --output-root "${GUARD_ROOT}" \
      --artifact-output "${GUARD_ARTIFACT}" \
      --run-class pilot \
      --expected-commit "${EXPECTED_COMMIT}" \
      --large-file-persistence persistent_local \
      --required-output workload/ENGINEERING_SELF_TEST_REPORT.json \
      --required-output workload/RUN_COMPLETE.json \
      --required-output workload/terminal_audit.json \
      --required-output workload/run_manifest.json \
      --required-output workload/scheduler/dynamic_run.json \
      --required-output workload/aggregate/plot_curve_points.csv \
      --source-file scripts/run_e8_multitask_exp_coldstart.sh \
      --source-file scripts/bootstrap_e8_multitask_exp_coldstart.sh \
      --source-file src/drpo/e8_multitask_exp_tuning.py \
    --source-file src/drpo/e8_experiment_config.py \
    --source-file scripts/preflight_e8_multitask_config.py \
      --source-file "${CONFIG_REPO_PATH}" \
      "${CONFIG_SOURCE_ARGS[@]}" \
      --source-file docs/experiments/EXT-C-E8-MULTITASK-EXP-COLDSTART-01_RUNBOOK.md \
      --progress-glob 'workload/scheduler/queue_events.jsonl' \
      --progress-glob 'workload/logs/*.log' \
      -- \
      bash "${ROOT_DIR}/scripts/run_e8_multitask_exp_coldstart.sh" engineering-self-test-internal
    then
      python "${ROOT_DIR}/scripts/verify_experiment_package_hardened.py" \
        --repo-root "${ROOT_DIR}" "${GUARD_ARTIFACT}"
      write_attempt_state complete
      echo "ENGINEERING_SELF_TEST_ROOT=${OUTPUT_ROOT}"
      echo "ENGINEERING_SELF_TEST_GUARD_ARTIFACT=${GUARD_ARTIFACT}"
      return 0
    fi
    write_attempt_state failed
    local_attempt=$((local_attempt + 1))
  done
  write_local_ai_recovery engineering_self_test
  fail "engineering self-test exhausted automatic recovery attempts"
}

prepare() {
  check_source
  activate_runtime
  preflight_gpu
  "${ROOT_DIR}/scripts/run_e8_multitask_p0.sh" \
    --work-dir "${P0_WORK_DIR}" prepare
  "${ROOT_DIR}/scripts/run_e8_multitask_p0.sh" \
    --work-dir "${P0_WORK_DIR}" qualify
  python "${ROOT_DIR}/scripts/run_countdown_e8_oracle_bank_v2.py" \
    --config "${ROOT_DIR}/configs/countdown_e8_oracle_offline_bank_v2_0p5b.yaml" \
    --work_dir "${COUNTDOWN_WORK_DIR}"
  python "${ROOT_DIR}/scripts/v2_bank_convert.py" \
    --input "${COUNTDOWN_WORK_DIR}/data/oracle_offline_bank_v2_train.jsonl" \
    --output "${COUNTDOWN_WORK_DIR}/data/offline_bank_v2.jsonl" \
    --manifest "${COUNTDOWN_WORK_DIR}/data/offline_bank_v2.convert_manifest.json" \
    --model "${MODEL_DIR}"
  run_module prepare \
    --p0-work-dir "${P0_WORK_DIR}" \
    --p0-config "${P0_CONFIG_PATH}" \
    --countdown-bank "${COUNTDOWN_WORK_DIR}/data/offline_bank_v2.jsonl" \
    --countdown-validation "${COUNTDOWN_WORK_DIR}/data/val.jsonl"
  python - <<PY
import json
from pathlib import Path
from drpo.e8_multitask_exp_tuning import load_config
config = load_config(Path("${CONFIG_PATH}"))
value = {
    "schema_version": 1,
    "run_id": "${RUN_ID}",
    "source_commit": "${EXPECTED_COMMIT}",
    "model_repo": config["model"]["base_model"],
    "model_revision": config["model"]["revision"],
    "model_path": str(Path("${MODEL_DIR}").resolve()),
    "test_partition_accessed": False,
}
path = Path("${OUTPUT_ROOT}") / "source_provenance.json"
path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY
}

calibrate() {
  check_source
  activate_runtime
  mkdir -p "${OUTPUT_ROOT}/logs/calibration"
  run_module calibrate --base-model-path "${MODEL_DIR}" \
    >"${OUTPUT_ROOT}/logs/calibration/no_calibration_identity_gate.log" 2>&1
}

liveness() {
  check_source
  activate_runtime
  CUDA_VISIBLE_DEVICES=0 LOCAL_RANK=0 run_module liveness \
    --task countdown \
    --base-model-path "${MODEL_DIR}"
}

run_queue() {
  check_source
  activate_runtime
  preflight_gpu
  run_module run-all --base-model-path "${MODEL_DIR}" --retry-incomplete
}

run_queue_with_retry() {
  local attempt=1
  local maximum_attempts="${E8_COLDSTART_QUEUE_ATTEMPTS:-3}"
  [[ "${maximum_attempts}" =~ ^[1-9][0-9]*$ ]] || fail "queue attempts must be a positive integer"
  while ! run_queue; do
    if [[ "${attempt}" -ge "${maximum_attempts}" ]]; then
      fail "dynamic queue still failed after ${attempt} guarded attempts"
    fi
    attempt=$((attempt + 1))
    echo "Retrying only incomplete/unscheduled cells inside the same guard: attempt ${attempt}"
  done
}

finish() {
  check_source
  activate_runtime
  run_module aggregate
  run_module audit
  run_module finalize
  python - <<PY
import json
from pathlib import Path
root = Path("${OUTPUT_ROOT}")
audit = json.loads((root / "terminal_audit.json").read_text())
assert audit["all_training_and_evaluation_complete"], audit
plot = root / "aggregate" / "plot_curve_points.csv"
print("RAW_RESULTS_ROOT=" + str(root.resolve()))
print("PLOT_CSV=" + str(plot.resolve()))
PY
}

refresh_recovery_plan() {
  run_module recovery-plan --base-model-path "${MODEL_DIR}" >/dev/null
}

plan_flag() {
  local key="$1"
  python - "${OUTPUT_ROOT}/recovery/RECOVERY_PLAN.json" "${key}" <<'PY'
import json
import sys
value = json.loads(open(sys.argv[1], encoding="utf-8").read())[sys.argv[2]]
raise SystemExit(0 if value is True else 1)
PY
}

finish_from_plan() {
  refresh_recovery_plan
  if ! plan_flag aggregate_complete; then
    run_module aggregate
  fi
  refresh_recovery_plan
  if ! plan_flag audit_complete; then
    run_module audit
  fi
  refresh_recovery_plan
  if ! plan_flag finalized; then
    run_module finalize
  fi
  refresh_recovery_plan
  plan_flag finalized || fail "recovery plan is not finalized after aggregate/audit/finalize"
}

delivery_preflight() {
  mkdir -p "${RECOVERY_ROOT}"
  local log="${RECOVERY_ROOT}/delivery_preflight.log"
  local command=(
    python "${ROOT_DIR}/scripts/package_experiment_hardened.py"
    --repo-root "${ROOT_DIR}"
    --experiment-id "${EXPERIMENT_ID}"
    --package-kind experiment-checkpoint
    --result-dir "${OUTPUT_ROOT}"
    --output "${DELIVERY_PREFLIGHT_PACKAGE}"
    --base-commit "${EXPECTED_COMMIT}"
    --no-repository-changes
    --large-file-persistence persistent_local
    --max-package-mib 23
    --source-file scripts/run_e8_multitask_exp_coldstart.sh
    --source-file scripts/bootstrap_e8_multitask_exp_coldstart.sh
    --source-file src/drpo/e8_multitask_exp_tuning.py
    --source-file src/drpo/e8_experiment_config.py
    --source-file scripts/preflight_e8_multitask_config.py
    --source-file "${CONFIG_REPO_PATH}"
    "${CONFIG_SOURCE_ARGS[@]}"
  )
  if [[ "${REQUIRE_ORIGIN_MAIN}" == "1" ]]; then
    command+=(--require-origin-main-match)
  fi
  if "${command[@]}" >"${log}" 2>&1; then
    return 0
  fi
  if grep -Fq "exceeding hard limit" "${log}"; then
    echo "Delivery preflight exceeded 23 MiB; compacting full logs with verified tails."
    run_module compact-logs
    "${command[@]}" >"${log}" 2>&1 || {
      tail -n 80 "${log}" >&2
      fail "delivery preflight still fails after deterministic log compaction"
    }
    return 0
  fi
  tail -n 80 "${log}" >&2
  fail "delivery preflight failed for a non-size reason"
}

guarded_full_internal() {
  recover_import_if_requested
  export E8_COLDSTART_RECOVERY_PACKAGE="${RECOVERY_PACKAGE}"
  export E8_COLDSTART_RECOVERY_INTERVAL_CELLS="${E8_COLDSTART_RECOVERY_INTERVAL_CELLS:-5}"
  export E8_COLDSTART_RECOVERY_REQUIRE_ORIGIN_MAIN="${REQUIRE_ORIGIN_MAIN}"
  refresh_recovery_plan
  if ! plan_flag prepare_complete; then
    prepare
  fi
  refresh_recovery_plan
  if ! plan_flag calibration_complete; then
    calibrate
  fi
  refresh_recovery_plan
  if ! plan_flag liveness_complete; then
    liveness
  fi
  refresh_recovery_plan
  if ! plan_flag cells_complete; then
    run_queue_with_retry
  fi
  finish_from_plan
  delivery_preflight
}

run_formal_guard_attempt() {
  [[ "${OUTPUT_ROOT}" == "${GUARD_ROOT}/workload" ]] || \
    fail "formal output root must be the guard workload child: ${GUARD_ROOT}/workload"
  [[ ! -e "${GUARD_ROOT}" ]] || fail "formal guard attempt root must be new: ${GUARD_ROOT}"
  [[ ! -e "${GUARD_ARTIFACT}" ]] || fail "guard artifact already exists: ${GUARD_ARTIFACT}"
  mkdir -p "$(dirname "${GUARD_ARTIFACT}")"
  local origin_main_args=()
  if [[ "${REQUIRE_ORIGIN_MAIN}" == "1" ]]; then
    origin_main_args+=(--require-origin-main-match)
  fi
  python "${ROOT_DIR}/scripts/run_experiment_guard_hardened.py" \
    --experiment-id "${EXPERIMENT_ID}" \
    --repo-root "${ROOT_DIR}" \
    --output-root "${GUARD_ROOT}" \
    --artifact-output "${GUARD_ARTIFACT}" \
    --run-class "${RUN_CLASS}" \
    --expected-commit "${EXPECTED_COMMIT}" \
    "${origin_main_args[@]}" \
    --large-file-persistence persistent_local \
    --required-output workload/RUN_COMPLETE.json \
    --required-output workload/terminal_audit.json \
    --required-output workload/run_manifest.json \
    --required-output workload/scientific_run_manifest.json \
    --required-output workload/scheduler/dynamic_run.json \
    --required-output workload/aggregate/plot_curve_points.csv \
    --source-file scripts/run_e8_multitask_exp_coldstart.sh \
    --source-file scripts/bootstrap_e8_multitask_exp_coldstart.sh \
    --source-file src/drpo/e8_multitask_exp_tuning.py \
    --source-file src/drpo/e8_experiment_config.py \
    --source-file scripts/preflight_e8_multitask_config.py \
    --source-file "${CONFIG_REPO_PATH}" \
    "${CONFIG_SOURCE_ARGS[@]}" \
    --source-file requirements/e8_multitask_exp_coldstart.txt \
    --source-file src/drpo/countdown_qwen_arena_onefile.py \
    --source-file src/drpo/countdown_e8_alpha1_c_scan_common.py \
    --source-file src/drpo/countdown_e8_alpha1_c_scan_runtime.py \
    --source-file src/drpo/countdown_e8_alpha1_c_scan_trainer.py \
    --source-file src/drpo/countdown_e8_alpha1_highc_scan_common.py \
    --source-file src/drpo/countdown_e8_alpha1_highc_scan_runtime.py \
    --source-file src/drpo/countdown_e8_oracle_bank_v2.py \
    --source-file scripts/v2_bank_convert.py \
    --source-file src/drpo/e8_multitask_p0.py \
    --source-file src/drpo/e8_multitask_tasks.py \
    --source-file configs/e8_multitask_p0.yaml \
    --source-file scripts/run_e8_multitask_p0.sh \
    --source-file docs/handoff.md \
    --source-file experiments/registry.yaml \
    --progress-glob 'workload/scheduler/queue_events.jsonl' \
    --progress-glob 'workload/logs/*.log' \
    -- \
    bash "${ROOT_DIR}/scripts/run_e8_multitask_exp_coldstart.sh" guarded-full-internal
}

report_formal_success() {
  python "${ROOT_DIR}/scripts/verify_experiment_package_hardened.py" \
    --repo-root "${ROOT_DIR}" "${GUARD_ARTIFACT}"
  write_attempt_state complete
  echo "RAW_COMPLETE_RESULTS_ZIP=${GUARD_ARTIFACT}"
  sha256sum "${GUARD_ARTIFACT}"
  echo "PLOT_CSV=${OUTPUT_ROOT}/aggregate/plot_curve_points.csv"
  sha256sum "${OUTPUT_ROOT}/aggregate/plot_curve_points.csv"
  echo "RECOVERY_CHECKPOINT=${RECOVERY_PACKAGE}"
  if [[ -f "${RECOVERY_PACKAGE}" ]]; then
    sha256sum "${RECOVERY_PACKAGE}"
  else
    echo "RECOVERY_CHECKPOINT_UNAVAILABLE=completed_without_queue_checkpoint"
  fi
}

guarded_full() {
  check_source
  check_authoritative_main_at_invocation
  ensure_setup
  command -v flock >/dev/null || fail "flock is required for single-writer recovery safety"
  mkdir -p "${RECOVERY_ROOT}"
  exec 9>"${RECOVERY_ROOT}/runtime.lock"
  if ! flock -n 9; then
    write_local_ai_recovery concurrent_runtime_lock
    fail "another experiment or recovery process still holds ${RECOVERY_ROOT}/runtime.lock"
  fi
  # Setup can be long. Re-check the current checkout and formal main authority
  # under the per-run lock immediately before a completed-attempt fast return.
  check_source
  check_authoritative_main_at_invocation
  if reuse_successful_attempt; then
    write_attempt_state complete
    echo "RAW_COMPLETE_RESULTS_ZIP=${GUARD_ARTIFACT}"
    sha256sum "${GUARD_ARTIFACT}"
    echo "PLOT_CSV=${OUTPUT_ROOT}/aggregate/plot_curve_points.csv"
    sha256sum "${OUTPUT_ROOT}/aggregate/plot_curve_points.csv"
    return 0
  fi
  local automatic_attempts="${E8_COLDSTART_AUTO_RECOVERY_ATTEMPTS:-3}"
  [[ "${automatic_attempts}" =~ ^[1-9][0-9]*$ ]] || fail "automatic recovery attempts must be positive"
  local local_attempt=1
  while [[ "${local_attempt}" -le "${automatic_attempts}" ]]; do
    select_next_attempt
    write_attempt_state running
    if run_formal_guard_attempt; then
      report_formal_success
      return 0
    fi
    write_attempt_state failed
    echo "Guard attempt ${local_attempt} failed; starting a new isolated recovery attempt." >&2
    local_attempt=$((local_attempt + 1))
  done
  write_local_ai_recovery formal_guard_attempts_exhausted
  if invoke_local_ai_recovery formal_guard_attempts_exhausted; then
    select_next_attempt
    write_attempt_state running_after_local_ai
    if run_formal_guard_attempt; then
      report_formal_success
      return 0
    fi
    write_attempt_state failed_after_local_ai
    write_local_ai_recovery post_local_ai_guard_failure
  else
    write_local_ai_recovery local_ai_hook_unavailable_or_failed
  fi
  fail "formal run exhausted automatic recovery attempts; use the generated local-AI handoff"
}

case "${MODE}" in
  self-test) engineering_self_test ;;
  setup) runtime_setup_lock; setup; runtime_setup_unlock ;;
  prepare) prepare ;;
  preflight) check_source; activate_runtime; config_preflight ;;
  plan) check_source; activate_runtime; run_module plan ;;
  calibrate) calibrate ;;
  liveness) liveness ;;
  run|resume) run_queue ;;
  finish) finish ;;
  engineering-self-test-internal) engineering_self_test_internal ;;
  guarded-full-internal) guarded_full_internal ;;
  diagnose) write_local_ai_recovery "${2:-manual_diagnosis}" ;;
  full) guarded_full ;;
  *) fail "usage: $0 {self-test|setup|preflight|prepare|plan|calibrate|liveness|run|resume|finish|full}" ;;
esac
