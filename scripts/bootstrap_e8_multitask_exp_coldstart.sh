#!/usr/bin/env bash
set -Eeuo pipefail

umask 077

EXPERIMENT_ID="EXT-C-E8-MULTITASK-EXP-COLDSTART-01"
EXPECTED_REPOSITORY="https://github.com/easonhuo/drpo.git"
MODE="${E8_COLDSTART_EXECUTION_MODE:-${1:-full}}"
CURRENT_STAGE="bootstrap_init"
BOOTSTRAP_STATUS="initializing"
BOOTSTRAP_OWNS_ROOT=0

fail() {
  echo "ERROR: $*" >&2
  if [[ "${BOOTSTRAP_OWNS_ROOT}" -eq 1 ]] && declare -F write_state >/dev/null; then
    BOOTSTRAP_STATUS="failed"
    write_state "${BOOTSTRAP_STATUS}"
    if [[ -f "${CHECKOUT:-}/scripts/run_e8_multitask_exp_coldstart.sh" ]]; then
      E8_COLDSTART_EXPECTED_COMMIT="${TARGET_COMMIT:-}" \
        E8_COLDSTART_RUNTIME_ROOT="${RUNTIME_ROOT:-${BOOTSTRAP_ROOT:-.}/runtime}" \
        E8_COLDSTART_BOOTSTRAP_STATE="${STATE_FILE}" \
        bash "${CHECKOUT}/scripts/run_e8_multitask_exp_coldstart.sh" \
          diagnose "bootstrap_${CURRENT_STAGE}" || true
    fi
    echo "BOOTSTRAP_FAILED_STAGE=${CURRENT_STAGE}" >&2
    echo "BOOTSTRAP_STATE=${STATE_FILE}" >&2
  fi
  exit 2
}

case "${MODE}" in
  full|self-test) ;;
  *) fail "usage: $0 {full|self-test}" ;;
esac

if [[ -n "${E8_COLDSTART_BOOTSTRAP_PARENT:-}" ]]; then
  BOOTSTRAP_PARENT="${E8_COLDSTART_BOOTSTRAP_PARENT}"
elif [[ -d /root && -w /root ]]; then
  BOOTSTRAP_PARENT="/root"
elif [[ -d /workspace && -w /workspace ]]; then
  BOOTSTRAP_PARENT="/workspace"
else
  BOOTSTRAP_PARENT="$(pwd -P)"
fi

BOOTSTRAP_ROOT="${E8_COLDSTART_BOOTSTRAP_ROOT:-${BOOTSTRAP_PARENT}/drpo-e8-coldstart-${MODE}}"
STATE_FILE="${BOOTSTRAP_ROOT}/BOOTSTRAP_STATE.env"
CHECKOUT="${BOOTSTRAP_ROOT}/repo"
RUNTIME_ROOT="${BOOTSTRAP_ROOT}/runtime"

write_state() {
  local status="$1"
  mkdir -p "${BOOTSTRAP_ROOT}"
  {
    printf 'experiment_id=%q\n' "${EXPERIMENT_ID}"
    printf 'mode=%q\n' "${MODE}"
    printf 'status=%q\n' "${status}"
    printf 'stage=%q\n' "${CURRENT_STAGE}"
    printf 'source_repo=%q\n' "${SOURCE_REPO:-}"
    printf 'source_remote=%q\n' "${SOURCE_REMOTE:-}"
    printf 'target_ref=%q\n' "${TARGET_REF:-}"
    printf 'target_commit=%q\n' "${TARGET_COMMIT:-}"
    printf 'checkout=%q\n' "${CHECKOUT}"
    printf 'runtime_root=%q\n' "${RUNTIME_ROOT}"
  } >"${STATE_FILE}"
}

on_error() {
  local status="$?"
  BOOTSTRAP_STATUS="failed"
  write_state "${BOOTSTRAP_STATUS}"
  if [[ -f "${CHECKOUT}/scripts/run_e8_multitask_exp_coldstart.sh" ]]; then
    E8_COLDSTART_EXPECTED_COMMIT="${TARGET_COMMIT:-}" \
      E8_COLDSTART_RUNTIME_ROOT="${RUNTIME_ROOT}" \
      E8_COLDSTART_BOOTSTRAP_STATE="${STATE_FILE}" \
      bash "${CHECKOUT}/scripts/run_e8_multitask_exp_coldstart.sh" \
        diagnose "bootstrap_${CURRENT_STAGE}" || true
  fi
  echo "BOOTSTRAP_FAILED_STAGE=${CURRENT_STAGE}" >&2
  echo "BOOTSTRAP_STATE=${STATE_FILE}" >&2
  exit "${status}"
}

RESUME_BOOTSTRAP=0
BOOTSTRAP_WAS_COMPLETE=0
if [[ -e "${BOOTSTRAP_ROOT}" ]]; then
  [[ -d "${BOOTSTRAP_ROOT}" ]] || fail "bootstrap root exists but is not a directory"
  [[ -f "${STATE_FILE}" ]] || fail "existing bootstrap root has no identity state: ${STATE_FILE}"
  [[ -d "${CHECKOUT}" ]] || fail "existing bootstrap root has no isolated checkout: ${CHECKOUT}"
  grep -Fqx "experiment_id=${EXPERIMENT_ID}" "${STATE_FILE}" || \
    fail "existing bootstrap state belongs to another experiment"
  grep -Fqx "mode=${MODE}" "${STATE_FILE}" || \
    fail "existing bootstrap state was created for another mode"
  RESUME_BOOTSTRAP=1
  if grep -Fqx "status=complete" "${STATE_FILE}"; then
    BOOTSTRAP_WAS_COMPLETE=1
  fi
  BOOTSTRAP_STATUS="recovering"
else
  mkdir -p "${BOOTSTRAP_ROOT}"
fi
BOOTSTRAP_OWNS_ROOT=1
write_state "${BOOTSTRAP_STATUS}"
trap on_error ERR

is_canonical_origin() {
  local url="$1"
  [[ "${url}" =~ ^https://([^/@]+(:[^/@]*)?@)?github\.com/easonhuo/drpo(\.git)?/?$ ]] ||
    [[ "${url}" =~ ^git@github\.com:easonhuo/drpo(\.git)?$ ]] ||
    [[ "${url}" =~ ^ssh://git@github\.com/easonhuo/drpo(\.git)?/?$ ]]
}

canonical_checkout() {
  local candidate="$1"
  [[ -d "${candidate}" ]] || return 1
  git -C "${candidate}" rev-parse --is-inside-work-tree >/dev/null 2>&1 || return 1
  local origin_url
  origin_url="$(git -C "${candidate}" remote get-url origin 2>/dev/null)" || return 1
  is_canonical_origin "${origin_url}" || return 1
  printf '%s\n' "$(cd "${candidate}" && pwd -P)"
}

CURRENT_STAGE="discover_canonical_checkout"
declare -a candidates=()
declare -A seen=()

add_candidate() {
  local candidate="$1"
  [[ -n "${candidate}" ]] || return 0
  [[ -z "${seen[${candidate}]:-}" ]] || return 0
  seen["${candidate}"]=1
  candidates+=("${candidate}")
}

add_candidate "${E8_COLDSTART_EXISTING_REPO:-}"
add_candidate "/root/drpo"
add_candidate "/root/d4rl2"
add_candidate "/workspace/drpo"
add_candidate "/data/drpo"
add_candidate "/mnt/data/drpo"
add_candidate "$(pwd -P)"

if [[ "${RESUME_BOOTSTRAP}" -eq 1 ]]; then
  SOURCE_REPO="${CHECKOUT}"
  SOURCE_REMOTE="origin"
else
  SOURCE_REPO=""
  SOURCE_REMOTE=""
  for candidate in "${candidates[@]}"; do
    if resolved="$(canonical_checkout "${candidate}")"; then
      SOURCE_REPO="${resolved}"
      SOURCE_REMOTE="origin"
      break
    fi
  done
fi

if [[ -z "${SOURCE_REPO}" ]]; then
  IFS=: read -r -a search_roots <<< \
    "${E8_COLDSTART_SEARCH_ROOTS:-/root:/home:/workspace:/data:/mnt}"
  for search_root in "${search_roots[@]}"; do
    [[ -d "${search_root}" ]] || continue
    while IFS= read -r -d '' git_marker; do
      candidate="$(dirname "${git_marker}")"
      if resolved="$(canonical_checkout "${candidate}")"; then
        SOURCE_REPO="${resolved}"
        SOURCE_REMOTE="origin"
        break 2
      fi
    done < <(
      find "${search_root}" -maxdepth 5 \
        \( -type d -o -type f \) -name .git -print0 2>/dev/null
    )
  done
fi

if [[ -z "${SOURCE_REPO}" ]]; then
  CURRENT_STAGE="clone_fallback"
  SOURCE_REPO="${BOOTSTRAP_ROOT}/source-cache"
  git clone --filter=blob:none --no-checkout "${EXPECTED_REPOSITORY}" "${SOURCE_REPO}"
  SOURCE_REMOTE="origin"
fi

CURRENT_STAGE="verify_selected_origin"
origin_url="$(git -C "${SOURCE_REPO}" remote get-url "${SOURCE_REMOTE}")"
is_canonical_origin "${origin_url}" || fail "selected checkout does not have canonical origin"

if [[ "${MODE}" == "full" ]]; then
  TARGET_REF="refs/heads/main"
  LOCAL_FETCH_REF="refs/remotes/origin/main"
else
  TARGET_REF="refs/pull/309/head"
  LOCAL_FETCH_REF="refs/e8-coldstart-bootstrap/pr-309-head"
fi

CURRENT_STAGE="fetch_authoritative_ref"
if [[ "${BOOTSTRAP_WAS_COMPLETE}" -eq 1 ]]; then
  TARGET_COMMIT="$(git -C "${CHECKOUT}" rev-parse HEAD)"
else
  git -C "${SOURCE_REPO}" fetch --no-tags --force "${SOURCE_REMOTE}" \
    "${TARGET_REF}:${LOCAL_FETCH_REF}"
  TARGET_COMMIT="$(git -C "${SOURCE_REPO}" rev-parse "${LOCAL_FETCH_REF}^{commit}")"
  [[ "${TARGET_COMMIT}" =~ ^[0-9a-f]{40}$ ]] || fail "resolved commit is not a full SHA"

  REMOTE_COMMIT="$(
    git -C "${SOURCE_REPO}" ls-remote "${SOURCE_REMOTE}" "${TARGET_REF}" |
      awk -v ref="${TARGET_REF}" '$2 == ref {print $1}'
  )"
  [[ "${REMOTE_COMMIT}" == "${TARGET_COMMIT}" ]] || \
    fail "fetch/authoritative-ref mismatch for ${TARGET_REF}"
fi

CURRENT_STAGE="create_isolated_worktree"
if [[ "${RESUME_BOOTSTRAP}" -eq 0 ]]; then
  git -C "${SOURCE_REPO}" worktree add --detach "${CHECKOUT}" "${TARGET_COMMIT}"
fi
[[ "$(git -C "${CHECKOUT}" rev-parse HEAD)" == "${TARGET_COMMIT}" ]] || \
  fail "existing isolated checkout no longer matches authoritative ${TARGET_REF}; local AI review is required"
[[ -z "$(git -C "${CHECKOUT}" status --porcelain=v1 --untracked-files=all)" ]] || \
  fail "isolated checkout is not clean"
[[ -f "${CHECKOUT}/scripts/run_e8_multitask_exp_coldstart.sh" ]] || \
  fail "target commit does not contain the reviewed experiment entrypoint"

BOOTSTRAP_STATUS="prepared"
write_state "${BOOTSTRAP_STATUS}"

CURRENT_STAGE="execute_${MODE}"
export E8_COLDSTART_EXPECTED_COMMIT="${TARGET_COMMIT}"
export E8_COLDSTART_RUNTIME_ROOT="${RUNTIME_ROOT}"
export E8_COLDSTART_BOOTSTRAP_STATE="${STATE_FILE}"
bash "${CHECKOUT}/scripts/run_e8_multitask_exp_coldstart.sh" "${MODE}"

CURRENT_STAGE="complete"
BOOTSTRAP_STATUS="complete"
write_state "${BOOTSTRAP_STATUS}"
echo "BOOTSTRAP_SOURCE_REPO=${SOURCE_REPO}"
echo "BOOTSTRAP_CHECKOUT=${CHECKOUT}"
echo "BOOTSTRAP_TARGET_COMMIT=${TARGET_COMMIT}"
echo "BOOTSTRAP_STATE=${STATE_FILE}"
