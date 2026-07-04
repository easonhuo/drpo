#!/usr/bin/env bash
set -uo pipefail

PROGRAM_NAME="DRPO Update Launcher"
CONFIG_DIR="${DRPO_LAUNCHER_CONFIG_DIR:-$HOME/.config/drpo-update}"
LOG_DIR="${DRPO_LAUNCHER_LOG_DIR:-$CONFIG_DIR/launcher-logs}"
LOCK_DIR="${DRPO_LAUNCHER_LOCK_DIR:-$CONFIG_DIR/launcher.lock}"
DIAGNOSTIC_DIR="${DRPO_LAUNCHER_DIAGNOSTIC_DIR:-$HOME/Downloads}"
OSASCRIPT_BIN="${DRPO_LAUNCHER_OSASCRIPT:-/usr/bin/osascript}"
PYTHON_BIN="${DRPO_PYTHON:-python3}"

notify() {
  local title="$1"
  local message="$2"
  if [[ "${DRPO_LAUNCHER_NO_NOTIFY:-0}" == "1" || ! -x "$OSASCRIPT_BIN" ]]; then
    return 0
  fi
  "$OSASCRIPT_BIN" \
    -e 'on run argv' \
    -e 'display notification (item 2 of argv) with title (item 1 of argv)' \
    -e 'end run' \
    "$title" "$message" >/dev/null 2>&1 || true
}

fail() {
  local code="$1"
  shift
  printf '\n%s: %s\n' "$PROGRAM_NAME" "$*" >&2
  notify "DRPO update failed" "$*"
  exit "$code"
}

absolute_path() {
  local input="$1"
  local directory
  local filename
  directory="$(cd -P "$(dirname "$input")" 2>/dev/null && pwd)" || return 1
  filename="$(basename "$input")"
  printf '%s/%s\n' "$directory" "$filename"
}

PACKAGE_INPUT="${1:-}"
[[ -n "$PACKAGE_INPUT" ]] || fail 64 "No update package was supplied."
PACKAGE="$(absolute_path "$PACKAGE_INPUT")" || fail 66 "Cannot resolve package path: $PACKAGE_INPUT"
[[ -f "$PACKAGE" ]] || fail 66 "Package not found: $PACKAGE"

case "${PACKAGE##*.}" in
  drpoupdate|DRPOUPDATE|zip|ZIP) ;;
  *) fail 65 "Expected a .drpoupdate or .zip package: $PACKAGE" ;;
esac

mkdir -p "$CONFIG_DIR" "$LOG_DIR" "$DIAGNOSTIC_DIR"
acquire_lock() {
  if mkdir "$LOCK_DIR" 2>/dev/null; then
    printf '%s\n' "$$" > "$LOCK_DIR/pid"
    return 0
  fi
  local owner_pid=""
  if [[ -f "$LOCK_DIR/pid" ]]; then
    owner_pid="$(head -n 1 "$LOCK_DIR/pid" 2>/dev/null || true)"
  fi
  if [[ "$owner_pid" =~ ^[0-9]+$ ]] && ! kill -0 "$owner_pid" 2>/dev/null; then
    rm -rf "$LOCK_DIR"
    mkdir "$LOCK_DIR"
    printf '%s\n' "$$" > "$LOCK_DIR/pid"
    return 0
  fi
  return 1
}
acquire_lock || fail 75 "Another DRPO update is already running."
cleanup() {
  local status=$?
  trap - EXIT HUP INT TERM
  rm -f "${PACKAGE_ALIAS:-}"
  rm -rf "${RUN_DIAGNOSTIC_DIR:-}" "$LOCK_DIR"
  exit "$status"
}
trap cleanup EXIT
trap 'exit 129' HUP
trap 'exit 130' INT
trap 'exit 143' TERM

REPO="${DRPO_LAUNCHER_REPO:-}"
if [[ -z "$REPO" ]]; then
  REPO_CONFIG="$CONFIG_DIR/repo_path"
  [[ -f "$REPO_CONFIG" ]] || fail 78 \
    "Repository is not configured. Run tools/drpo-update-macos/install.sh once."
  REPO="$(head -n 1 "$REPO_CONFIG")"
fi
REPO="$(absolute_path "$REPO")" || fail 78 "Cannot resolve repository path: $REPO"
git -C "$REPO" rev-parse --show-toplevel >/dev/null 2>&1 || fail 78 \
  "Configured repository is not a Git checkout: $REPO"
REPO="$(git -C "$REPO" rev-parse --show-toplevel)"

UPDATER="${DRPO_LAUNCHER_UPDATER:-$HOME/bin/drpo-update}"
if [[ ! -x "$UPDATER" ]]; then
  UPDATER="$REPO/tools/drpo-update/drpo-update"
fi
[[ -x "$UPDATER" ]] || fail 69 \
  "drpo-update is not installed. Run tools/drpo-update-macos/install.sh once."

PACKAGE_FOR_UPDATER="$PACKAGE"
PACKAGE_ALIAS=""
case "${PACKAGE##*.}" in
  drpoupdate|DRPOUPDATE)
    alias_base="$(mktemp "$CONFIG_DIR/.launcher-package.XXXXXX")" || \
      fail 73 "Could not reserve a temporary ZIP alias."
    rm -f "$alias_base"
    PACKAGE_ALIAS="${alias_base}.zip"
    if ! ln "$PACKAGE" "$PACKAGE_ALIAS" 2>/dev/null; then
      cp "$PACKAGE" "$PACKAGE_ALIAS" || \
        fail 73 "Could not create a temporary ZIP alias for: $PACKAGE"
    fi
    PACKAGE_FOR_UPDATER="$PACKAGE_ALIAS"
    ;;
esac

TIMESTAMP="$(date '+%Y%m%d-%H%M%S')"
PACKAGE_STEM="$(basename "$PACKAGE")"
PACKAGE_STEM="${PACKAGE_STEM%.*}"
SAFE_STEM="$(printf '%s' "$PACKAGE_STEM" | tr -cs 'A-Za-z0-9._-' '_')"
LOG_FILE="$LOG_DIR/${TIMESTAMP}_${SAFE_STEM}.log"
RUN_DIAGNOSTIC_DIR="$(mktemp -d "$CONFIG_DIR/.launcher-diagnostics.XXXXXX")" || \
  fail 73 "Could not create a per-run diagnostic staging directory."

printf '%s\n' "=============================================="
printf '%s\n' "DRPO Update"
printf 'Package:    %s\n' "$PACKAGE"
printf 'Repository: %s\n' "$REPO"
printf 'Log:        %s\n' "$LOG_FILE"
printf '%s\n\n' "=============================================="

set +e
"$UPDATER" "$PACKAGE_FOR_UPDATER" --yes --diagnostic-dir "$RUN_DIAGNOSTIC_DIR" 2>&1 | tee "$LOG_FILE"
STATUS=${PIPESTATUS[0]}
set -e

if [[ "$STATUS" -eq 0 ]]; then
  HEAD_SHA="$(git -C "$REPO" rev-parse HEAD 2>/dev/null || true)"
  SHORT_SHA="${HEAD_SHA:0:12}"
  printf '\nUpdate completed successfully.\nCommit: %s\n' "$HEAD_SHA" | tee -a "$LOG_FILE"
  notify "DRPO update complete" "Commit $SHORT_SHA"
  exit 0
fi

DIAGNOSTIC=""
for candidate in "$RUN_DIAGNOSTIC_DIR"/DRPO_DIAGNOSTIC_*.zip; do
  [[ -f "$candidate" ]] || continue
  destination="$DIAGNOSTIC_DIR/$(basename "$candidate")"
  if [[ -e "$destination" ]]; then
    destination="${destination%.zip}_${TIMESTAMP}_$$.zip"
  fi
  if mv "$candidate" "$destination" 2>/dev/null || \
      { cp "$candidate" "$destination" && rm -f "$candidate"; }; then
    DIAGNOSTIC="$destination"
  fi
done
if [[ -z "$DIAGNOSTIC" ]]; then
  DIAGNOSTIC="$DIAGNOSTIC_DIR/DRPO_MACOS_LAUNCHER_DIAGNOSTIC_${TIMESTAMP}_$$.zip"
  STATE_FILE="$RUN_DIAGNOSTIC_DIR/launcher_state.txt"
  {
    printf 'launcher_exit_code=%s\n' "$STATUS"
    printf 'package=%s\n' "$PACKAGE"
    printf 'repository=%s\n' "$REPO"
    printf 'head=%s\n' "$(git -C "$REPO" rev-parse HEAD 2>/dev/null || true)"
    printf 'status=%s\n' "$(git -C "$REPO" status --short 2>/dev/null || true)"
  } > "$STATE_FILE"
  "$PYTHON_BIN" - "$DIAGNOSTIC" "$LOG_FILE" "$STATE_FILE" <<'PY'
from pathlib import Path
import sys
import zipfile

output, log, state = map(Path, sys.argv[1:])
with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
    archive.writestr(
        "README.txt",
        "DRPO macOS launcher failure diagnostic. The repository was not modified by the launcher.\n",
    )
    archive.write(log, "launcher.log")
    archive.write(state, "launcher_state.txt")
PY
fi
{
  printf '\nUpdate failed with exit code %s.\n' "$STATUS"
  printf 'Log: %s\n' "$LOG_FILE"
  printf 'Diagnostic: %s\n' "$DIAGNOSTIC"
} | tee -a "$LOG_FILE" >&2
if [[ -n "$DIAGNOSTIC" ]]; then
  notify "DRPO update failed" "Diagnostic: $(basename "$DIAGNOSTIC")"
else
  printf 'No new diagnostic ZIP was found in %s.\n' "$DIAGNOSTIC_DIR" >&2
  notify "DRPO update failed" "See launcher log: $(basename "$LOG_FILE")"
fi
exit "$STATUS"
