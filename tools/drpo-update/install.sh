#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  bash tools/drpo-update/install.sh [/absolute/path/to/drpo] [--copy]

Default installation uses a symlink from ~/bin/drpo-update to the repository
copy, so future repository updates automatically update the installed helper.
Use --copy only when a symlink is undesirable.
USAGE
}

MODE="symlink"
REPO_PATH=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --copy) MODE="copy"; shift ;;
    --help|-h) usage; exit 0 ;;
    --*) echo "ERROR: unknown option: $1" >&2; usage; exit 2 ;;
    *)
      [[ -z "$REPO_PATH" ]] || { echo "ERROR: only one repository path may be supplied" >&2; exit 2; }
      REPO_PATH="$1"; shift ;;
  esac
done
REPO_PATH="${REPO_PATH:-$(pwd)}"
REPO_PATH="$(cd "$REPO_PATH" && pwd)"

command -v git >/dev/null 2>&1 || { echo "ERROR: git is not installed" >&2; exit 1; }
git -C "$REPO_PATH" rev-parse --show-toplevel >/dev/null 2>&1 || {
  echo "ERROR: $REPO_PATH is not a Git repository" >&2
  exit 1
}
TOPLEVEL="$(git -C "$REPO_PATH" rev-parse --show-toplevel)"
SOURCE="$TOPLEVEL/tools/drpo-update/drpo-update"
PY_SOURCE="$TOPLEVEL/tools/drpo-update/drpo_update.py"
SELECTION_SOURCE="$TOPLEVEL/tools/drpo-update/test_selection.py"
[[ -x "$SOURCE" ]] || { echo "ERROR: repository helper is missing or not executable: $SOURCE" >&2; exit 1; }
[[ -f "$PY_SOURCE" ]] || { echo "ERROR: repository helper runtime is missing: $PY_SOURCE" >&2; exit 1; }
[[ -f "$SELECTION_SOURCE" ]] || { echo "ERROR: repository test selector is missing: $SELECTION_SOURCE" >&2; exit 1; }

REMOTE_URL="$(git -C "$TOPLEVEL" remote get-url origin 2>/dev/null || true)"
if [[ "${DRPO_UPDATE_ALLOW_ANY_REMOTE:-0}" != "1" ]]; then
  case "$REMOTE_URL" in
    *github.com/easonhuo/drpo*|*github.com:easonhuo/drpo*) ;;
    *) echo "ERROR: origin does not look like easonhuo/drpo: $REMOTE_URL" >&2; exit 1 ;;
  esac
fi

BIN_DIR="$HOME/bin"
CFG_DIR="$HOME/.config/drpo-update"
mkdir -p "$BIN_DIR" "$CFG_DIR"
TARGET="$BIN_DIR/drpo-update"
PY_TARGET="$BIN_DIR/drpo_update.py"
SELECTION_TARGET="$BIN_DIR/test_selection.py"
TMP_TARGET="$BIN_DIR/.drpo-update.install.$$"
TMP_PY_TARGET="$BIN_DIR/.drpo_update.py.install.$$"
TMP_SELECTION_TARGET="$BIN_DIR/.test_selection.py.install.$$"
cleanup() {
  rm -f "$TMP_TARGET" "$TMP_PY_TARGET" "$TMP_SELECTION_TARGET"
}
trap cleanup EXIT
cleanup
if [[ "$MODE" == "symlink" ]]; then
  ln -s "$SOURCE" "$TMP_TARGET"
  mv -f "$TMP_TARGET" "$TARGET"
else
  cp "$SOURCE" "$TMP_TARGET"
  cp "$PY_SOURCE" "$TMP_PY_TARGET"
  cp "$SELECTION_SOURCE" "$TMP_SELECTION_TARGET"
  chmod +x "$TMP_TARGET"
  chmod 0644 "$TMP_PY_TARGET" "$TMP_SELECTION_TARGET"
  mv -f "$TMP_PY_TARGET" "$PY_TARGET"
  mv -f "$TMP_SELECTION_TARGET" "$SELECTION_TARGET"
  mv -f "$TMP_TARGET" "$TARGET"
fi
printf '%s\n' "$TOPLEVEL" > "$CFG_DIR/repo_path"
printf '%s\n' "$MODE" > "$CFG_DIR/install_mode"

SHELL_RC="$HOME/.zshrc"
if [[ "${SHELL##*/}" == "bash" ]]; then
  SHELL_RC="$HOME/.bashrc"
fi
touch "$SHELL_RC"
if ! grep -Fq 'export PATH="$HOME/bin:$PATH"' "$SHELL_RC"; then
  printf '\n# DRPO local update helper\nexport PATH="$HOME/bin:$PATH"\n' >> "$SHELL_RC"
fi

cat <<MSG
Installed drpo-update successfully.

Repository: $TOPLEVEL
Command:    $TARGET
Mode:       $MODE
Version:    $("$TARGET" --version)

Open a new terminal, then apply future update packages with:
  drpo-update ~/Downloads/<bundle>.zip --yes
MSG
