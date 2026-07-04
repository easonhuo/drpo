#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -P "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
DESTINATION="${DRPO_MACOS_APP_DEST:-$HOME/Applications/DRPO Update.app}"
OSACOMPILE_BIN="${DRPO_MACOS_OSACOMPILE:-/usr/bin/osacompile}"
PYTHON_FOR_PLIST="${DRPO_PYTHON:-python3}"
LSREGISTER_BIN="${DRPO_MACOS_LSREGISTER:-/System/Library/Frameworks/CoreServices.framework/Frameworks/LaunchServices.framework/Support/lsregister}"
CODESIGN_BIN="${DRPO_MACOS_CODESIGN:-/usr/bin/codesign}"
OSASCRIPT_BIN="${DRPO_MACOS_OSASCRIPT:-/usr/bin/osascript}"
DRY_RUN=0
SKIP_CLI_INSTALL="${DRPO_MACOS_SKIP_CLI_INSTALL:-0}"

usage() {
  cat <<'USAGE'
Usage:
  bash tools/drpo-update-macos/install.sh [--destination PATH] [--dry-run] [--skip-cli-install]

Installs a locally generated, ad-hoc-signed DRPO Update.app and registers the
.drpoupdate file extension. Future canonical update ZIPs may use the
.drpoupdate suffix and can then be opened by double-clicking them in Finder.
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --destination)
      [[ $# -ge 2 ]] || { echo "ERROR: --destination requires a path" >&2; exit 2; }
      DESTINATION="$2"
      shift 2
      ;;
    --dry-run) DRY_RUN=1; shift ;;
    --skip-cli-install) SKIP_CLI_INSTALL=1; shift ;;
    --help|-h) usage; exit 0 ;;
    *) echo "ERROR: unknown option: $1" >&2; usage; exit 2 ;;
  esac
done

if [[ "$(uname -s)" != "Darwin" && "${DRPO_MACOS_ALLOW_NON_DARWIN:-0}" != "1" ]]; then
  echo "ERROR: DRPO Update.app can only be installed on macOS." >&2
  exit 1
fi

for source in launcher.applescript run_update.sh Info.plist; do
  [[ -f "$SCRIPT_DIR/$source" ]] || {
    echo "ERROR: missing launcher source: $SCRIPT_DIR/$source" >&2
    exit 1
  }
done

git -C "$REPO_ROOT" rev-parse --show-toplevel >/dev/null 2>&1 || {
  echo "ERROR: repository root is not a Git checkout: $REPO_ROOT" >&2
  exit 1
}
REPO_ROOT="$(git -C "$REPO_ROOT" rev-parse --show-toplevel)"

if [[ "$DRY_RUN" -eq 1 ]]; then
  cat <<PLAN
DRPO Update.app installation plan
Repository:       $REPO_ROOT
Destination:      $DESTINATION
Install CLI:      $([[ "$SKIP_CLI_INSTALL" == "1" ]] && echo no || echo yes)
File association: .drpoupdate -> com.easonhuo.drpo-update
PLAN
  exit 0
fi

[[ -x "$OSACOMPILE_BIN" ]] || { echo "ERROR: osacompile not found: $OSACOMPILE_BIN" >&2; exit 1; }
[[ -x "$LSREGISTER_BIN" ]] || { echo "ERROR: lsregister not found: $LSREGISTER_BIN" >&2; exit 1; }
[[ -x "$CODESIGN_BIN" ]] || { echo "ERROR: codesign not found: $CODESIGN_BIN" >&2; exit 1; }
[[ -x "$OSASCRIPT_BIN" ]] || { echo "ERROR: osascript not found: $OSASCRIPT_BIN" >&2; exit 1; }

if [[ "$SKIP_CLI_INSTALL" != "1" ]]; then
  bash "$REPO_ROOT/tools/drpo-update/install.sh" "$REPO_ROOT"
fi

DEST_PARENT="$(dirname "$DESTINATION")"
mkdir -p "$DEST_PARENT"
WORK_DIR="$(mktemp -d "$DEST_PARENT/.drpo-update-app.XXXXXX")"
TEMP_APP="$WORK_DIR/DRPO Update.app"
BACKUP_APP="$DEST_PARENT/.DRPO Update.app.backup.$$"
INSTALLED_NEW=0

cleanup() {
  local status=$?
  trap - EXIT HUP INT TERM
  rm -rf "$WORK_DIR"
  if [[ "$status" -ne 0 && "$INSTALLED_NEW" -eq 1 ]]; then
    rm -rf "$DESTINATION"
  fi
  if [[ -e "$BACKUP_APP" ]]; then
    if [[ "$status" -eq 0 ]]; then
      rm -rf "$BACKUP_APP"
    else
      mv "$BACKUP_APP" "$DESTINATION"
    fi
  fi
  exit "$status"
}
trap cleanup EXIT
trap 'exit 129' HUP
trap 'exit 130' INT
trap 'exit 143' TERM

"$OSACOMPILE_BIN" -o "$TEMP_APP" "$SCRIPT_DIR/launcher.applescript"
APP_EXECUTABLE=""
if [[ -f "$TEMP_APP/Contents/Info.plist" ]]; then
  APP_EXECUTABLE="$($PYTHON_FOR_PLIST - "$TEMP_APP/Contents/Info.plist" <<'PY'
import plistlib
import sys
from pathlib import Path
try:
    data = plistlib.loads(Path(sys.argv[1]).read_bytes())
except Exception:
    data = {}
print(data.get("CFBundleExecutable", ""))
PY
)"
fi
if [[ -z "$APP_EXECUTABLE" || ! -x "$TEMP_APP/Contents/MacOS/$APP_EXECUTABLE" ]]; then
  while IFS= read -r candidate; do
    [[ -x "$candidate" ]] || continue
    APP_EXECUTABLE="$(basename "$candidate")"
    break
  done < <(find "$TEMP_APP/Contents/MacOS" -maxdepth 1 -type f 2>/dev/null | sort)
fi
[[ -n "$APP_EXECUTABLE" && -x "$TEMP_APP/Contents/MacOS/$APP_EXECUTABLE" ]] || {
  echo "ERROR: osacompile did not produce an executable applet" >&2
  find "$TEMP_APP/Contents" -maxdepth 3 -type f -print >&2 2>/dev/null || true
  exit 1
}
mkdir -p "$TEMP_APP/Contents/Resources"
cp "$SCRIPT_DIR/run_update.sh" "$TEMP_APP/Contents/Resources/run_update.sh"
chmod 0755 "$TEMP_APP/Contents/Resources/run_update.sh"
"$PYTHON_FOR_PLIST" - \
  "$TEMP_APP/Contents/Info.plist" \
  "$SCRIPT_DIR/Info.plist" \
  "$APP_EXECUTABLE" <<'PY'
import plistlib
import sys
from pathlib import Path
native_path = Path(sys.argv[1])
overlay_path = Path(sys.argv[2])
executable = sys.argv[3]
native = plistlib.loads(native_path.read_bytes())
overlay = plistlib.loads(overlay_path.read_bytes())

# osacompile selects the executable, signature, icon, and droplet runtime
# metadata for the current macOS release. Preserve those native values and
# merge only DRPO branding plus the document/UTI declaration.
preserved = {
    key: native.get(key)
    for key in (
        "CFBundleExecutable",
        "CFBundleIconFile",
        "CFBundlePackageType",
        "CFBundleSignature",
    )
}
native.update(overlay)
for key, value in preserved.items():
    if value is not None:
        native[key] = value
native["CFBundleExecutable"] = executable
native_path.write_bytes(plistlib.dumps(native, sort_keys=False))
PY
"$CODESIGN_BIN" --force --deep --sign - "$TEMP_APP" >/dev/null
"$CODESIGN_BIN" --verify --deep --strict "$TEMP_APP"

if [[ -e "$DESTINATION" ]]; then
  rm -rf "$BACKUP_APP"
  mv "$DESTINATION" "$BACKUP_APP"
fi
mv "$TEMP_APP" "$DESTINATION"
INSTALLED_NEW=1

"$LSREGISTER_BIN" -f "$DESTINATION"
"$OSASCRIPT_BIN" -l JavaScript \
  -e 'ObjC.import("CoreServices");' \
  -e 'const status = $.LSSetDefaultRoleHandlerForContentType($("com.easonhuo.drpoupdate"), $.kLSRolesAll, $("com.easonhuo.drpo-update"));' \
  -e 'if (status !== 0) { throw new Error("LSSetDefaultRoleHandlerForContentType failed: " + status); }'
touch "$DESTINATION"
INSTALLED_NEW=0

cat <<MSG
Installed DRPO Update.app successfully.

Application: $DESTINATION
Repository:  $REPO_ROOT
File type:   .drpoupdate

Future update packages can be downloaded with a .drpoupdate suffix and opened
by double-clicking them. The app opens Terminal and delegates to the canonical
drpo-update helper; it does not bypass tests, review, commit, push, or diagnostics.
MSG
