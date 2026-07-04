#!/usr/bin/env bash
set -euo pipefail

DESTINATION="${DRPO_MACOS_APP_DEST:-$HOME/Applications/DRPO Update.app}"
LSREGISTER_BIN="${DRPO_MACOS_LSREGISTER:-/System/Library/Frameworks/CoreServices.framework/Frameworks/LaunchServices.framework/Support/lsregister}"

if [[ -e "$DESTINATION" ]]; then
  if [[ -x "$LSREGISTER_BIN" ]]; then
    "$LSREGISTER_BIN" -u "$DESTINATION" >/dev/null 2>&1 || true
  fi
  rm -rf "$DESTINATION"
  echo "Removed $DESTINATION"
else
  echo "DRPO Update.app is not installed at $DESTINATION"
fi

echo "The drpo-update CLI and repository configuration were left unchanged."
