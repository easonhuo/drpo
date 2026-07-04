# DRPO Update.app for macOS

This is a thin macOS launcher for the existing transactional `drpo-update`
helper. It does not implement a second update path and does not bypass package
validation, isolated integration, tests, review, commit, push, or diagnostics.

## Install once

From the DRPO repository root on macOS:

```bash
bash tools/drpo-update-macos/install.sh
```

The installer:

1. installs or refreshes the canonical `drpo-update` CLI;
2. builds `~/Applications/DRPO Update.app` locally with `osacompile`;
3. preserves the native `osacompile` droplet executable and runtime metadata;
4. registers `.drpoupdate` as a ZIP-backed DRPO update package type and sets
   `DRPO Update.app` as its default LaunchServices handler.

The app is generated locally and ad-hoc signed instead of being shipped as a
downloaded executable bundle. Future `.drpoupdate` files are data files, so this
avoids the normal downloaded-script Gatekeeper path used by `.command` launchers.

## Daily use

Canonical packages remain ordinary ZIP archives. The producer may simply use a
`.drpoupdate` output filename:

```bash
python3 scripts/package_update.py \
  --repo . \
  --package-root /path/to/staging \
  --output ~/Downloads/DRPO_E7_FIX.drpoupdate
```

Then double-click `DRPO_E7_FIX.drpoupdate` in Finder. The launcher opens Terminal,
runs:

```text
drpo-update <package> --yes
```

Repository branch, dirty-worktree, local/remote synchronization, and package
base failures are printed directly in that Terminal window with
`DRPO_UPDATE_PREFLIGHT_FAILED`, concrete values, affected files, and repair
commands. The diagnostic ZIP remains available for deeper audit but is not the
only place where the preflight reason appears.

and reports the resulting commit on success. On failure it prints the launcher
log and the new diagnostic ZIP path when one was produced. Diagnostics are first
written to a private per-run staging directory and then published to `Downloads`,
so same-second filesystem timestamp resolution cannot hide a newly created ZIP.
An existing same-named diagnostic is preserved rather than overwritten.
If the canonical helper cannot create a diagnostic, the launcher publishes a
small fallback ZIP containing the launcher log and repository state.

Because the app delegates to the CLI without disabling its defaults, a
successful push also publishes the versioned and `DRPO_MAIN_LATEST` bundle plus
their SHA-256 files in `Downloads`. A post-push export failure remains a failure
in the app log and diagnostic, but it never rolls back the already pushed commit.

The launcher also accepts `.zip` when invoked explicitly, but `.drpoupdate` is
the registered double-click format.

## Behavior and boundaries

- Repository discovery uses `~/.config/drpo-update/repo_path`, the same path
  written by the canonical CLI installer.
- Only one launcher-run update may execute at a time.
- Paths containing spaces or non-ASCII characters are quoted safely.
- The launcher keeps the Terminal session visible; it does not hide update
  output or auto-approve any additional review prompt beyond the existing
  `--yes` behavior.
- There is no auto-update, background daemon, remote download, or alternate Git
  integration implementation.

## Remove the app

```bash
bash tools/drpo-update-macos/uninstall.sh
```

This removes only the app and file association. It leaves the CLI and repository
configuration intact.
