from __future__ import annotations

import os
import plistlib
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
TOOL_DIR = REPO_ROOT / "tools" / "drpo-update-macos"


def run(
    cmd: list[str],
    *,
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    proc = subprocess.run(
        cmd,
        cwd=cwd,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if check and proc.returncode != 0:
        raise AssertionError(
            f"command failed: {' '.join(cmd)}\nstdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
        )
    return proc


def init_repo(path: Path) -> str:
    path.mkdir(parents=True)
    run(["git", "init", "-q", str(path)])
    run(["git", "-C", str(path), "config", "user.name", "Launcher Test"])
    run(["git", "-C", str(path), "config", "user.email", "launcher@test.invalid"])
    (path / "README.md").write_text("test\n", encoding="utf-8")
    run(["git", "-C", str(path), "add", "README.md"])
    run(["git", "-C", str(path), "commit", "-q", "-m", "base"])
    return run(["git", "-C", str(path), "rev-parse", "HEAD"]).stdout.strip()


def test_launcher_shell_scripts_pass_bash_syntax() -> None:
    run(
        [
            "bash",
            "-n",
            str(TOOL_DIR / "run_update.sh"),
            str(TOOL_DIR / "install.sh"),
            str(TOOL_DIR / "uninstall.sh"),
        ]
    )


def test_info_plist_registers_zip_backed_drpoupdate_type() -> None:
    with (TOOL_DIR / "Info.plist").open("rb") as handle:
        payload = plistlib.load(handle)

    assert payload["CFBundleIdentifier"] == "com.easonhuo.drpo-update"
    assert "CFBundleExecutable" not in payload
    assert "CFBundleSignature" not in payload
    document_type = payload["CFBundleDocumentTypes"][0]
    assert document_type["CFBundleTypeExtensions"] == ["drpoupdate"]
    assert document_type["LSItemContentTypes"] == ["com.easonhuo.drpoupdate"]
    exported = payload["UTExportedTypeDeclarations"][0]
    assert exported["UTTypeIdentifier"] == "com.easonhuo.drpoupdate"
    assert "public.zip-archive" in exported["UTTypeConformsTo"]


def test_applescript_handles_open_events_and_uses_terminal_runner() -> None:
    source = (TOOL_DIR / "launcher.applescript").read_text(encoding="utf-8")
    assert "on open droppedItems" in source
    assert 'tell application "Terminal"' in source
    assert "Contents/Resources/run_update.sh" in source
    assert "quoted form of packagePath" in source


def test_installer_dry_run_is_available_off_macos(tmp_path: Path) -> None:
    home = tmp_path / "home"
    home.mkdir()
    destination = home / "Applications/DRPO Update.app"
    env = os.environ.copy()
    env.update(
        {
            "HOME": str(home),
            "DRPO_MACOS_ALLOW_NON_DARWIN": "1",
            "DRPO_MACOS_SKIP_CLI_INSTALL": "1",
        }
    )
    proc = run(
        [
            "bash",
            str(TOOL_DIR / "install.sh"),
            "--dry-run",
            "--destination",
            str(destination),
        ],
        env=env,
    )
    assert "DRPO Update.app installation plan" in proc.stdout
    assert str(destination) in proc.stdout
    assert ".drpoupdate -> com.easonhuo.drpo-update" in proc.stdout
    assert not destination.exists()


def test_installer_builds_and_registers_app_with_fake_macos_tools(
    tmp_path: Path,
) -> None:
    home = tmp_path / "home"
    home.mkdir()
    destination = home / "Applications/DRPO Update.app"
    calls = tmp_path / "calls.log"

    fake_osacompile = tmp_path / "osacompile"
    fake_osacompile.write_text(
        """#!/usr/bin/env bash
set -euo pipefail
[[ "$1" == "-o" ]]
destination="$2"
source_file="$3"
mkdir -p "$destination/Contents/MacOS" "$destination/Contents/Resources/Scripts"
printf '#!/usr/bin/env bash\\nexit 0\\n' > "$destination/Contents/MacOS/Compiled Applet"
chmod +x "$destination/Contents/MacOS/Compiled Applet"
cat > "$destination/Contents/Info.plist" <<'PLIST'
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>CFBundleExecutable</key>
  <string>Compiled Applet</string>
</dict>
</plist>
PLIST
printf 'compiled from %s\\n' "$source_file" > "$destination/Contents/Resources/Scripts/main.scpt"
""",
        encoding="utf-8",
    )
    fake_osacompile.chmod(0o755)

    fake_lsregister = tmp_path / "lsregister"
    fake_lsregister.write_text(
        f"#!/usr/bin/env bash\nprintf '%s\\n' \"$*\" >> {str(calls)!r}\n",
        encoding="utf-8",
    )
    fake_lsregister.chmod(0o755)

    fake_codesign = tmp_path / "codesign"
    fake_codesign.write_text(
        f"#!/usr/bin/env bash\nprintf 'codesign %s\\n' \"$*\" >> {str(calls)!r}\n",
        encoding="utf-8",
    )
    fake_codesign.chmod(0o755)

    fake_osascript = tmp_path / "osascript"
    fake_osascript.write_text(
        f"#!/usr/bin/env bash\nprintf 'osascript %s\\n' \"$*\" >> {str(calls)!r}\n",
        encoding="utf-8",
    )
    fake_osascript.chmod(0o755)

    env = os.environ.copy()
    env.update(
        {
            "HOME": str(home),
            "DRPO_MACOS_ALLOW_NON_DARWIN": "1",
            "DRPO_MACOS_SKIP_CLI_INSTALL": "1",
            "DRPO_MACOS_OSACOMPILE": str(fake_osacompile),
            "DRPO_MACOS_LSREGISTER": str(fake_lsregister),
            "DRPO_MACOS_CODESIGN": str(fake_codesign),
            "DRPO_MACOS_OSASCRIPT": str(fake_osascript),
        }
    )
    proc = run(
        [
            "bash",
            str(TOOL_DIR / "install.sh"),
            "--destination",
            str(destination),
            "--skip-cli-install",
        ],
        env=env,
    )
    assert "Installed DRPO Update.app successfully" in proc.stdout
    assert (destination / "Contents/MacOS/Compiled Applet").is_file()
    runner = destination / "Contents/Resources/run_update.sh"
    assert os.access(runner, os.X_OK)
    with (destination / "Contents/Info.plist").open("rb") as handle:
        payload = plistlib.load(handle)
    assert payload["CFBundleIdentifier"] == "com.easonhuo.drpo-update"
    assert payload["CFBundleExecutable"] == "Compiled Applet"
    call_text = calls.read_text(encoding="utf-8")
    assert f"codesign --force --deep --sign - {destination.parent}/.drpo-update-app." in call_text
    assert f"-f {destination}" in call_text
    assert "LSSetDefaultRoleHandlerForContentType" in call_text


def test_runner_success_path_handles_spaces_and_reports_commit(tmp_path: Path) -> None:
    home = tmp_path / "home with spaces"
    home.mkdir()
    repo = tmp_path / "repo with spaces"
    head = init_repo(repo)
    package = tmp_path / "更新 包.drpoupdate"
    package.write_bytes(b"not inspected by fake updater")
    config_dir = home / ".config/drpo-update"
    config_dir.mkdir(parents=True)
    (config_dir / "repo_path").write_text(str(repo) + "\n", encoding="utf-8")

    recorded = tmp_path / "updater-args.txt"
    forwarded_payload = tmp_path / "forwarded-package.bin"
    fake_updater = tmp_path / "drpo-update"
    fake_updater.write_text(
        f"#!/usr/bin/env bash\nprintf '%s\\n' \"$@\" > {str(recorded)!r}\n"
        f'cp "$1" {str(forwarded_payload)!r}\necho updater-success\n',
        encoding="utf-8",
    )
    fake_updater.chmod(0o755)

    env = os.environ.copy()
    env.update(
        {
            "HOME": str(home),
            "DRPO_LAUNCHER_UPDATER": str(fake_updater),
            "DRPO_LAUNCHER_NO_NOTIFY": "1",
        }
    )
    proc = run(["bash", str(TOOL_DIR / "run_update.sh"), str(package)], env=env)
    assert "Update completed successfully" in proc.stdout
    assert head in proc.stdout
    args = recorded.read_text(encoding="utf-8").splitlines()
    forwarded = Path(args[0])
    assert forwarded.suffix == ".zip"
    assert forwarded.parent == config_dir
    assert not forwarded.exists()
    assert forwarded_payload.read_bytes() == package.read_bytes()
    assert "--yes" in args
    assert "--diagnostic-dir" in args
    assert "--no-export-main-bundle" not in args
    logs = list((config_dir / "launcher-logs").glob("*.log"))
    assert len(logs) == 1
    assert "updater-success" in logs[0].read_text(encoding="utf-8")
    assert "Update completed successfully" in logs[0].read_text(encoding="utf-8")


def test_runner_failure_reports_new_diagnostic_and_preserves_exit_code(
    tmp_path: Path,
) -> None:
    home = tmp_path / "home"
    home.mkdir()
    repo = tmp_path / "repo"
    init_repo(repo)
    package = tmp_path / "failure.drpoupdate"
    package.write_bytes(b"fake")
    config_dir = home / ".config/drpo-update"
    config_dir.mkdir(parents=True)
    (config_dir / "repo_path").write_text(str(repo) + "\n", encoding="utf-8")
    downloads = home / "Downloads"
    downloads.mkdir()

    diagnostic_dir_capture = tmp_path / "diagnostic_dir.txt"
    fake_updater = tmp_path / "drpo-update"
    fake_updater.write_text(
        """#!/usr/bin/env bash
set -u
diagnostic_dir=""
while [[ $# -gt 0 ]]; do
  if [[ "$1" == "--diagnostic-dir" ]]; then
    diagnostic_dir="$2"
    shift 2
  else
    shift
  fi
done
printf '%s\n' "$diagnostic_dir" > "$DRPO_TEST_CAPTURE"
touch "$diagnostic_dir/DRPO_DIAGNOSTIC_test.zip"
echo updater-failed >&2
exit 7
""",
        encoding="utf-8",
    )
    fake_updater.chmod(0o755)

    env = os.environ.copy()
    env.update(
        {
            "HOME": str(home),
            "DRPO_LAUNCHER_UPDATER": str(fake_updater),
            "DRPO_LAUNCHER_NO_NOTIFY": "1",
            "DRPO_TEST_CAPTURE": str(diagnostic_dir_capture),
        }
    )
    proc = run(
        ["bash", str(TOOL_DIR / "run_update.sh"), str(package)],
        env=env,
        check=False,
    )
    assert proc.returncode == 7
    assert "Update failed with exit code 7" in proc.stderr
    assert "DRPO_DIAGNOSTIC_test.zip" in proc.stderr
    staging_dir = Path(diagnostic_dir_capture.read_text(encoding="utf-8").strip())
    assert staging_dir.parent == config_dir
    assert staging_dir != downloads
    assert not staging_dir.exists()
    assert (downloads / "DRPO_DIAGNOSTIC_test.zip").is_file()
    log = next((config_dir / "launcher-logs").glob("*.log"))
    assert "Update failed with exit code 7" in log.read_text(encoding="utf-8")


def test_runner_preserves_existing_same_named_diagnostic(tmp_path: Path) -> None:
    home = tmp_path / "home"
    home.mkdir()
    repo = tmp_path / "repo"
    init_repo(repo)
    package = tmp_path / "failure.drpoupdate"
    package.write_bytes(b"fake")
    config_dir = home / ".config/drpo-update"
    config_dir.mkdir(parents=True)
    (config_dir / "repo_path").write_text(str(repo) + "\n", encoding="utf-8")
    downloads = home / "Downloads"
    downloads.mkdir()
    existing = downloads / "DRPO_DIAGNOSTIC_test.zip"
    existing.write_bytes(b"old")

    fake_updater = tmp_path / "drpo-update"
    fake_updater.write_text(
        """#!/usr/bin/env bash
set -u
diagnostic_dir=""
while [[ $# -gt 0 ]]; do
  if [[ "$1" == "--diagnostic-dir" ]]; then
    diagnostic_dir="$2"
    shift 2
  else
    shift
  fi
done
printf 'new' > "$diagnostic_dir/DRPO_DIAGNOSTIC_test.zip"
exit 9
""",
        encoding="utf-8",
    )
    fake_updater.chmod(0o755)

    env = os.environ.copy()
    env.update(
        {
            "HOME": str(home),
            "DRPO_LAUNCHER_UPDATER": str(fake_updater),
            "DRPO_LAUNCHER_NO_NOTIFY": "1",
        }
    )
    proc = run(
        ["bash", str(TOOL_DIR / "run_update.sh"), str(package)],
        env=env,
        check=False,
    )
    assert proc.returncode == 9
    assert existing.read_bytes() == b"old"
    published = [path for path in downloads.glob("DRPO_DIAGNOSTIC_test*.zip") if path != existing]
    assert len(published) == 1
    assert published[0].read_bytes() == b"new"
    assert published[0].name in proc.stderr


def test_runner_creates_launcher_diagnostic_when_updater_creates_none(
    tmp_path: Path,
) -> None:
    home = tmp_path / "home"
    home.mkdir()
    repo = tmp_path / "repo"
    head = init_repo(repo)
    package = tmp_path / "failure.drpoupdate"
    package.write_bytes(b"fake")
    config_dir = home / ".config/drpo-update"
    config_dir.mkdir(parents=True)
    (config_dir / "repo_path").write_text(str(repo) + "\n", encoding="utf-8")
    downloads = home / "Downloads"
    downloads.mkdir()

    fake_updater = tmp_path / "drpo-update"
    fake_updater.write_text(
        "#!/usr/bin/env bash\necho updater-failed-without-diagnostic >&2\nexit 11\n",
        encoding="utf-8",
    )
    fake_updater.chmod(0o755)

    env = os.environ.copy()
    env.update(
        {
            "HOME": str(home),
            "DRPO_LAUNCHER_UPDATER": str(fake_updater),
            "DRPO_LAUNCHER_NO_NOTIFY": "1",
        }
    )
    proc = run(
        ["bash", str(TOOL_DIR / "run_update.sh"), str(package)],
        env=env,
        check=False,
    )
    assert proc.returncode == 11
    diagnostics = list(downloads.glob("DRPO_MACOS_LAUNCHER_DIAGNOSTIC_*.zip"))
    assert len(diagnostics) == 1
    assert diagnostics[0].name in proc.stderr
    import zipfile

    with zipfile.ZipFile(diagnostics[0]) as archive:
        assert set(archive.namelist()) == {
            "README.txt",
            "launcher.log",
            "launcher_state.txt",
        }
        assert "updater-failed-without-diagnostic" in archive.read("launcher.log").decode()
        state = archive.read("launcher_state.txt").decode()
    assert f"head={head}" in state


def test_runner_rejects_concurrent_launch(tmp_path: Path) -> None:
    home = tmp_path / "home"
    lock_dir = home / ".config/drpo-update/launcher.lock"
    lock_dir.mkdir(parents=True)
    package = tmp_path / "update.drpoupdate"
    package.write_bytes(b"fake")
    env = os.environ.copy()
    env.update({"HOME": str(home), "DRPO_LAUNCHER_NO_NOTIFY": "1"})
    proc = run(
        ["bash", str(TOOL_DIR / "run_update.sh"), str(package)],
        env=env,
        check=False,
    )
    assert proc.returncode == 75
    assert "Another DRPO update is already running" in proc.stderr


def test_uninstaller_removes_only_app(tmp_path: Path) -> None:
    home = tmp_path / "home"
    destination = home / "Applications/DRPO Update.app"
    destination.mkdir(parents=True)
    cli = home / "bin/drpo-update"
    cli.parent.mkdir(parents=True)
    cli.write_text("keep\n", encoding="utf-8")

    fake_lsregister = tmp_path / "lsregister"
    fake_lsregister.write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")
    fake_lsregister.chmod(0o755)
    env = os.environ.copy()
    env.update(
        {
            "HOME": str(home),
            "DRPO_MACOS_APP_DEST": str(destination),
            "DRPO_MACOS_LSREGISTER": str(fake_lsregister),
        }
    )
    proc = run(["bash", str(TOOL_DIR / "uninstall.sh")], env=env)
    assert "Removed" in proc.stdout
    assert not destination.exists()
    assert cli.is_file()
