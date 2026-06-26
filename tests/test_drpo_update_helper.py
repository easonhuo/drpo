from __future__ import annotations

import os
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
HELPER_DIR = REPO_ROOT / "tools" / "drpo-update"


def run(cmd: list[str], *, cwd: Path | None = None, env: dict[str, str] | None = None, check: bool = True):
    proc = subprocess.run(cmd, cwd=cwd, env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if check and proc.returncode != 0:
        raise AssertionError(f"command failed: {' '.join(cmd)}\n{proc.stdout}\n{proc.stderr}")
    return proc


def test_helper_reports_version():
    proc = run([str(HELPER_DIR / "drpo-update"), "--version"])
    assert proc.stdout.strip() == "drpo-update 2.0.0"


def test_installer_defaults_to_repository_symlink(tmp_path: Path):
    origin = tmp_path / "origin.git"
    repo = tmp_path / "repo"
    run(["git", "init", "--bare", str(origin)])
    run(["git", "init", str(repo)])
    run(["git", "-C", str(repo), "remote", "add", "origin", str(origin)])
    target_dir = repo / "tools" / "drpo-update"
    target_dir.mkdir(parents=True)
    for name in ("drpo-update", "drpo_update.py", "install.sh"):
        source = HELPER_DIR / name
        destination = target_dir / name
        destination.write_bytes(source.read_bytes())
        os.chmod(destination, source.stat().st_mode)

    home = tmp_path / "home"
    home.mkdir()
    env = os.environ.copy()
    env.update({"HOME": str(home), "DRPO_UPDATE_ALLOW_ANY_REMOTE": "1", "SHELL": "/bin/zsh"})
    proc = run(["bash", str(target_dir / "install.sh"), str(repo)], env=env)
    installed = home / "bin" / "drpo-update"
    assert installed.is_symlink()
    assert installed.resolve() == (target_dir / "drpo-update").resolve()
    assert (home / ".config" / "drpo-update" / "repo_path").read_text().strip() == str(repo.resolve())
    assert "drpo-update 2.0.0" in proc.stdout


def test_provenance_records_user_verified_original_hash():
    text = (HELPER_DIR / "SOURCE_PROVENANCE.md").read_text()
    assert "f344b0cffc163ecdeb80ec8d07b564d00c3538ad22e03887f87bd1ce2a85f4f3" in text
    assert "byte-identical" in text
