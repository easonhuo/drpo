#!/usr/bin/env python3
"""Internal bundle builder used by the canonical package producer."""
from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import tempfile
import uuid
from pathlib import Path

FULL_SHA_RE = re.compile(r"^[0-9a-f]{40}$")


class BuildError(RuntimeError):
    pass


def run(
    cmd: list[str],
    *,
    cwd: Path | None = None,
    check: bool = True,
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    proc = subprocess.run(
        cmd,
        cwd=cwd,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env,
        check=False,
    )
    if check and proc.returncode != 0:
        detail = (proc.stderr or proc.stdout).strip()
        raise BuildError(
            f"command failed ({proc.returncode}): {' '.join(cmd)}\n{detail}"
        )
    return proc


def git(
    repo: Path,
    *args: str,
    check: bool = True,
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    return run(["git", "-C", str(repo), *args], check=check, env=env)


def git_text(repo: Path, *args: str, env: dict[str, str] | None = None) -> str:
    return git(repo, *args, env=env).stdout.strip()


def package_patch(root: Path) -> Path:
    patches = [
        path
        for path in root.iterdir()
        if path.is_file() and path.suffix in {".patch", ".diff"}
    ]
    if len(patches) != 1:
        raise BuildError(
            f"expected exactly one .patch/.diff file, found {len(patches)}"
        )
    if patches[0].stat().st_size == 0:
        raise BuildError("update patch is empty")
    return patches[0]


def summary_title(path: Path) -> str:
    for line in path.read_text(errors="replace").splitlines():
        if line.startswith("#") and line.lstrip("#").strip():
            return line.lstrip("#").strip()
    return "Apply ChatGPT-reviewed DRPO update"


def build(repo: Path, root: Path, message: str | None) -> str:
    """Create a one-parent patch commit and thin bundle from an isolated index."""

    repo = Path(git_text(repo, "rev-parse", "--show-toplevel"))
    base_file = root / "BASE_COMMIT.txt"
    summary_file = root / "CHANGE_SUMMARY.md"
    if not base_file.is_file() or not summary_file.is_file():
        raise BuildError("BASE_COMMIT.txt and CHANGE_SUMMARY.md are required")
    base = base_file.read_text().strip()
    if not FULL_SHA_RE.fullmatch(base):
        raise BuildError("BASE_COMMIT.txt must contain one full lowercase SHA")
    resolved = git(repo, "rev-parse", f"{base}^{{commit}}", check=False)
    if resolved.returncode != 0 or resolved.stdout.strip() != base:
        raise BuildError(f"base commit is not available in repository: {base}")

    patch = package_patch(root)
    title = message or summary_title(summary_file)
    temp = Path(tempfile.mkdtemp(prefix="drpo-bundle-build-"))
    index_file = temp / "candidate.index"
    ref = f"refs/drpo-update/package-{uuid.uuid4().hex}"
    index_env = os.environ.copy()
    index_env["GIT_INDEX_FILE"] = str(index_file)
    try:
        git(repo, "read-tree", base, env=index_env)
        apply = git(
            repo,
            "apply",
            "--cached",
            str(patch),
            check=False,
            env=index_env,
        )
        if apply.returncode != 0:
            detail = (apply.stderr or apply.stdout).strip()
            raise BuildError(f"patch does not apply to base:\n{detail}")
        tree = git_text(repo, "write-tree", "--missing-ok", env=index_env)
        patch_commit = git_text(
            repo,
            "-c",
            "user.name=DRPO Update Builder",
            "-c",
            "user.email=drpo-update-builder@local.invalid",
            "commit-tree",
            tree,
            "-p",
            base,
            "-m",
            title,
        )
        parents = git_text(repo, "show", "-s", "--format=%P", patch_commit).split()
        if parents != [base]:
            raise BuildError(f"created patch commit has unexpected parents: {parents}")

        git(repo, "update-ref", ref, patch_commit)
        bundle = root / "change.bundle"
        commit_file = root / "PATCH_COMMIT.txt"
        bundle.unlink(missing_ok=True)
        git(repo, "bundle", "create", str(bundle), ref, f"^{base}")
        commit_file.write_text(patch_commit + "\n")
        verify = git(repo, "bundle", "verify", str(bundle), check=False)
        if verify.returncode != 0:
            detail = (verify.stderr or verify.stdout).strip()
            raise BuildError(f"created bundle failed verification:\n{detail}")
        return patch_commit
    finally:
        git(repo, "update-ref", "-d", ref, check=False)
        shutil.rmtree(temp, ignore_errors=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--package-root", type=Path, required=True)
    parser.add_argument("--message")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        commit = build(args.repo.resolve(), args.package_root.resolve(), args.message)
    except BuildError as exc:
        print(f"ERROR: {exc}", file=os.sys.stderr)
        return 2
    print(f"Created change.bundle\nPATCH_COMMIT={commit}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
