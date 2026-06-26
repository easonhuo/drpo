#!/usr/bin/env python3
"""Transactional local applicator for DRPO update packages.

The helper preserves the legacy patch-only contract and adds an optional Git
bundle path. Bundle-backed packages can be integrated on top of a newer main
when their original base remains an ancestor and Git can cherry-pick the patch
commit without conflicts. All integration and tests happen in an isolated
worktree; the user's main branch moves only after successful verification.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import stat
import subprocess
import sys
import tempfile
import time
import uuid
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

VERSION = "2.0.0"
EXPECTED_REMOTE_FRAGMENTS = ("github.com/easonhuo/drpo", "github.com:easonhuo/drpo")
FULL_SHA_RE = re.compile(r"^[0-9a-f]{40}$")


class UpdateError(RuntimeError):
    """Expected validation or integration failure."""


@dataclass
class Package:
    source: Path
    extracted_root: Path
    base_file: Path
    summary_file: Path
    patch_file: Path
    test_file: Path | None
    bundle_file: Path | None
    patch_commit_file: Path | None

    @property
    def has_git_bundle(self) -> bool:
        return self.bundle_file is not None


@dataclass
class ApplyReport:
    package: str
    repository: str
    package_base: str = ""
    head_before: str = ""
    patch_commit: str | None = None
    integration_mode: str = ""
    tests: list[str] = field(default_factory=list)
    conflicts: list[str] = field(default_factory=list)
    integrated_commit: str | None = None
    head_after: str | None = None
    pushed: bool = False
    status: str = "started"
    error: str | None = None
    created_at_unix: int = field(default_factory=lambda: int(time.time()))

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "tool_version": VERSION,
            **self.__dict__,
        }


def run(
    cmd: Sequence[str],
    *,
    cwd: Path | None = None,
    check: bool = True,
    capture: bool = True,
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    proc = subprocess.run(
        list(cmd),
        cwd=str(cwd) if cwd else None,
        check=False,
        text=True,
        stdout=subprocess.PIPE if capture else None,
        stderr=subprocess.PIPE if capture else None,
        env=env,
    )
    if check and proc.returncode != 0:
        detail = (proc.stderr or proc.stdout or "").strip()
        raise UpdateError(f"command failed ({proc.returncode}): {' '.join(cmd)}\n{detail}")
    return proc


def git(repo: Path, *args: str, check: bool = True, capture: bool = True) -> subprocess.CompletedProcess[str]:
    return run(("git", "-C", str(repo), *args), check=check, capture=capture)


def git_text(repo: Path, *args: str) -> str:
    return git(repo, *args).stdout.strip()


def resolve_repo(cli_repo: str | None) -> Path:
    if cli_repo:
        candidate = Path(cli_repo).expanduser().resolve()
    elif os.environ.get("DRPO_UPDATE_REPO"):
        candidate = Path(os.environ["DRPO_UPDATE_REPO"]).expanduser().resolve()
    else:
        cfg = Path.home() / ".config" / "drpo-update" / "repo_path"
        if not cfg.is_file():
            raise UpdateError(
                "helper is not configured; run tools/drpo-update/install.sh from the local repository"
            )
        candidate = Path(cfg.read_text().strip()).expanduser().resolve()
    git(candidate, "rev-parse", "--show-toplevel")
    return Path(git_text(candidate, "rev-parse", "--show-toplevel")).resolve()


def _safe_zip_member(info: zipfile.ZipInfo) -> None:
    path = Path(info.filename)
    if path.is_absolute() or ".." in path.parts:
        raise UpdateError(f"unsafe ZIP member path: {info.filename}")
    mode = info.external_attr >> 16
    if stat.S_ISLNK(mode):
        raise UpdateError(f"ZIP symlink members are forbidden: {info.filename}")


def _copy_directory(source: Path, destination: Path) -> None:
    for path in source.rglob("*"):
        if path.is_symlink():
            raise UpdateError(f"package directory contains symlink: {path}")
    shutil.copytree(source, destination, dirs_exist_ok=True)


def extract_package(source: Path, temp_root: Path) -> Package:
    source = source.expanduser().resolve()
    if not source.exists():
        raise UpdateError(f"package not found: {source}")
    extract_dir = temp_root / "package"
    extract_dir.mkdir(parents=True)
    if source.is_dir():
        _copy_directory(source, extract_dir)
    elif source.suffix.lower() == ".zip":
        with zipfile.ZipFile(source) as archive:
            for info in archive.infolist():
                _safe_zip_member(info)
            archive.extractall(extract_dir)
    else:
        raise UpdateError("package must be a .zip file or directory")

    base_files = list(extract_dir.rglob("BASE_COMMIT.txt"))
    if len(base_files) != 1:
        raise UpdateError(f"expected exactly one BASE_COMMIT.txt, found {len(base_files)}")
    root = base_files[0].parent
    base_file = root / "BASE_COMMIT.txt"
    summary_file = root / "CHANGE_SUMMARY.md"
    if not summary_file.is_file():
        raise UpdateError("CHANGE_SUMMARY.md is missing")
    patches = [p for p in root.iterdir() if p.is_file() and p.suffix in {".patch", ".diff"}]
    if len(patches) != 1:
        raise UpdateError(f"expected exactly one .patch/.diff file, found {len(patches)}")
    if patches[0].stat().st_size == 0:
        raise UpdateError("update patch is empty")

    test_file = root / "TEST_COMMANDS.sh"
    bundle_file = root / "change.bundle"
    patch_commit_file = root / "PATCH_COMMIT.txt"
    bundle_exists = bundle_file.is_file()
    commit_exists = patch_commit_file.is_file()
    if bundle_exists != commit_exists:
        raise UpdateError("change.bundle and PATCH_COMMIT.txt must appear together")

    return Package(
        source=source,
        extracted_root=root,
        base_file=base_file,
        summary_file=summary_file,
        patch_file=patches[0],
        test_file=test_file if test_file.is_file() else None,
        bundle_file=bundle_file if bundle_exists else None,
        patch_commit_file=patch_commit_file if commit_exists else None,
    )


def read_full_sha(path: Path, label: str) -> str:
    value = path.read_text().strip()
    if not FULL_SHA_RE.fullmatch(value):
        raise UpdateError(f"{label} must contain exactly one full lowercase Git SHA")
    return value


def validate_repo(repo: Path) -> None:
    if git_text(repo, "status", "--porcelain"):
        raise UpdateError("repository has uncommitted changes; commit or stash them first")
    branch = git_text(repo, "branch", "--show-current")
    if branch != "main":
        raise UpdateError(f"expected branch main, found {branch or '<detached>'}")
    remote = git_text(repo, "remote", "get-url", "origin")
    if os.environ.get("DRPO_UPDATE_ALLOW_ANY_REMOTE") != "1":
        if not any(fragment in remote for fragment in EXPECTED_REMOTE_FRAGMENTS):
            raise UpdateError(f"origin is not easonhuo/drpo: {remote}")


def refresh_main(repo: Path) -> None:
    if os.environ.get("DRPO_UPDATE_SKIP_FETCH") == "1":
        return
    print("[1/8] Updating local main...")
    git(repo, "fetch", "origin", "main", capture=False)
    git(repo, "merge", "--ff-only", "origin/main", capture=False)


def resolve_commit(repo: Path, sha: str, label: str) -> str:
    proc = git(repo, "rev-parse", f"{sha}^{{commit}}", check=False)
    if proc.returncode != 0:
        raise UpdateError(f"{label} is not available as a commit in the local repository: {sha}")
    return proc.stdout.strip()


def summary_title(summary_file: Path) -> str:
    for line in summary_file.read_text(errors="replace").splitlines():
        stripped = line.lstrip("#").strip() if line.startswith("#") else ""
        if stripped:
            return stripped
    return "Apply ChatGPT-reviewed DRPO update"


def add_detached_worktree(repo: Path, path: Path, commit: str) -> None:
    git(repo, "worktree", "add", "--detach", "--quiet", str(path), commit)


def remove_worktree(repo: Path, path: Path) -> None:
    if path.exists():
        git(repo, "worktree", "remove", "--force", str(path), check=False)


def verify_bundle_and_patch(repo: Path, package: Package, base: str, temp_root: Path) -> str:
    assert package.bundle_file and package.patch_commit_file
    requested = read_full_sha(package.patch_commit_file, "PATCH_COMMIT.txt")
    verify = git(repo, "bundle", "verify", str(package.bundle_file), check=False)
    if verify.returncode != 0:
        raise UpdateError(f"git bundle verify failed:\n{(verify.stderr or verify.stdout).strip()}")

    imported_ref = f"refs/drpo-update/import/{uuid.uuid4().hex}"
    try:
        git(repo, "fetch", "--quiet", str(package.bundle_file), f"{requested}:{imported_ref}")
        patch_commit = resolve_commit(repo, imported_ref, "PATCH_COMMIT.txt")
        if patch_commit != requested:
            raise UpdateError("fetched bundle commit does not match PATCH_COMMIT.txt")
        parents = git_text(repo, "show", "-s", "--format=%P", patch_commit).split()
        if parents != [base]:
            raise UpdateError(
                f"bundle patch commit must have exactly one parent equal to BASE_COMMIT.txt; got {parents}"
            )

        compare_tree = temp_root / "patch-equivalence"
        add_detached_worktree(repo, compare_tree, base)
        try:
            apply = git(compare_tree, "apply", "--index", str(package.patch_file), check=False)
            if apply.returncode != 0:
                raise UpdateError(
                    "update.patch does not apply to BASE_COMMIT.txt while checking bundle equivalence:\n"
                    + (apply.stderr or apply.stdout).strip()
                )
            patch_tree = git_text(compare_tree, "write-tree")
        finally:
            remove_worktree(repo, compare_tree)
        commit_tree = git_text(repo, "rev-parse", f"{patch_commit}^{{tree}}")
        if patch_tree != commit_tree:
            raise UpdateError(
                "change.bundle and update.patch produce different repository trees; refusing ambiguous package"
            )
        return patch_commit
    finally:
        git(repo, "update-ref", "-d", imported_ref, check=False)


def run_tests(worktree: Path, test_file: Path | None, report: ApplyReport) -> None:
    print("[5/8] Running tests in isolated worktree...")
    if test_file:
        report.tests.append("TEST_COMMANDS.sh")
        proc = run(("bash", str(test_file)), cwd=worktree, check=False, capture=False)
        if proc.returncode != 0:
            raise UpdateError(f"TEST_COMMANDS.sh failed with exit code {proc.returncode}")
        return

    pytest = shutil.which("pytest")
    if pytest:
        report.tests.append("pytest -q")
        proc = run((pytest, "-q"), cwd=worktree, check=False, capture=False)
        if proc.returncode != 0:
            raise UpdateError(f"pytest failed with exit code {proc.returncode}")
    else:
        probe = run((sys.executable, "-c", "import pytest"), check=False)
        if probe.returncode == 0:
            report.tests.append("python -m pytest -q")
            proc = run((sys.executable, "-m", "pytest", "-q"), cwd=worktree, check=False, capture=False)
            if proc.returncode != 0:
                raise UpdateError(f"pytest failed with exit code {proc.returncode}")
        else:
            print("WARNING: pytest is not installed; skipping pytest", file=sys.stderr)

    ruff = shutil.which("ruff")
    if ruff:
        report.tests.append("ruff check .")
        proc = run((ruff, "check", "."), cwd=worktree, check=False, capture=False)
        if proc.returncode != 0:
            raise UpdateError(f"ruff failed with exit code {proc.returncode}")
    else:
        print("WARNING: ruff is not installed; skipping ruff", file=sys.stderr)


def conflict_paths(worktree: Path) -> list[str]:
    proc = git(worktree, "diff", "--name-only", "--diff-filter=U", check=False)
    return [line for line in proc.stdout.splitlines() if line.strip()]


def create_integration_commit(
    repo: Path,
    package: Package,
    current: str,
    base: str,
    patch_commit: str | None,
    message: str,
    temp_root: Path,
    report: ApplyReport,
) -> tuple[Path, str, str]:
    worktree = temp_root / "integration"
    branch = f"drpo-update/integration-{uuid.uuid4().hex}"
    git(repo, "worktree", "add", "--quiet", "-b", branch, str(worktree), current)
    try:
        configured_name = git(repo, "config", "--get", "user.name", check=False).stdout.strip()
        configured_email = git(repo, "config", "--get", "user.email", check=False).stdout.strip()
        git(worktree, "config", "user.name", configured_name or "drpo-update")
        git(worktree, "config", "user.email", configured_email or "drpo-update@local.invalid")
        if patch_commit:
            report.integration_mode = "git-bundle-three-way" if current != base else "git-bundle-exact-base"
            proc = git(worktree, "cherry-pick", patch_commit, check=False, capture=True)
            if proc.returncode != 0:
                report.conflicts = conflict_paths(worktree)
                git(worktree, "cherry-pick", "--abort", check=False)
                detail = (proc.stderr or proc.stdout).strip()
                raise UpdateError(f"three-way cherry-pick failed; main was not modified\n{detail}")
            git(worktree, "commit", "--amend", "--reset-author", "-m", message, capture=False)
        else:
            if current != base:
                raise UpdateError(
                    "bundle base is stale and the package has no change.bundle; regenerate once or use a bundle-backed package"
                )
            report.integration_mode = "legacy-patch-exact-base"
            proc = git(worktree, "apply", "--index", str(package.patch_file), check=False)
            if proc.returncode != 0:
                raise UpdateError(f"git apply failed:\n{(proc.stderr or proc.stdout).strip()}")
            git(
                worktree,
                "-c",
                "user.name=drpo-update",
                "-c",
                "user.email=drpo-update@local.invalid",
                "commit",
                "-m",
                message,
                capture=False,
            )
        integrated = git_text(worktree, "rev-parse", "HEAD")
        return worktree, branch, integrated
    except Exception:
        remove_worktree(repo, worktree)
        git(repo, "branch", "-D", branch, check=False)
        raise


def write_report(report: ApplyReport) -> Path:
    configured = os.environ.get("DRPO_UPDATE_REPORT_DIR")
    report_dir = Path(configured).expanduser() if configured else Path.home() / ".config" / "drpo-update" / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    marker = (report.patch_commit or report.package_base or "unknown")[:12]
    path = report_dir / f"{report.created_at_unix}-{marker}-{uuid.uuid4().hex[:8]}.json"
    path.write_text(json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n")
    return path


def display_review(worktree: Path, current: str, integrated: str, package: Package, report: ApplyReport) -> None:
    print("[6/8] Review summary")
    print(f"Original package base: {report.package_base}")
    print(f"Current main before integration: {current}")
    print(f"Integration mode: {report.integration_mode}")
    print(f"Candidate commit: {integrated}")
    print(f"Package: {package.source}\n")
    print("\n".join(package.summary_file.read_text(errors="replace").splitlines()[:120]))
    print()
    stat_proc = git(worktree, "diff", "--stat", f"{current}..{integrated}")
    print(stat_proc.stdout.rstrip())
    git(worktree, "diff", "--check", f"{current}..{integrated}")


def apply_update(args: argparse.Namespace) -> int:
    repo = resolve_repo(args.repo)
    report = ApplyReport(package=str(Path(args.package).expanduser().resolve()), repository=str(repo))
    temp_root = Path(tempfile.mkdtemp(prefix="drpo-update-"))
    package: Package | None = None
    integration_worktree: Path | None = None
    integration_branch: str | None = None
    try:
        package = extract_package(Path(args.package), temp_root)
        validate_repo(repo)
        refresh_main(repo)
        current = git_text(repo, "rev-parse", "HEAD")
        base_requested = read_full_sha(package.base_file, "BASE_COMMIT.txt")
        base = resolve_commit(repo, base_requested, "BASE_COMMIT.txt")
        report.package_base = base
        report.head_before = current

        patch_commit: str | None = None
        if package.has_git_bundle:
            print("[2/8] Verifying Git bundle and patch equivalence...")
            patch_commit = verify_bundle_and_patch(repo, package, base, temp_root)
            report.patch_commit = patch_commit
            ancestry = git(repo, "merge-base", "--is-ancestor", base, current, check=False)
            if ancestry.returncode != 0:
                raise UpdateError(
                    "BASE_COMMIT.txt is not an ancestor of current main; automatic integration is intentionally disabled"
                )
        elif current != base:
            raise UpdateError(
                "bundle base commit does not match current main and no Git bundle is present\n"
                f"  package base: {base}\n  current HEAD: {current}"
            )
        else:
            print("[2/8] Legacy package detected; exact-base patch path will be used.")

        print("[3/8] Preparing isolated integration worktree...")
        message = args.message or summary_title(package.summary_file)
        integration_worktree, integration_branch, integrated = create_integration_commit(
            repo, package, current, base, patch_commit, message, temp_root, report
        )
        report.integrated_commit = integrated
        print("[4/8] Candidate integration created; main is still untouched.")
        run_tests(integration_worktree, package.test_file, report)
        display_review(integration_worktree, current, integrated, package, report)

        if not args.yes:
            answer = input(f"Tests passed. Fast-forward main to '{message}' and push? [y/N] ").strip()
            if answer.lower() != "y":
                report.status = "stopped_before_main_update"
                path = write_report(report)
                print(f"Stopped before changing main. Report: {path}")
                return 0

        print("[7/8] Fast-forwarding verified commit onto main...")
        git(repo, "merge", "--ff-only", integrated, capture=False)
        report.head_after = git_text(repo, "rev-parse", "HEAD")
        report.status = "committed_local"

        if args.no_push:
            print("[8/8] Push skipped (--no-push).")
        else:
            print("[8/8] Pushing origin/main...")
            push = git(repo, "push", "origin", "main", check=False, capture=False)
            if push.returncode != 0:
                report.status = "committed_local_push_failed"
                raise UpdateError("push failed; verified commit remains on local main")
            report.pushed = True
            report.status = "success"
        if args.no_push:
            report.status = "success_no_push"
        path = write_report(report)
        print(f"Done.\nCommit: {report.head_after}\nApply report: {path}")
        return 0
    except (UpdateError, OSError, zipfile.BadZipFile) as exc:
        report.error = str(exc)
        if report.status not in {"committed_local_push_failed"}:
            report.status = "failed"
        try:
            path = write_report(report)
            print(f"ERROR: {exc}\nApply report: {path}", file=sys.stderr)
        except Exception:
            print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    finally:
        if integration_worktree is not None:
            remove_worktree(repo, integration_worktree)
        if integration_branch is not None:
            git(repo, "branch", "-D", integration_branch, check=False)
        shutil.rmtree(temp_root, ignore_errors=True)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="drpo-update")
    parser.add_argument("package", nargs="?")
    parser.add_argument("--yes", "-y", action="store_true")
    parser.add_argument("--no-push", action="store_true")
    parser.add_argument("--message", "-m")
    parser.add_argument("--repo", help=argparse.SUPPRESS)
    parser.add_argument("--version", action="store_true")
    args = parser.parse_args(argv)
    if args.version:
        print(f"drpo-update {VERSION}")
        raise SystemExit(0)
    if not args.package:
        parser.error("the following arguments are required: package")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    return apply_update(parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
