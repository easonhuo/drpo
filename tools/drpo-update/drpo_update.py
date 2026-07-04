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
import hashlib
import json
import os
import platform
import re
import shutil
import stat
import subprocess
import sys
import tempfile
import time
import uuid
import zipfile
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

TOOL_DIR = Path(__file__).resolve().parent
if str(TOOL_DIR) not in sys.path:
    sys.path.insert(0, str(TOOL_DIR))

from test_selection import (  # noqa: E402
    CommandOutcome,
    TestExecutionError,
    TestSelectionError,
    execute_test_plan,
    select_test_plan,
)

VERSION = "2.4.0"
EXPECTED_REMOTE_FRAGMENTS = ("github.com/easonhuo/drpo", "github.com:easonhuo/drpo")
FULL_SHA_RE = re.compile(r"^[0-9a-f]{40}$")
RECOVERY_ARTIFACT_KINDS = {
    "experiment-checkpoint",
    "experiment-failed",
    "experiment-raw-complete",
}
APPLICABLE_ARTIFACT_KINDS = {"governance", "experiment-final"}


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
    source_integrated_commit_pre_normalization: str | None = None
    integrated_commit: str | None = None
    handoff_normalization: dict[str, object] | None = None
    head_after: str | None = None
    pushed: bool = False
    remote_head_after_push: str | None = None
    main_bundle_exported: bool = False
    main_bundle_path: str | None = None
    main_bundle_latest_path: str | None = None
    main_bundle_sha256: str | None = None
    main_bundle_export_skipped: str | None = None
    status: str = "started"
    error: str | None = None
    requested_test_mode: str = "auto"
    selected_test_mode: str | None = None
    test_selection: dict[str, object] | None = None
    test_commands: list[dict[str, object]] = field(default_factory=list)
    diagnostic_zip: str | None = None
    diagnostic_error: str | None = None
    failure_phase: str | None = None
    timings_seconds: dict[str, float] = field(default_factory=dict)
    created_at_unix: int = field(default_factory=lambda: int(time.time()))

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": 4,
            "tool_version": VERSION,
            **self.__dict__,
        }


@contextmanager
def timed_phase(report: ApplyReport, name: str):
    started = time.perf_counter()
    try:
        yield
    finally:
        report.timings_seconds[name] = round(time.perf_counter() - started, 6)


def finalize_total_timing(report: ApplyReport, started: float) -> None:
    report.timings_seconds["total"] = round(time.perf_counter() - started, 6)


def print_timing_summary(report: ApplyReport) -> None:
    if not report.timings_seconds:
        return
    print("Timing summary:")
    for name, duration in sorted(
        report.timings_seconds.items(),
        key=lambda item: (item[0] == "total", item[0]),
    ):
        print(f"  {name}: {duration:.3f}s")


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


def run_logged_command(
    cmd: Sequence[str],
    *,
    cwd: Path,
    log_path: Path,
) -> CommandOutcome:
    """Run one command with merged stdout/stderr streamed to console and disk."""

    log_path.parent.mkdir(parents=True, exist_ok=True)
    command = tuple(str(item) for item in cmd)
    header = f"$ {' '.join(command)}\n"
    try:
        proc = subprocess.Popen(
            list(command),
            cwd=str(cwd),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
    except OSError as exc:
        text = header + f"command start failed: {exc}\n"
        log_path.write_text(text)
        print(text.rstrip(), file=sys.stderr)
        return CommandOutcome(
            label="TEST_COMMANDS.sh",
            command=command,
            returncode=127,
            log_file=str(log_path),
            error=str(exc),
        )

    with log_path.open("w") as handle:
        handle.write(header)
        assert proc.stdout is not None
        for line in proc.stdout:
            print(line, end="")
            handle.write(line)
        returncode = proc.wait()
        handle.write(f"\n[exit_code] {returncode}\n")
    return CommandOutcome(
        label="TEST_COMMANDS.sh",
        command=command,
        returncode=returncode,
        log_file=str(log_path),
    )


def append_test_outcome(report: ApplyReport, outcome: CommandOutcome) -> None:
    payload = outcome.to_dict()
    if outcome.log_file:
        payload["log_file"] = Path(outcome.log_file).name
    report.test_commands.append(payload)


def logged_command_output(outcome: CommandOutcome) -> str:
    """Return command output captured by ``run_logged_command`` without log framing."""
    if not outcome.log_file:
        return outcome.error or ""
    path = Path(outcome.log_file)
    if not path.is_file():
        return outcome.error or ""
    lines = path.read_text(encoding="utf-8").splitlines()
    if lines and lines[0].startswith("$ "):
        lines = lines[1:]
    if lines and lines[-1].startswith("[exit_code]"):
        lines = lines[:-1]
    while lines and not lines[-1].strip():
        lines.pop()
    return "\n".join(lines)


def git(
    repo: Path,
    *args: str,
    check: bool = True,
    capture: bool = True,
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    return run(
        ("git", "-C", str(repo), *args),
        check=check,
        capture=capture,
        env=env,
    )


def git_text(repo: Path, *args: str, env: dict[str, str] | None = None) -> str:
    return git(repo, *args, env=env).stdout.strip()


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


def read_artifact_package_kind(root: Path) -> str | None:
    """Return a hardened experiment-artifact kind when the manifest is present.

    Recovery artifacts preserve evidence and intentionally may carry an empty
    ``update.patch``. They are not repository updates and must never reach the
    patch-integration path.
    """

    manifest_path = root / "ARTIFACT_MANIFEST.json"
    if not manifest_path.is_file():
        return None
    try:
        payload = json.loads(manifest_path.read_text())
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise UpdateError(f"ARTIFACT_MANIFEST.json is not valid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise UpdateError("ARTIFACT_MANIFEST.json must contain a JSON object")
    package_kind = payload.get("package_kind")
    if not isinstance(package_kind, str) or not package_kind:
        raise UpdateError("ARTIFACT_MANIFEST.json is missing package_kind")
    if package_kind in RECOVERY_ARTIFACT_KINDS:
        raise UpdateError(
            f"{package_kind} is a recovery/evidence package, not a repository update. "
            "Do not pass it to drpo-update. Preserve it for audit, then create and "
            "apply an experiment-final repository-closure package after the "
            "handoff, registry, and compact result files are updated."
        )
    if package_kind not in APPLICABLE_ARTIFACT_KINDS:
        raise UpdateError(
            f"unsupported ARTIFACT_MANIFEST package_kind for drpo-update: {package_kind}"
        )
    return package_kind


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
    read_artifact_package_kind(root)
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
    print("[1/10] Updating local main...")
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

        index_file = temp_root / "patch-equivalence.index"
        index_env = os.environ.copy()
        index_env["GIT_INDEX_FILE"] = str(index_file)
        git(repo, "read-tree", base, env=index_env)
        apply = git(
            repo,
            "apply",
            "--cached",
            str(package.patch_file),
            check=False,
            env=index_env,
        )
        if apply.returncode != 0:
            raise UpdateError(
                "update.patch does not apply to BASE_COMMIT.txt while checking "
                "bundle equivalence:\n"
                + (apply.stderr or apply.stdout).strip()
            )
        patch_tree = git_text(repo, "write-tree", "--missing-ok", env=index_env)
        commit_tree = git_text(repo, "rev-parse", f"{patch_commit}^{{tree}}")
        if patch_tree != commit_tree:
            raise UpdateError(
                "change.bundle and update.patch produce different repository trees; refusing ambiguous package"
            )
        return patch_commit
    finally:
        git(repo, "update-ref", "-d", imported_ref, check=False)


def run_handoff_normalization(
    repo: Path,
    worktree: Path,
    *,
    current: str,
    base: str,
    source_patch_commit: str | None,
    report: ApplyReport,
    log_dir: Path,
) -> str:
    """Run the trusted Stage 5 normalizer and amend generated artifacts atomically.

    The executable and authority policy come from pre-integration current main
    (``repo``), not from the candidate worktree.  In the current manual mode the
    command is a deterministic no-op.  Future delta mode requires a bundle-backed
    source commit so exact-base intent can be checked independently of the
    normalized integration commit.
    """
    authority_file = repo / "docs" / "handoff_versions" / "AUTHORITY.yaml"
    normalizer = repo / "scripts" / "handoff_authority.py"
    if not authority_file.is_file() or not normalizer.is_file():
        report.handoff_normalization = {
            "status": "not_configured",
            "mode": "legacy_manual",
        }
        return git_text(worktree, "rev-parse", "HEAD")

    import yaml

    authority = yaml.safe_load(authority_file.read_text(encoding="utf-8"))
    mode = authority.get("mode") if isinstance(authority, dict) else None
    if mode == "delta" and source_patch_commit is None:
        raise UpdateError(
            "delta-authority update packages must be bundle-backed so exact-base "
            "intent can be verified"
        )
    source_commit = source_patch_commit or git_text(worktree, "rev-parse", "HEAD")
    command = [
        sys.executable,
        str(normalizer),
        "normalize",
        "--repo-root",
        str(worktree),
        "--trusted-repo-root",
        str(repo),
        "--current-before",
        current,
        "--source-base",
        base,
        "--source-patch-commit",
        source_commit,
        "--json",
    ]
    outcome = run_logged_command(
        tuple(command),
        cwd=worktree,
        log_path=log_dir / "handoff-normalization.log",
    )
    append_test_outcome(report, outcome)
    output = logged_command_output(outcome)
    if not outcome.passed:
        raise UpdateError(
            "trusted handoff normalization failed; main was not modified: "
            + output.strip()
        )
    try:
        payload = json.loads(output)
    except json.JSONDecodeError as exc:
        raise UpdateError("trusted handoff normalizer returned invalid JSON") from exc
    if payload.get("status") != "PASS":
        raise UpdateError("trusted handoff normalizer did not return PASS")
    report.handoff_normalization = payload

    status = git(worktree, "status", "--porcelain", check=False).stdout
    if status.strip():
        git(worktree, "add", "-A")
        git(worktree, "commit", "--amend", "--no-edit", capture=False)
    normalized = git_text(worktree, "rev-parse", "HEAD")

    if mode == "delta":
        verify_outcome = run_logged_command(
            (
                sys.executable,
                str(normalizer),
                "verify",
                "--repo-root",
                str(worktree),
                "--json",
            ),
            cwd=worktree,
            log_path=log_dir / "handoff-normalization-verify.log",
        )
        append_test_outcome(report, verify_outcome)
        if not verify_outcome.passed:
            raise UpdateError(
                "normalized handoff state failed deterministic verification; main was not modified"
            )
        verify_output = logged_command_output(verify_outcome)
        try:
            verify_payload = json.loads(verify_output)
        except json.JSONDecodeError as exc:
            raise UpdateError("handoff authority verifier returned invalid JSON") from exc
        report.handoff_normalization["post_amend_verify"] = verify_payload
    return normalized


def run_package_tests(
    worktree: Path,
    test_file: Path | None,
    report: ApplyReport,
    log_dir: Path,
) -> UpdateError | None:
    print("[6/11] Running package tests in isolated worktree...")
    if test_file:
        report.tests.append("TEST_COMMANDS.sh")
        outcome = run_logged_command(
            ("bash", str(test_file)),
            cwd=worktree,
            log_path=log_dir / "package-tests.log",
        )
        append_test_outcome(report, outcome)
        if not outcome.passed:
            return UpdateError(
                f"TEST_COMMANDS.sh failed with exit code {outcome.returncode}"
            )
        return None
    print("WARNING: TEST_COMMANDS.sh is absent; relying on the repository test selector", file=sys.stderr)
    return None


def candidate_changed_paths(worktree: Path, current: str, integrated: str) -> list[str]:
    proc = git(worktree, "diff", "--name-only", f"{current}..{integrated}")
    return [line for line in proc.stdout.splitlines() if line.strip()]


def run_selected_tests(
    repo: Path,
    worktree: Path,
    current: str,
    integrated: str,
    requested_mode: str,
    report: ApplyReport,
    log_dir: Path,
) -> UpdateError | None:
    print("[7/11] Selecting and running repository integration tests...")
    impact_map = repo / "tools" / "drpo-update" / "test_impact_map.json"
    if not impact_map.is_file():
        raise UpdateError(f"trusted test impact map is missing from current main: {impact_map}")
    changed_paths = candidate_changed_paths(worktree, current, integrated)
    try:
        plan = select_test_plan(
            changed_paths,
            impact_map,
            requested_mode=requested_mode,
        )
        try:
            executed = execute_test_plan(
                plan,
                worktree=worktree,
                python_executable=sys.executable,
                log_dir=log_dir,
                outcome_callback=lambda outcome: append_test_outcome(report, outcome),
            )
        except TestExecutionError as exc:
            executed = [outcome.label for outcome in exc.outcomes]
            report.selected_test_mode = plan.selected_mode
            report.test_selection = plan.to_dict()
            report.tests.extend(executed)
            return UpdateError(f"repository test gate failed: {exc}")
    except TestSelectionError as exc:
        return UpdateError(f"repository test selection failed: {exc}")
    report.selected_test_mode = plan.selected_mode
    report.test_selection = plan.to_dict()
    report.tests.extend(executed)
    print(
        "Repository test gate: "
        f"mode={plan.selected_mode}, risk={plan.risk}, "
        f"groups={','.join(plan.matched_groups) or '-'}, "
        f"unknown={len(plan.unknown_paths)}"
    )
    print(f"Selection reason: {plan.reason}")
    return None


def conflict_paths(worktree: Path) -> list[str]:
    proc = git(worktree, "diff", "--name-only", "--diff-filter=U", check=False)
    return [line for line in proc.stdout.splitlines() if line.strip()]


def prepare_integration_worktree(
    repo: Path,
    current: str,
    temp_root: Path,
) -> tuple[Path, str]:
    worktree = temp_root / "integration"
    branch = f"drpo-update/integration-{uuid.uuid4().hex}"
    git(repo, "worktree", "add", "--quiet", "-b", branch, str(worktree), current)
    configured_name = git(repo, "config", "--get", "user.name", check=False).stdout.strip()
    configured_email = git(repo, "config", "--get", "user.email", check=False).stdout.strip()
    git(worktree, "config", "user.name", configured_name or "drpo-update")
    git(worktree, "config", "user.email", configured_email or "drpo-update@local.invalid")
    return worktree, branch


def create_integration_commit(
    worktree: Path,
    package: Package,
    current: str,
    base: str,
    patch_commit: str | None,
    message: str,
    report: ApplyReport,
) -> str:
    if patch_commit:
        report.integration_mode = (
            "git-bundle-three-way" if current != base else "git-bundle-exact-base"
        )
        proc = git(worktree, "cherry-pick", patch_commit, check=False, capture=True)
        if proc.returncode != 0:
            report.conflicts = conflict_paths(worktree)
            detail = (proc.stderr or proc.stdout).strip()
            raise UpdateError(f"three-way cherry-pick failed; main was not modified\n{detail}")
        git(worktree, "commit", "--amend", "--reset-author", "-m", message, capture=False)
    else:
        if current != base:
            raise UpdateError(
                "bundle base is stale and the package has no change.bundle; "
                "regenerate once or use a bundle-backed package"
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
    return git_text(worktree, "rev-parse", "HEAD")


def write_report(report: ApplyReport) -> Path:
    configured = os.environ.get("DRPO_UPDATE_REPORT_DIR")
    report_dir = Path(configured).expanduser() if configured else Path.home() / ".config" / "drpo-update" / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    marker = (report.patch_commit or report.package_base or "unknown")[:12]
    path = report_dir / f"{report.created_at_unix}-{marker}-{uuid.uuid4().hex[:8]}.json"
    path.write_text(json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n")
    return path



def diagnostic_output_path(report: ApplyReport, configured_dir: str | None) -> Path:
    directory = (
        Path(configured_dir).expanduser()
        if configured_dir
        else Path(os.environ.get("DRPO_UPDATE_DIAGNOSTIC_DIR", "")).expanduser()
        if os.environ.get("DRPO_UPDATE_DIAGNOSTIC_DIR")
        else Path.home() / "Downloads"
    )
    directory.mkdir(parents=True, exist_ok=True)
    marker = (report.head_before or report.package_base or "unknown")[:12]
    timestamp = time.strftime("%Y%m%d-%H%M%S", time.localtime())
    return directory / f"DRPO_DIAGNOSTIC_{marker}_{timestamp}_{uuid.uuid4().hex[:8]}.zip"


def main_bundle_output_dir(configured_dir: str | None) -> Path:
    """Resolve the post-push main-bundle directory (Downloads by default)."""

    directory = (
        Path(configured_dir).expanduser()
        if configured_dir
        else Path(os.environ.get("DRPO_UPDATE_MAIN_BUNDLE_DIR", "")).expanduser()
        if os.environ.get("DRPO_UPDATE_MAIN_BUNDLE_DIR")
        else Path.home() / "Downloads"
    )
    directory.mkdir(parents=True, exist_ok=True)
    return directory.resolve()


def remote_main_sha(repo: Path) -> str:
    proc = git(repo, "ls-remote", "origin", "refs/heads/main", check=False)
    if proc.returncode != 0:
        raise UpdateError(
            "could not verify origin/main after push:\n"
            + (proc.stderr or proc.stdout or "git ls-remote failed").strip()
        )
    lines = [line for line in proc.stdout.splitlines() if line.strip()]
    if len(lines) != 1:
        raise UpdateError(f"expected one origin/main ref after push, found {len(lines)}")
    sha, ref = lines[0].split(None, 1)
    if ref != "refs/heads/main" or not FULL_SHA_RE.fullmatch(sha):
        raise UpdateError(f"unexpected origin/main response: {lines[0]}")
    return sha


def _atomic_write_text(path: Path, text: str) -> None:
    temp = path.with_name(f".{path.name}.tmp-{uuid.uuid4().hex}")
    temp.write_text(text)
    os.replace(temp, path)


def _atomic_copy(source: Path, destination: Path) -> None:
    temp = destination.with_name(f".{destination.name}.tmp-{uuid.uuid4().hex}")
    shutil.copy2(source, temp)
    os.replace(temp, destination)


def export_main_bundle(repo: Path, output_dir: Path, head: str) -> dict[str, str]:
    """Atomically export the pushed main ref as versioned and stable bundles."""

    if git_text(repo, "branch", "--show-current") != "main":
        raise UpdateError("main bundle export requires the checked-out main branch")
    if git_text(repo, "rev-parse", "HEAD") != head:
        raise UpdateError("main bundle export HEAD changed unexpectedly")
    output_dir.mkdir(parents=True, exist_ok=True)
    temp_root = Path(tempfile.mkdtemp(prefix="drpo-main-bundle-", dir=output_dir))
    try:
        candidate = temp_root / "candidate.bundle"
        created = git(repo, "bundle", "create", str(candidate), "main", check=False)
        if created.returncode != 0:
            raise UpdateError(
                "git bundle create failed:\n"
                + (created.stderr or created.stdout or "unknown error").strip()
            )
        verified = git(repo, "bundle", "verify", str(candidate), check=False)
        if verified.returncode != 0:
            raise UpdateError(
                "created main bundle failed verification:\n"
                + (verified.stderr or verified.stdout or "unknown error").strip()
            )
        heads = git(repo, "bundle", "list-heads", str(candidate), check=False)
        expected = f"{head} refs/heads/main"
        if heads.returncode != 0 or expected not in heads.stdout.splitlines():
            raise UpdateError(
                "created main bundle does not advertise the verified main ref; "
                f"expected '{expected}'"
            )
        digest = _sha256(candidate)
        versioned = output_dir / f"DRPO_MAIN_{head[:12]}.bundle"
        latest = output_dir / "DRPO_MAIN_LATEST.bundle"
        versioned_sha = versioned.with_name(versioned.name + ".sha256")
        latest_sha = latest.with_name(latest.name + ".sha256")
        _atomic_copy(candidate, versioned)
        _atomic_write_text(versioned_sha, f"{digest}  {versioned.name}\n")
        _atomic_copy(candidate, latest)
        _atomic_write_text(latest_sha, f"{digest}  {latest.name}\n")
        return {
            "versioned": str(versioned),
            "latest": str(latest),
            "sha256": digest,
        }
    finally:
        shutil.rmtree(temp_root, ignore_errors=True)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_git_command(repo: Path, destination: Path, *args: str) -> None:
    proc = git(repo, *args, check=False)
    destination.parent.mkdir(parents=True, exist_ok=True)
    content = (
        f"$ git {' '.join(args)}\n"
        f"[exit_code] {proc.returncode}\n\n"
        f"{proc.stdout or ''}"
        f"{proc.stderr or ''}"
    )
    content = re.sub(r"(https?://)[^/@\s]+@", r"\1<redacted>@", content)
    destination.write_text(content)


def _safe_conflict_path(path: str) -> Path:
    relative = Path(path)
    if relative.is_absolute() or ".." in relative.parts or not relative.parts:
        raise UpdateError(f"unsafe conflict path reported by Git: {path}")
    return relative


def capture_conflict_materials(worktree: Path, destination: Path, paths: Sequence[str]) -> None:
    for path in paths:
        relative = _safe_conflict_path(path)
        conflict_dir = destination / relative
        conflict_dir.mkdir(parents=True, exist_ok=True)
        for stage, label in ((1, "base"), (2, "ours"), (3, "theirs")):
            proc = subprocess.run(
                ["git", "-C", str(worktree), "show", f":{stage}:{path}"],
                check=False,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            if proc.returncode == 0:
                (conflict_dir / label).write_bytes(proc.stdout)
            else:
                (conflict_dir / f"{label}.missing.txt").write_bytes(proc.stderr)
        candidate = worktree / relative
        if candidate.is_file() and not candidate.is_symlink():
            shutil.copy2(candidate, conflict_dir / "worktree")


def _copy_original_package(source: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    try:
        if source.is_file():
            shutil.copy2(source, destination / source.name)
        elif source.is_dir():
            _copy_directory(source, destination / "original_package")
        else:
            raise OSError(f"package path no longer exists: {source}")
    except (OSError, UpdateError) as exc:
        (destination / "COPY_ERROR.txt").write_text(
            f"Original package could not be copied safely: {exc}\n"
        )


def _create_repository_bundle(
    repo: Path,
    destination: Path,
    commits: dict[str, str | None],
) -> str | None:
    namespace = f"refs/drpo-update/diagnostic/{uuid.uuid4().hex}"
    temporary_refs: list[str] = []
    try:
        for label, commit in commits.items():
            if not commit or not FULL_SHA_RE.fullmatch(commit):
                continue
            if git(repo, "cat-file", "-e", f"{commit}^{{commit}}", check=False).returncode != 0:
                continue
            ref = f"{namespace}/{label}"
            git(repo, "update-ref", ref, commit)
            temporary_refs.append(ref)
        proc = git(repo, "bundle", "create", str(destination), "--all", check=False)
        if proc.returncode != 0:
            return (proc.stderr or proc.stdout or "git bundle create failed").strip()
        return None
    finally:
        for ref in temporary_refs:
            git(repo, "update-ref", "-d", ref, check=False)


def create_diagnostic_zip(
    *,
    output_path: Path,
    staging_root: Path,
    report: ApplyReport,
    report_path: Path,
    repo: Path | None,
    package_source: Path,
    package: Package | None,
    integration_worktree: Path | None,
    current: str | None,
    base: str | None,
    patch_commit: str | None,
    integrated: str | None,
    logs_dir: Path,
) -> Path:
    """Create one upload-ready failure ZIP without mutating the user's main branch."""

    if staging_root.exists():
        shutil.rmtree(staging_root)
    staging_root.mkdir(parents=True)
    (staging_root / "README.txt").write_text(
        "DRPO update diagnostic bundle\n"
        "\n"
        "This bundle was generated automatically after drpo-update failed.\n"
        "It contains the original update package, apply report, complete gate logs,\n"
        "a Git repository bundle, candidate metadata, conflict stages when present,\n"
        "and a dependency/environment inventory. Secrets and arbitrary environment\n"
        "variables are intentionally not collected.\n"
    )
    shutil.copy2(report_path, staging_root / "apply_report.json")
    _copy_original_package(package_source, staging_root / "inputs")

    if logs_dir.is_dir():
        shutil.copytree(logs_dir, staging_root / "logs", dirs_exist_ok=True)

    environment = {
        "platform": platform.platform(),
        "python_version": sys.version,
        "python_executable": sys.executable,
        "tool_version": VERSION,
        "home": str(Path.home()),
        "selected_environment": {
            key: os.environ.get(key)
            for key in ("SHELL", "VIRTUAL_ENV", "PYTHONPATH", "PATH")
            if os.environ.get(key) is not None
        },
    }
    environment_dir = staging_root / "environment"
    environment_dir.mkdir()
    (environment_dir / "system.json").write_text(
        json.dumps(environment, indent=2, sort_keys=True) + "\n"
    )
    for name, command in (
        ("git-version.txt", ("git", "--version")),
        ("python-packages.json", (sys.executable, "-m", "pip", "list", "--format=json")),
    ):
        proc = run(command, check=False)
        (environment_dir / name).write_text(
            (proc.stdout or "") + (proc.stderr or "") + f"\n[exit_code] {proc.returncode}\n"
        )

    git_dir = staging_root / "git"
    git_dir.mkdir()
    if repo is not None and git(repo, "rev-parse", "--git-dir", check=False).returncode == 0:
        commits = {
            "current": current,
            "base": base,
            "patch": patch_commit,
            "candidate": integrated,
        }
        bundle_error = _create_repository_bundle(repo, git_dir / "repository.bundle", commits)
        if bundle_error:
            (git_dir / "repository-bundle-error.txt").write_text(bundle_error + "\n")
        _write_git_command(repo, git_dir / "status.txt", "status", "--porcelain=v2", "--branch")
        _write_git_command(repo, git_dir / "refs.txt", "show-ref")
        _write_git_command(repo, git_dir / "log.txt", "log", "--oneline", "--decorate", "-30", "--all")
        _write_git_command(repo, git_dir / "remotes.txt", "remote", "-v")
    else:
        (git_dir / "repository-unavailable.txt").write_text(
            "Repository was unavailable before failure diagnostics were created.\n"
        )

    candidate_dir = staging_root / "candidate"
    candidate_dir.mkdir()
    identity = {
        "current": current,
        "base": base,
        "patch_commit": patch_commit,
        "integrated_commit": integrated,
        "integration_mode": report.integration_mode,
        "conflicts": report.conflicts,
        "package_extracted_root": str(package.extracted_root) if package else None,
    }
    (candidate_dir / "identity.json").write_text(
        json.dumps(identity, indent=2, sort_keys=True) + "\n"
    )
    if package is not None:
        shutil.copy2(package.patch_file, candidate_dir / "package-update.patch")
        if repo is not None:
            patch_paths = git(
                repo,
                "apply",
                "--numstat",
                str(package.patch_file),
                check=False,
            )
            (candidate_dir / "package-patch-paths.txt").write_text(
                patch_paths.stdout + patch_paths.stderr
            )
    if integration_worktree is not None and integration_worktree.exists():
        _write_git_command(
            integration_worktree,
            candidate_dir / "worktree-status.txt",
            "status",
            "--porcelain=v2",
            "--branch",
        )
        if current and integrated:
            _write_git_command(
                integration_worktree,
                candidate_dir / "changed-files.txt",
                "diff",
                "--name-status",
                f"{current}..{integrated}",
            )
            diff_proc = git(
                integration_worktree,
                "diff",
                "--binary",
                f"{current}..{integrated}",
                check=False,
            )
            (candidate_dir / "current-to-candidate.patch").write_text(
                diff_proc.stdout + diff_proc.stderr
            )
        if report.conflicts:
            capture_conflict_materials(
                integration_worktree,
                candidate_dir / "conflicts",
                report.conflicts,
            )

    files = []
    for path in sorted(staging_root.rglob("*")):
        if path.is_file() and path.name != "diagnostic_manifest.json":
            files.append(
                {
                    "path": path.relative_to(staging_root).as_posix(),
                    "size": path.stat().st_size,
                    "sha256": _sha256(path),
                }
            )
    (staging_root / "diagnostic_manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "created_at_unix": int(time.time()),
                "failure_phase": report.failure_phase,
                "status": report.status,
                "files": files,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    candidate_zip = output_path.with_name(f".{output_path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with zipfile.ZipFile(candidate_zip, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for path in sorted(staging_root.rglob("*")):
                if path.is_file():
                    archive.write(path, path.relative_to(staging_root).as_posix())
        with zipfile.ZipFile(candidate_zip) as archive:
            corrupt = archive.testzip()
            if corrupt:
                raise UpdateError(f"diagnostic ZIP checksum verification failed: {corrupt}")
            required = {"apply_report.json", "diagnostic_manifest.json", "README.txt"}
            missing = sorted(required.difference(archive.namelist()))
            if missing:
                raise UpdateError(
                    "diagnostic ZIP is missing required members: " + ", ".join(missing)
                )
        os.replace(candidate_zip, output_path)
    finally:
        candidate_zip.unlink(missing_ok=True)
    return output_path

def display_review(worktree: Path, current: str, integrated: str, package: Package, report: ApplyReport) -> None:
    print("[7/10] Review summary")
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
    total_started = time.perf_counter()
    package_source = Path(args.package).expanduser().resolve()
    report = ApplyReport(
        package=str(package_source),
        repository="unresolved",
        requested_test_mode=args.test_mode,
    )
    temp_root = Path(tempfile.mkdtemp(prefix="drpo-update-"))
    logs_dir = temp_root / "logs"
    package: Package | None = None
    repo: Path | None = None
    integration_worktree: Path | None = None
    integration_branch: str | None = None
    current: str | None = None
    base: str | None = None
    patch_commit: str | None = None
    integrated: str | None = None
    phase = "repository_resolution"
    try:
        repo = resolve_repo(args.repo)
        report.repository = str(repo)

        phase = "package_extract"
        with timed_phase(report, phase):
            package = extract_package(package_source, temp_root)
        phase = "repository_preflight"
        with timed_phase(report, phase):
            validate_repo(repo)
        phase = "refresh_main"
        with timed_phase(report, phase):
            refresh_main(repo)
        phase = "base_resolution"
        with timed_phase(report, phase):
            current = git_text(repo, "rev-parse", "HEAD")
            base_requested = read_full_sha(package.base_file, "BASE_COMMIT.txt")
            base = resolve_commit(repo, base_requested, "BASE_COMMIT.txt")
        report.package_base = base
        report.head_before = current

        if package.has_git_bundle:
            print("[2/10] Verifying Git bundle and patch equivalence...")
            phase = "bundle_verification"
            with timed_phase(report, phase):
                patch_commit = verify_bundle_and_patch(repo, package, base, temp_root)
            report.patch_commit = patch_commit
            ancestry = git(repo, "merge-base", "--is-ancestor", base, current, check=False)
            if ancestry.returncode != 0:
                raise UpdateError(
                    "BASE_COMMIT.txt is not an ancestor of current main; "
                    "automatic integration is intentionally disabled"
                )
        elif current != base:
            raise UpdateError(
                "bundle base commit does not match current main and no Git bundle is present\n"
                f"  package base: {base}\n  current HEAD: {current}"
            )
        else:
            print("[2/10] Legacy package detected; exact-base patch path will be used.")

        print("[3/10] Preparing isolated integration worktree...")
        message = args.message or summary_title(package.summary_file)
        phase = "integration"
        with timed_phase(report, phase):
            integration_worktree, integration_branch = prepare_integration_worktree(
                repo,
                current,
                temp_root,
            )
            integrated = create_integration_commit(
                integration_worktree,
                package,
                current,
                base,
                patch_commit,
                message,
                report,
            )
        report.source_integrated_commit_pre_normalization = integrated
        print("[4/11] Candidate source integration created; main is still untouched.")

        phase = "handoff_normalization"
        with timed_phase(report, phase):
            integrated = run_handoff_normalization(
                repo,
                integration_worktree,
                current=current,
                base=base,
                source_patch_commit=patch_commit,
                report=report,
                log_dir=logs_dir,
            )
        report.integrated_commit = integrated
        print("[5/11] Trusted handoff normalization complete; main is still untouched.")

        gate_errors: list[UpdateError] = []
        phase = "package_tests"
        with timed_phase(report, phase):
            package_error = run_package_tests(
                integration_worktree,
                package.test_file,
                report,
                logs_dir,
            )
        if package_error:
            gate_errors.append(package_error)

        phase = "repository_test_gate"
        with timed_phase(report, phase):
            repository_error = run_selected_tests(
                repo,
                integration_worktree,
                current,
                integrated,
                args.test_mode,
                report,
                logs_dir / "repository-gates",
            )
        if repository_error:
            gate_errors.append(repository_error)
        if gate_errors:
            phase = "aggregated_test_gates"
            detail = "\n".join(f"- {error}" for error in gate_errors)
            raise UpdateError(
                "candidate integration failed one or more aggregated test gates; "
                "main was not modified:\n" + detail
            )

        phase = "review"
        with timed_phase(report, phase):
            display_review(integration_worktree, current, integrated, package, report)

        if not args.yes:
            answer = input(
                f"Tests passed. Fast-forward main to '{message}' and push? [y/N] "
            ).strip()
            if answer.lower() != "y":
                report.status = "stopped_before_main_update"
                finalize_total_timing(report, total_started)
                path = write_report(report)
                print_timing_summary(report)
                print(f"Stopped before changing main. Report: {path}")
                return 0

        print("[9/11] Fast-forwarding verified commit onto main...")
        phase = "main_fast_forward"
        with timed_phase(report, phase):
            git(repo, "merge", "--ff-only", integrated, capture=False)
            report.head_after = git_text(repo, "rev-parse", "HEAD")
        report.status = "committed_local"

        if args.no_push:
            print("[10/11] Push skipped (--no-push).")
            report.main_bundle_export_skipped = "no_push"
            report.status = "success_no_push"
        else:
            print("[10/11] Pushing origin/main...")
            phase = "push"
            with timed_phase(report, phase):
                push = git(repo, "push", "origin", "main", check=False, capture=False)
            if push.returncode != 0:
                report.status = "committed_local_push_failed"
                raise UpdateError("push failed; verified commit remains on local main")
            report.pushed = True
            phase = "post_push_remote_verification"
            with timed_phase(report, phase):
                remote_head = remote_main_sha(repo)
            report.remote_head_after_push = remote_head
            if remote_head != report.head_after:
                report.status = "committed_local_push_verification_failed"
                raise UpdateError(
                    "push returned success but origin/main does not match local HEAD; "
                    f"local={report.head_after} remote={remote_head}"
                )
            if args.no_export_main_bundle:
                print("[11/11] Main bundle export skipped (--no-export-main-bundle).")
                report.main_bundle_export_skipped = "disabled_by_flag"
            else:
                print("[11/11] Exporting verified origin/main bundle to Downloads...")
                phase = "main_bundle_export"
                try:
                    with timed_phase(report, phase):
                        exported = export_main_bundle(
                            repo,
                            main_bundle_output_dir(args.main_bundle_dir),
                            report.head_after,
                        )
                except Exception as exc:
                    report.status = "pushed_main_bundle_export_failed"
                    raise UpdateError(
                        "origin/main was pushed and verified, but automatic main bundle "
                        f"export failed: {exc}"
                    ) from exc
                report.main_bundle_exported = True
                report.main_bundle_path = exported["versioned"]
                report.main_bundle_latest_path = exported["latest"]
                report.main_bundle_sha256 = exported["sha256"]
                print(f"Main bundle: {report.main_bundle_path}")
                print(f"Latest bundle: {report.main_bundle_latest_path}")
            report.status = "success"
        finalize_total_timing(report, total_started)
        path = write_report(report)
        print_timing_summary(report)
        print(f"Done.\nCommit: {report.head_after}\nApply report: {path}")
        return 0
    except (UpdateError, OSError, zipfile.BadZipFile) as exc:
        report.error = str(exc)
        report.failure_phase = phase
        if report.status not in {
            "committed_local_push_failed",
            "committed_local_push_verification_failed",
            "pushed_main_bundle_export_failed",
        }:
            report.status = "failed"
        finalize_total_timing(report, total_started)
        report_path: Path | None = None
        try:
            diagnostic_path = diagnostic_output_path(report, args.diagnostic_dir)
            report.diagnostic_zip = str(diagnostic_path)
            report_path = write_report(report)
            create_diagnostic_zip(
                output_path=diagnostic_path,
                staging_root=temp_root / "diagnostic-staging",
                report=report,
                report_path=report_path,
                repo=repo,
                package_source=package_source,
                package=package,
                integration_worktree=integration_worktree,
                current=current,
                base=base,
                patch_commit=patch_commit,
                integrated=integrated,
                logs_dir=logs_dir,
            )
        except Exception as diagnostic_exc:  # diagnostics must not hide root failure
            report.diagnostic_error = str(diagnostic_exc)
            report.diagnostic_zip = None
            try:
                if report_path is None:
                    report_path = write_report(report)
                else:
                    report_path.write_text(
                        json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n"
                    )
            except Exception as report_exc:
                print(
                    f"ERROR: {exc}\nDiagnostic reporting also failed: {report_exc}",
                    file=sys.stderr,
                )
                return 1
        print_timing_summary(report)
        message = f"ERROR: {exc}"
        if report_path is not None:
            message += f"\nApply report: {report_path}"
        if report.diagnostic_zip:
            message += f"\nDiagnostic ZIP: {report.diagnostic_zip}"
        elif report.diagnostic_error:
            message += f"\nDiagnostic ZIP generation failed: {report.diagnostic_error}"
        print(message, file=sys.stderr)
        return 1
    finally:
        if repo is not None and integration_worktree is not None:
            remove_worktree(repo, integration_worktree)
        if repo is not None and integration_branch is not None:
            git(repo, "branch", "-D", integration_branch, check=False)
        shutil.rmtree(temp_root, ignore_errors=True)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="drpo-update")
    parser.add_argument("package", nargs="?")
    parser.add_argument("--yes", "-y", action="store_true")
    parser.add_argument("--no-push", action="store_true")
    parser.add_argument(
        "--no-export-main-bundle",
        action="store_true",
        help="do not export DRPO_MAIN bundles after a verified successful push",
    )
    parser.add_argument(
        "--main-bundle-dir",
        help="post-push main bundle directory (default: ~/Downloads)",
    )
    parser.add_argument(
        "--doctor",
        action="store_true",
        help="run non-destructive updater self-tests and exit",
    )
    parser.add_argument(
        "--test-mode",
        choices=("auto", "fast", "full"),
        default="auto",
        help="auto selects a focused gate unless high-risk or unknown paths require the full suite",
    )
    parser.add_argument("--message", "-m")
    parser.add_argument(
        "--diagnostic-dir",
        help="failure diagnostic ZIP directory (default: ~/Downloads)",
    )
    parser.add_argument("--repo", help=argparse.SUPPRESS)
    parser.add_argument("--version", action="store_true")
    args = parser.parse_args(argv)
    if args.version:
        print(f"drpo-update {VERSION}")
        raise SystemExit(0)
    if not args.package and not args.doctor:
        parser.error("the following arguments are required: package")
    return args


def run_doctor(args: argparse.Namespace) -> int:
    """Run key updater paths in synthetic repositories without touching real main."""

    try:
        repo = resolve_repo(args.repo)
    except Exception as exc:
        print(f"DOCTOR REPOSITORY: FAIL ({exc})", file=sys.stderr)
        return 1

    failures: list[str] = []
    static_commands = [
        (
            "PYTHON COMPILE",
            [
                sys.executable,
                "-m",
                "compileall",
                "-q",
                "tools/drpo-update",
                "scripts/package_update.py",
                "scripts/verify_update_package.py",
            ],
        ),
        (
            "SHELL SYNTAX",
            ["bash", "-n", "tools/drpo-update/drpo-update", "tools/drpo-update/install.sh"],
        ),
    ]
    for label, command in static_commands:
        proc = run(command, cwd=repo, check=False, capture=True)
        if proc.returncode == 0:
            print(f"DOCTOR {label}: PASS")
        else:
            failures.append(label)
            print(f"DOCTOR {label}: FAIL", file=sys.stderr)
            detail = (proc.stdout or "") + (proc.stderr or "")
            if detail.strip():
                print(detail.rstrip(), file=sys.stderr)

    # Invoke each transactional scenario in a fresh pytest process.  This keeps
    # synthetic repositories and any child test gates isolated from one another.
    transaction_nodes = [
        "tests/test_update_git_bundle.py::test_bundle_verifier_proves_patch_tree_equivalence",
        "tests/test_update_git_bundle.py::test_stale_ancestral_bundle_merges_nonconflicting_main",
        "tests/test_update_git_bundle.py::test_bundle_conflict_fails_without_modifying_main",
        "tests/test_update_git_bundle.py::test_failed_package_tests_leave_main_untouched",
        "tests/test_update_git_bundle.py::test_default_failure_diagnostic_is_written_to_downloads",
        "tests/test_update_git_bundle.py::test_successful_push_defaults_versioned_and_latest_main_bundles_to_downloads",
        "tests/test_update_git_bundle.py::test_no_push_never_exports_official_main_bundle",
        "tests/test_update_git_bundle.py::test_post_push_export_failure_generates_diagnostic_without_rolling_back_push",
        "tests/test_update_packager.py::test_canonical_packager_always_emits_bundle_pair_and_manifest",
        "tests/test_update_packager.py::test_production_verifier_rejects_legacy_patch_only_package",
    ]
    transaction_failures: list[str] = []
    transaction_env = os.environ.copy()
    transaction_env.pop("PYTEST_CURRENT_TEST", None)
    transaction_env["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
    for node in transaction_nodes:
        proc = run(
            [sys.executable, "-m", "pytest", "-q", node],
            cwd=repo,
            check=False,
            capture=True,
            env=transaction_env,
        )
        if proc.returncode != 0:
            transaction_failures.append(node)
            detail = (proc.stdout or "") + (proc.stderr or "")
            if detail.strip():
                print(f"DOCTOR TRANSACTION NODE FAIL: {node}", file=sys.stderr)
                print(detail.rstrip(), file=sys.stderr)
    if transaction_failures:
        failures.append("TRANSACTION PATHS")
        print("DOCTOR TRANSACTION PATHS: FAIL", file=sys.stderr)
    else:
        print("DOCTOR TRANSACTION PATHS: PASS")

    if failures:
        print("DRPO UPDATE DOCTOR: FAIL (" + ", ".join(failures) + ")", file=sys.stderr)
        return 1
    print("DRPO UPDATE DOCTOR: PASS")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.doctor:
        return run_doctor(args)
    return apply_update(args)


if __name__ == "__main__":
    raise SystemExit(main())
