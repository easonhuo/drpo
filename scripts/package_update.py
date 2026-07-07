#!/usr/bin/env python3
"""Build the canonical bundle-backed DRPO code-update ZIP.

The staging directory must already contain the human-authored contract files and
full after-images under ``modified_files/``.  This command is the only supported
producer for new ChatGPT code-update packages.  The local applicator continues
to consume historical exact-base patch-only packages for compatibility, but
this producer never emits one.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import stat
import subprocess
import sys
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
BUNDLE_BUILDER = ROOT / "scripts" / "create_update_git_bundle.py"
PACKAGE_VERIFIER = ROOT / "scripts" / "verify_update_package.py"
PACKAGE_PREFLIGHT = ROOT / "docs" / "update_packaging_hardening" / "preflight_update_package"
REQUIRED = {
    "BASE_COMMIT.txt",
    "update.patch",
    "CHANGE_SUMMARY.md",
    "TEST_COMMANDS.sh",
}
FORBIDDEN_PLACEHOLDERS = ("/ABS/PATH", "/abs/path", "TODO_PATH", "<PATH>")


class PackageBuildError(RuntimeError):
    """Expected package contract or build failure."""


def run(
    cmd: list[str],
    *,
    cwd: Path | None = None,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    proc = subprocess.run(
        cmd,
        cwd=cwd,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if check and proc.returncode != 0:
        detail = (proc.stderr or proc.stdout).strip()
        raise PackageBuildError(
            f"command failed ({proc.returncode}): {' '.join(cmd)}\n{detail}"
        )
    return proc


def git(repo: Path, *args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    return run(["git", "-C", str(repo), *args], check=check)


def git_text(repo: Path, *args: str) -> str:
    return git(repo, *args).stdout.strip()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def safe_relative(path: str) -> PurePosixPath:
    value = PurePosixPath(path)
    if value.is_absolute() or not value.parts or ".." in value.parts:
        raise PackageBuildError(f"unsafe repository-relative path: {path}")
    return value


def assert_safe_tree(root: Path) -> None:
    for path in root.rglob("*"):
        if path.is_symlink():
            raise PackageBuildError(f"symlinks are forbidden in update packages: {path}")


def parse_name_status(repo: Path, base: str, head: str) -> list[dict[str, str]]:
    proc = git(repo, "diff", "--name-status", "--find-renames", "--no-renames", base, head)
    rows: list[dict[str, str]] = []
    for line in proc.stdout.splitlines():
        if not line.strip():
            continue
        status, path = line.split("\t", 1)
        safe_relative(path)
        rows.append({"status": status, "path": path})
    return rows


def file_mode(repo: Path, commit: str, path: str) -> str:
    line = git_text(repo, "ls-tree", commit, "--", path)
    if not line:
        raise PackageBuildError(f"path is missing from candidate commit: {path}")
    return line.split()[0]


def candidate_blob(repo: Path, commit: str, path: str) -> str:
    line = git_text(repo, "ls-tree", commit, "--", path)
    if not line:
        raise PackageBuildError(f"path is missing from candidate commit: {path}")
    return line.split()[2]


def validate_staging(root: Path) -> None:
    if not root.is_dir():
        raise PackageBuildError(f"package root is not a directory: {root}")
    assert_safe_tree(root)
    missing = sorted(name for name in REQUIRED if not (root / name).is_file())
    if missing:
        raise PackageBuildError(f"missing required package files: {', '.join(missing)}")
    if not (root / "modified_files").is_dir():
        raise PackageBuildError("modified_files/ is required")
    patch = root / "update.patch"
    if not patch.read_bytes().strip():
        raise PackageBuildError("update.patch is empty")
    tests = root / "TEST_COMMANDS.sh"
    if not (stat.S_IMODE(tests.stat().st_mode) & 0o111):
        raise PackageBuildError("TEST_COMMANDS.sh must be executable")
    text = tests.read_text(errors="replace")
    if not text.startswith("#!/usr/bin/env bash\n") or "set -euo pipefail" not in text:
        raise PackageBuildError("TEST_COMMANDS.sh must use bash and set -euo pipefail")
    placeholders = [token for token in FORBIDDEN_PLACEHOLDERS if token in text]
    if placeholders:
        raise PackageBuildError(
            "TEST_COMMANDS.sh contains placeholder path tokens: " + ", ".join(placeholders)
        )


def load_bundle_builder():
    spec = importlib.util.spec_from_file_location("drpo_bundle_builder", BUNDLE_BUILDER)
    if spec is None or spec.loader is None:
        raise PackageBuildError(f"cannot load bundle builder: {BUNDLE_BUILDER}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def compare_modified_files(
    repo: Path,
    root: Path,
    base: str,
    patch_commit: str,
) -> list[dict[str, Any]]:
    changed = parse_name_status(repo, base, patch_commit)
    expected_after = {row["path"] for row in changed if row["status"] != "D"}
    supplied: set[str] = set()
    modified_root = root / "modified_files"
    for path in modified_root.rglob("*"):
        if path.is_dir():
            continue
        relative = path.relative_to(modified_root).as_posix()
        safe_relative(relative)
        supplied.add(relative)
    if supplied != expected_after:
        missing = sorted(expected_after - supplied)
        extra = sorted(supplied - expected_after)
        raise PackageBuildError(
            "modified_files inventory mismatch; "
            f"missing={missing or '[]'} extra={extra or '[]'}"
        )

    manifest_rows: list[dict[str, Any]] = []
    for row in changed:
        path = row["path"]
        item: dict[str, Any] = {"path": path, "status": row["status"]}
        if row["status"] != "D":
            supplied_path = modified_root / Path(path)
            blob = candidate_blob(repo, patch_commit, path)
            actual_blob = git_text(repo, "hash-object", str(supplied_path))
            if actual_blob != blob:
                raise PackageBuildError(
                    f"modified_files content mismatch for {path}: {actual_blob} != {blob}"
                )
            mode = file_mode(repo, patch_commit, path)
            supplied_mode = "100755" if os.access(supplied_path, os.X_OK) else "100644"
            if mode not in {"100644", "100755"}:
                raise PackageBuildError(f"unsupported candidate file mode {mode} for {path}")
            if mode != supplied_mode:
                raise PackageBuildError(
                    f"modified_files executable mode mismatch for {path}: "
                    f"package={supplied_mode} candidate={mode}"
                )
            item.update(
                {
                    "git_mode": mode,
                    "git_blob": blob,
                    "sha256": sha256(supplied_path),
                    "size_bytes": supplied_path.stat().st_size,
                }
            )
        manifest_rows.append(item)
    return manifest_rows


def write_manifest(
    root: Path,
    *,
    repo: Path,
    base: str,
    patch_commit: str,
    changed: list[dict[str, Any]],
) -> None:
    files: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.name == "UPDATE_PACKAGE_MANIFEST.json":
            continue
        relative = path.relative_to(root).as_posix()
        files.append(
            {
                "path": relative,
                "sha256": sha256(path),
                "size_bytes": path.stat().st_size,
                "executable": bool(stat.S_IMODE(path.stat().st_mode) & 0o111),
            }
        )
    payload = {
        "schema_version": 1,
        "policy_id": "GOV-UPDATE-BUNDLE-DEFAULT-01",
        "producer": "scripts/package_update.py",
        "package_format": "bundle-backed-v1",
        "base_commit": base,
        "patch_commit": patch_commit,
        "repository_head_at_build": git_text(repo, "rev-parse", "HEAD"),
        "changed_files": changed,
        "files": files,
    }
    (root / "UPDATE_PACKAGE_MANIFEST.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n"
    )


def write_zip(root: Path, output: Path) -> None:
    output = output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temp = output.with_name(f".{output.name}.tmp-{os.getpid()}")
    temp.unlink(missing_ok=True)
    try:
        with zipfile.ZipFile(temp, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for path in sorted(root.rglob("*")):
                if not path.is_file():
                    continue
                relative = path.relative_to(root).as_posix()
                info = zipfile.ZipInfo(relative)
                info.create_system = 3
                mode = stat.S_IMODE(path.stat().st_mode)
                info.external_attr = (stat.S_IFREG | mode) << 16
                info.compress_type = zipfile.ZIP_DEFLATED
                archive.writestr(info, path.read_bytes())
        os.replace(temp, output)
    finally:
        temp.unlink(missing_ok=True)


def failure_report_paths(output: Path) -> tuple[Path, Path]:
    output = output.expanduser().resolve()
    return (
        output.with_name(f"{output.name}.preflight-failed.json"),
        output.with_name(f"{output.name}.preflight-failed.md"),
    )


def clear_failure_reports(output: Path) -> None:
    for path in failure_report_paths(output):
        path.unlink(missing_ok=True)


def write_failure_reports(
    output: Path,
    *,
    title: str,
    detail: str,
    stdout: str,
    stderr: str,
) -> None:
    json_path, md_path = failure_report_paths(output)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "status": "FAIL",
        "title": title,
        "detail": detail,
        "stdout": stdout,
        "stderr": stderr,
    }
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    md_path.write_text(
        f"# {title}\n\n"
        f"{detail.strip() or 'No detail was provided.'}\n\n"
        "## stdout\n\n"
        "```text\n" + (stdout.strip() or "<empty>") + "\n```\n\n"
        "## stderr\n\n"
        "```text\n" + (stderr.strip() or "<empty>") + "\n```\n",
        encoding="utf-8",
    )


def repo_supports_release_preflight(repo: Path) -> bool:
    return (repo / "tools" / "drpo-update" / "drpo_update.py").is_file()


def run_package_preflight(
    repo: Path,
    package_path: Path,
    *,
    report_path: Path | None = None,
    failure_output: Path | None = None,
) -> None:
    if not PACKAGE_PREFLIGHT.is_file():
        raise PackageBuildError(
            f"package preflight helper is missing: {PACKAGE_PREFLIGHT}"
        )
    command = [
        sys.executable,
        str(PACKAGE_PREFLIGHT),
        "--repo",
        str(repo),
        "--package",
        str(package_path),
        "--json",
    ]
    if report_path is not None:
        command.extend(["--report", str(report_path)])
    proc = run(command, check=False)
    if proc.returncode != 0:
        write_failure_reports(
            failure_output or package_path,
            title="DRPO update package preflight failed",
            detail="Runnable package was not emitted because release preflight failed.",
            stdout=proc.stdout,
            stderr=proc.stderr,
        )
        raise PackageBuildError(
            "release preflight failed; runnable package was not emitted:\n"
            + (proc.stderr or proc.stdout).strip()
        )


def build(
    repo: Path,
    package_root: Path,
    output: Path,
    message: str | None,
    *,
    unsafe_skip_preflight: bool = False,
) -> str:
    repo = Path(git_text(repo, "rev-parse", "--show-toplevel"))
    output = output.expanduser().resolve()
    temp_output = output.with_name(f".{output.stem}.tmp-preflight-{os.getpid()}.zip")
    temp_report = temp_output.with_name(f"{temp_output.name}.preflight.json")
    validate_staging(package_root)
    base = (package_root / "BASE_COMMIT.txt").read_text().strip()
    if git(repo, "rev-parse", f"{base}^{{commit}}", check=False).stdout.strip() != base:
        raise PackageBuildError(f"BASE_COMMIT.txt is unavailable in repository: {base}")

    for generated in ("change.bundle", "PATCH_COMMIT.txt", "UPDATE_PACKAGE_MANIFEST.json"):
        (package_root / generated).unlink(missing_ok=True)
    builder = load_bundle_builder()
    try:
        patch_commit = builder.build(repo, package_root, message)
    except Exception as exc:  # normalize builder errors for the CLI
        raise PackageBuildError(str(exc)) from exc
    changed = compare_modified_files(repo, package_root, base, patch_commit)
    write_manifest(
        package_root,
        repo=repo,
        base=base,
        patch_commit=patch_commit,
        changed=changed,
    )

    temp_output.unlink(missing_ok=True)
    temp_report.unlink(missing_ok=True)
    try:
        write_zip(package_root, temp_output)
        verify = run(
            [
                sys.executable,
                str(PACKAGE_VERIFIER),
                "--repo",
                str(repo),
                "--package",
                str(temp_output),
            ],
            check=False,
        )
        if verify.returncode != 0:
            write_failure_reports(
                output,
                title="DRPO update package verification failed",
                detail="Runnable package was not emitted because canonical package verification failed.",
                stdout=verify.stdout,
                stderr=verify.stderr,
            )
            raise PackageBuildError(
                "canonical package verification failed; runnable package was not emitted:\n"
                + (verify.stderr or verify.stdout).strip()
            )
        if unsafe_skip_preflight or not repo_supports_release_preflight(repo):
            reason = (
                "UNSAFE: release preflight was explicitly skipped."
                if unsafe_skip_preflight
                else "UNSAFE: release preflight was skipped for a non-DRPO synthetic repository."
            )
            write_failure_reports(
                output,
                title="DRPO update package preflight skipped",
                detail=(
                    f"{reason} This package must not be treated as a fully gated "
                    "runnable package."
                ),
                stdout="",
                stderr="",
            )
        else:
            run_package_preflight(
                repo,
                temp_output,
                report_path=temp_report,
                failure_output=output,
            )
            clear_failure_reports(output)
        output.parent.mkdir(parents=True, exist_ok=True)
        os.replace(temp_output, output)
        return patch_commit
    except Exception:
        temp_output.unlink(missing_ok=True)
        temp_report.unlink(missing_ok=True)
        raise


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--package-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--message")
    parser.add_argument(
        "--unsafe-skip-preflight",
        action="store_true",
        help=(
            "UNSAFE: emit a package without release preflight. This is intended "
            "only for synthetic/bootstrap tests and marks the output as not fully gated."
        ),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        commit = build(
            args.repo.resolve(),
            args.package_root.resolve(),
            args.output,
            args.message,
            unsafe_skip_preflight=args.unsafe_skip_preflight,
        )
    except PackageBuildError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(f"Created canonical bundle-backed package: {args.output.expanduser().resolve()}")
    print(f"PATCH_COMMIT={commit}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
