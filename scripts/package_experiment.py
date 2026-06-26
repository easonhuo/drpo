#!/usr/bin/env python3
"""Build durable DRPO governance or experiment artifacts.

The final ZIP is compatible with the local ``drpo-update`` layout:
``update.patch``, ``BASE_COMMIT.txt``, ``CHANGE_SUMMARY.md``,
``TEST_COMMANDS.sh``, and ``modified_files/`` are always present.

Checkpoint, failed, and raw-complete packages may contain an empty update patch;
they are recovery artifacts rather than repository-completion artifacts. Governance
and experiment-final packages require a non-empty patch.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import stat
import subprocess
import tempfile
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

FINAL_KINDS = {"governance", "experiment-final"}
RESULT_KINDS = {
    "experiment-checkpoint",
    "experiment-failed",
    "experiment-raw-complete",
    "experiment-final",
}
ALL_KINDS = FINAL_KINDS | RESULT_KINDS


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def run(cmd: list[str], cwd: Path, *, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, cwd=cwd, check=check, text=True, capture_output=True)


def git_output(repo: Path, *args: str) -> str:
    return run(["git", *args], repo).stdout.strip()


def ensure_git_repo(repo: Path) -> None:
    result = run(["git", "rev-parse", "--is-inside-work-tree"], repo, check=False)
    if result.returncode != 0 or result.stdout.strip() != "true":
        raise SystemExit(f"Not a Git work tree: {repo}")


def validate_sha(value: str) -> str:
    value = value.strip()
    if len(value) != 40 or any(c not in "0123456789abcdefABCDEF" for c in value):
        raise SystemExit(f"Expected a full 40-character Git SHA, got: {value!r}")
    return value.lower()


def is_generated_artifact(path: Path) -> bool:
    """Exclude interpreter and test-run caches from durable update packages."""
    ignored_parts = {"__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache"}
    return bool(ignored_parts.intersection(path.parts)) or path.suffix in {".pyc", ".pyo"}


def changed_paths(repo: Path) -> list[Path]:
    tracked = git_output(repo, "diff", "--name-only", "HEAD", "--").splitlines()
    untracked = git_output(repo, "ls-files", "--others", "--exclude-standard").splitlines()
    paths = sorted({Path(x) for x in tracked + untracked if x.strip()})
    return [
        p for p in paths
        if (repo / p).is_file() and not is_generated_artifact(p)
    ]


def patch_for_repo(repo: Path, paths: list[Path]) -> str:
    tracked_names = set(git_output(repo, "ls-files").splitlines())
    tracked = [p.as_posix() for p in paths if p.as_posix() in tracked_names]
    untracked = [p.as_posix() for p in paths if p.as_posix() not in tracked_names]
    chunks: list[str] = []
    if tracked:
        result = run(
            ["git", "diff", "--binary", "--full-index", "HEAD", "--", *tracked],
            repo,
            check=True,
        )
        if result.stdout:
            chunks.append(result.stdout)
    for name in untracked:
        result = run(
            ["git", "diff", "--no-index", "--binary", "--", "/dev/null", name],
            repo,
            check=False,
        )
        if result.returncode not in (0, 1):
            raise RuntimeError(result.stderr)
        patch = result.stdout
        # ``git diff --no-index`` already emits ``b/<relative-path>`` when run
        # from the repository root. Keep it unchanged for ``git apply``.
        chunks.append(patch)
    return "".join(chunks)


def copy_changed_files(repo: Path, paths: Iterable[Path], destination: Path) -> None:
    for rel in paths:
        source = repo / rel
        target = destination / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)


def copy_result_tree(result_dir: Path, destination: Path, experiment_id: str) -> None:
    target = destination / experiment_id
    if target.exists():
        shutil.rmtree(target)
    shutil.copytree(result_dir, target, symlinks=False)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_checksums(stage: Path) -> dict[str, str]:
    checksums: dict[str, str] = {}
    for path in sorted(p for p in stage.rglob("*") if p.is_file()):
        rel = path.relative_to(stage).as_posix()
        if rel == "SHA256SUMS.txt":
            continue
        checksums[rel] = file_sha256(path)
    lines = [f"{digest}  {name}" for name, digest in checksums.items()]
    (stage / "SHA256SUMS.txt").write_text("\n".join(lines) + "\n")
    return checksums


def write_zip(stage: Path, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    temp_output = output.with_suffix(output.suffix + ".tmp")
    if temp_output.exists():
        temp_output.unlink()
    with zipfile.ZipFile(temp_output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
        for path in sorted(stage.rglob("*")):
            if path.is_file():
                zf.write(path, path.relative_to(stage).as_posix())
    temp_output.replace(output)


def find_terminal_audit(result_dir: Path) -> Path | None:
    candidates = [
        result_dir / "TERMINAL_AUDIT.json",
        result_dir / "terminal_audit.json",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def validate_result_markers(kind: str, result_dir: Path | None) -> None:
    if kind not in RESULT_KINDS:
        return
    if result_dir is None or not result_dir.is_dir():
        raise SystemExit(f"{kind} requires an existing --result-dir")
    if kind == "experiment-final":
        if not (result_dir / "RUN_COMPLETE.json").is_file():
            raise SystemExit("experiment-final requires RUN_COMPLETE.json")
        if find_terminal_audit(result_dir) is None:
            raise SystemExit(
                "experiment-final requires TERMINAL_AUDIT.json or terminal_audit.json"
            )
    elif kind == "experiment-failed":
        if not (result_dir / "RUN_FAILED.json").is_file():
            raise SystemExit("experiment-failed requires RUN_FAILED.json")
    elif kind == "experiment-raw-complete":
        if not (result_dir / "RUN_RAW_COMPLETE.json").is_file():
            raise SystemExit("experiment-raw-complete requires RUN_RAW_COMPLETE.json")


def build_summary(
    experiment_id: str,
    package_kind: str,
    base_commit: str,
    paths: list[Path],
    result_dir: Path | None,
    supplied: Path | None,
) -> str:
    if supplied:
        return supplied.read_text()
    changed = "\n".join(f"- `{p.as_posix()}`" for p in paths) or "- None"
    result_text = str(result_dir) if result_dir else "None"
    return f"""# Change Summary

- Governance/experiment ID: `{experiment_id}`
- Package kind: `{package_kind}`
- Base commit: `{base_commit}`
- Generated UTC: `{utc_now()}`
- Result directory: `{result_text}`

## Modified files

{changed}

## Purpose

Durably package the registered change or experiment evidence. Scientific acceptance is
controlled by `docs/handoff.md`, `experiments/registry.yaml`, and the package kind.

## Remaining items

Review the artifact manifest, execute `TEST_COMMANDS.sh`, and apply `update.patch` only
against the stated base commit.
"""


def executable_test_script(commands: list[str]) -> str:
    if not commands:
        commands = ["python3 -m pytest -q"]
    return "#!/usr/bin/env bash\nset -euo pipefail\n\n" + "\n".join(commands) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--package-kind", choices=sorted(ALL_KINDS), required=True)
    parser.add_argument("--result-dir", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--base-commit")
    parser.add_argument("--summary-file", type=Path)
    parser.add_argument("--test-command", action="append", default=[])
    parser.add_argument(
        "--changed-file",
        action="append",
        default=[],
        help="Repository-relative path. Repeat to override automatic change detection.",
    )
    parser.add_argument(
        "--source-file",
        action="append",
        default=[],
        help="Repository-relative source file copied into source_snapshot/.",
    )
    parser.add_argument("--warning-mib", type=float, default=25.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo = args.repo_root.resolve()
    ensure_git_repo(repo)
    base_commit = validate_sha(args.base_commit or git_output(repo, "rev-parse", "HEAD"))
    result_dir = args.result_dir.resolve() if args.result_dir else None
    validate_result_markers(args.package_kind, result_dir)

    if args.changed_file:
        paths = [Path(x) for x in args.changed_file]
        missing = [p for p in paths if not (repo / p).is_file()]
        if missing:
            raise SystemExit(f"Missing changed files: {missing}")
    else:
        paths = changed_paths(repo)
        if result_dir is not None:
            try:
                result_relative = result_dir.relative_to(repo)
            except ValueError:
                result_relative = None
            if result_relative is not None:
                paths = [
                    path
                    for path in paths
                    if path != result_relative and result_relative not in path.parents
                ]

    patch = patch_for_repo(repo, paths)
    if args.package_kind in FINAL_KINDS and not patch.strip():
        raise SystemExit(f"{args.package_kind} requires a non-empty update.patch")

    with tempfile.TemporaryDirectory(prefix="drpo_artifact_") as tmp:
        stage = Path(tmp)
        (stage / "modified_files").mkdir()
        (stage / "update.patch").write_text(patch)
        (stage / "BASE_COMMIT.txt").write_text(base_commit + "\n")
        (stage / "CHANGE_SUMMARY.md").write_text(
            build_summary(
                args.experiment_id,
                args.package_kind,
                base_commit,
                paths,
                result_dir,
                args.summary_file,
            )
        )
        test_script = stage / "TEST_COMMANDS.sh"
        test_script.write_text(executable_test_script(args.test_command))
        test_script.chmod(test_script.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
        copy_changed_files(repo, paths, stage / "modified_files")

        if result_dir:
            copy_result_tree(result_dir, stage / "results", args.experiment_id)

        source_root = stage / "source_snapshot"
        for name in args.source_file:
            rel = Path(name)
            source = repo / rel
            if not source.is_file():
                raise SystemExit(f"Missing source file: {rel}")
            target = source_root / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)

        manifest = {
            "schema_version": 1,
            "generated_utc": utc_now(),
            "experiment_id": args.experiment_id,
            "package_kind": args.package_kind,
            "base_commit": base_commit,
            "repository": "easonhuo/drpo",
            "branch": "main",
            "modified_files": [p.as_posix() for p in paths],
            "result_dir_name": args.experiment_id if result_dir else None,
            "source_files": list(args.source_file),
            "scientific_completion_claim": args.package_kind == "experiment-final",
            "durable_delivery_pending": True,
        }
        (stage / "ARTIFACT_MANIFEST.json").write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False) + "\n"
        )
        write_checksums(stage)
        write_zip(stage, args.output.resolve())

    size_mib = args.output.resolve().stat().st_size / (1024 * 1024)
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "size_mib": round(size_mib, 3),
                "warning_threshold_mib": args.warning_mib,
                "over_warning_threshold": size_mib > args.warning_mib,
                "base_commit": base_commit,
                "package_kind": args.package_kind,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
