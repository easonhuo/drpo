#!/usr/bin/env python3
"""Verify a DRPO durable artifact before delivery or application."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import tempfile
import zipfile
from pathlib import Path

REQUIRED_TOP_LEVEL = {
    "update.patch",
    "BASE_COMMIT.txt",
    "CHANGE_SUMMARY.md",
    "TEST_COMMANDS.sh",
    "ARTIFACT_MANIFEST.json",
    "SHA256SUMS.txt",
}
FINAL_KINDS = {"governance", "experiment-final"}


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def parse_checksums(text: str) -> dict[str, str]:
    rows: dict[str, str] = {}
    for raw in text.splitlines():
        if not raw.strip():
            continue
        digest, sep, name = raw.partition("  ")
        if not sep or not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise ValueError(f"Invalid checksum row: {raw!r}")
        rows[name] = digest
    return rows


def validate_test_commands(text: str) -> None:
    lowered = text.lower()
    forbidden = ["/abs/path", "todo", "placeholder", "replace_me"]
    bad = [token for token in forbidden if token in lowered]
    if bad:
        raise ValueError(f"TEST_COMMANDS.sh contains placeholder tokens: {bad}")
    if "set -euo pipefail" not in text:
        raise ValueError("TEST_COMMANDS.sh must use 'set -euo pipefail'")


def verify_result_markers(names: set[str], manifest: dict[str, object]) -> None:
    kind = str(manifest.get("package_kind"))
    experiment_id = str(manifest.get("experiment_id"))
    prefix = f"results/{experiment_id}/"
    if kind == "experiment-final":
        required = prefix + "RUN_COMPLETE.json"
        if required not in names:
            raise ValueError(f"Missing {required}")
        audits = {prefix + "TERMINAL_AUDIT.json", prefix + "terminal_audit.json"}
        if not names.intersection(audits):
            raise ValueError("experiment-final is missing a terminal audit")
    elif kind == "experiment-failed":
        required = prefix + "RUN_FAILED.json"
        if required not in names:
            raise ValueError(f"Missing {required}")
    elif kind == "experiment-raw-complete":
        required = prefix + "RUN_RAW_COMPLETE.json"
        if required not in names:
            raise ValueError(f"Missing {required}")


def run_git_apply_check(repo: Path, patch: bytes, expected_sha: str) -> None:
    actual = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repo, text=True, capture_output=True, check=True
    ).stdout.strip()
    if actual != expected_sha:
        raise ValueError(f"Repository HEAD {actual} does not match BASE_COMMIT {expected_sha}")
    with tempfile.NamedTemporaryFile(suffix=".patch") as handle:
        handle.write(patch)
        handle.flush()
        result = subprocess.run(
            ["git", "apply", "--check", "--cached", handle.name],
            cwd=repo,
            text=True,
            capture_output=True,
        )
    if result.returncode != 0:
        raise ValueError(f"git apply --check failed:\n{result.stderr}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("package", type=Path)
    parser.add_argument("--repo-root", type=Path)
    parser.add_argument("--warning-mib", type=float, default=25.0)
    parser.add_argument(
        "--skip-head-match",
        action="store_true",
        help="Run structural verification without checking repo HEAD or git apply.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    package = args.package.resolve()
    if not package.is_file():
        raise SystemExit(f"Package does not exist: {package}")
    if not zipfile.is_zipfile(package):
        raise SystemExit(f"Not a valid ZIP: {package}")

    with zipfile.ZipFile(package) as zf:
        names = {name for name in zf.namelist() if not name.endswith("/")}
        missing = REQUIRED_TOP_LEVEL - names
        if missing:
            raise SystemExit(f"Missing required top-level files: {sorted(missing)}")
        base_text = zf.read("BASE_COMMIT.txt").decode("utf-8")
        if not re.fullmatch(r"[0-9a-f]{40}\n", base_text):
            raise SystemExit("BASE_COMMIT.txt must contain exactly one lowercase full SHA")
        base_sha = base_text.strip()

        manifest = json.loads(zf.read("ARTIFACT_MANIFEST.json"))
        if manifest.get("base_commit") != base_sha:
            raise SystemExit("Manifest base_commit does not match BASE_COMMIT.txt")
        kind = str(manifest.get("package_kind"))
        patch = zf.read("update.patch")
        if kind in FINAL_KINDS and not patch.strip():
            raise SystemExit(f"{kind} requires a non-empty update.patch")
        if kind in FINAL_KINDS and not any(
            name.startswith("modified_files/") for name in names
        ):
            raise SystemExit("Final packages require complete files under modified_files/")

        checksums = parse_checksums(zf.read("SHA256SUMS.txt").decode("utf-8"))
        expected_names = names - {"SHA256SUMS.txt"}
        if set(checksums) != expected_names:
            missing_hashes = expected_names - set(checksums)
            extra_hashes = set(checksums) - expected_names
            raise SystemExit(
                f"Checksum inventory mismatch; missing={sorted(missing_hashes)}, "
                f"extra={sorted(extra_hashes)}"
            )
        for name, expected in checksums.items():
            actual = sha256_bytes(zf.read(name))
            if actual != expected:
                raise SystemExit(f"Checksum mismatch for {name}")

        validate_test_commands(zf.read("TEST_COMMANDS.sh").decode("utf-8"))
        verify_result_markers(names, manifest)

        modified = list(manifest.get("modified_files") or [])
        for name in modified:
            path = f"modified_files/{name}"
            if path not in names:
                raise SystemExit(f"Manifest modified file is missing: {path}")

        if args.repo_root and not args.skip_head_match:
            run_git_apply_check(args.repo_root.resolve(), patch, base_sha)

    size_mib = package.stat().st_size / (1024 * 1024)
    report = {
        "package": str(package),
        "package_kind": kind,
        "base_commit": base_sha,
        "size_mib": round(size_mib, 3),
        "warning_threshold_mib": args.warning_mib,
        "over_warning_threshold": size_mib > args.warning_mib,
        "checksum_files": len(checksums),
        "git_apply_check": bool(args.repo_root and not args.skip_head_match),
        "verified": True,
    }
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
