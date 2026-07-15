#!/usr/bin/env python3
"""Prepare a no-training result-delivery shadow from completed compact evidence."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from runspec_lib import RunSpecError, is_model_like, now_utc, safe_relpath, sha256_file

TEXT_SUFFIXES = {".csv", ".json", ".jsonl", ".log", ".md", ".tsv", ".txt", ".yaml", ".yml"}


def prepare_shadow(repo: Path, source_dir_value: str, output_value: str) -> dict[str, object]:
    source_rel = safe_relpath(source_dir_value)
    output_rel = safe_relpath(output_value)
    source_dir = repo / Path(source_rel.as_posix())
    output = repo / Path(output_rel.as_posix())
    if not source_dir.is_dir():
        raise RunSpecError(f"shadow source directory is missing: {source_rel}")
    required = {"README.md", "RESULT_SUMMARY.json"}
    present = {path.name for path in source_dir.iterdir() if path.is_file()}
    missing = sorted(required - present)
    if missing:
        raise RunSpecError(f"shadow source is missing required files: {missing}")

    files: list[dict[str, object]] = []
    for path in sorted(source_dir.rglob("*")):
        if path.is_symlink():
            raise RunSpecError(f"shadow source may not contain symlinks: {path.relative_to(repo)}")
        if not path.is_file():
            continue
        rel = path.relative_to(repo).as_posix()
        if is_model_like(rel):
            raise RunSpecError(f"shadow source contains model-like evidence: {rel}")
        if path.suffix.lower() not in TEXT_SUFFIXES:
            raise RunSpecError(f"shadow source contains non-text evidence: {rel}")
        files.append(
            {
                "path": rel,
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    if not files:
        raise RunSpecError("shadow source contains no files")

    summary = json.loads((source_dir / "RESULT_SUMMARY.json").read_text(encoding="utf-8"))
    if not isinstance(summary, dict):
        raise RunSpecError("RESULT_SUMMARY.json must contain an object")
    payload: dict[str, object] = {
        "schema_version": 1,
        "status": "PASS",
        "purpose": "results_repo_transport_shadow_no_training",
        "prepared_at": now_utc(),
        "source_dir": source_rel.as_posix(),
        "source_experiment_id": summary.get("experiment_id"),
        "source_result_status": summary.get("status"),
        "source_package": summary.get("package"),
        "source_file_count": len(files),
        "source_files": files,
        "training_executed": False,
        "source_modified": False,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--source-dir", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    repo = Path(args.repo_root).resolve()
    try:
        payload = prepare_shadow(repo, args.source_dir, args.output)
    except Exception as exc:  # noqa: BLE001
        print(f"Result delivery shadow: FAIL error={exc}")
        return 1
    print(
        "Result delivery shadow: PASS "
        f"source={payload['source_dir']} files={payload['source_file_count']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
