#!/usr/bin/env python3
"""Verify the Git-bundle portion of a DRPO update package."""
from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
import sys
import tempfile
from pathlib import Path


def load_core(repo: Path):
    script_repo = Path(__file__).resolve().parents[1]
    path = script_repo / "tools" / "drpo-update" / "drpo_update.py"
    spec = importlib.util.spec_from_file_location("drpo_update_core", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load helper core: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--json", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo = args.repo.resolve()
    core = load_core(repo)
    temp = Path(tempfile.mkdtemp(prefix="drpo-bundle-verify-"))
    try:
        package = core.extract_package(args.package, temp)
        if not package.has_git_bundle:
            raise core.UpdateError("package does not contain change.bundle and PATCH_COMMIT.txt")
        base_requested = core.read_full_sha(package.base_file, "BASE_COMMIT.txt")
        base = core.resolve_commit(repo, base_requested, "BASE_COMMIT.txt")
        patch_commit = core.verify_bundle_and_patch(repo, package, base, temp)
        report = {
            "status": "PASS",
            "base_commit": base,
            "patch_commit": patch_commit,
            "patch": package.patch_file.name,
            "bundle": package.bundle_file.name,
        }
        if args.json:
            print(json.dumps(report, indent=2, sort_keys=True))
        else:
            print("Update Git bundle: PASS")
            print(f"Base: {base}")
            print(f"Patch commit: {patch_commit}")
        return 0
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    finally:
        shutil.rmtree(temp, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
