#!/usr/bin/env python3
"""Verify the canonical bundle-backed DRPO code-update package contract."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import shutil
import stat
import sys
import tempfile
import zipfile
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]
CORE_PATH = ROOT / "tools" / "drpo-update" / "drpo_update.py"
REQUIRED = {
    "BASE_COMMIT.txt",
    "update.patch",
    "CHANGE_SUMMARY.md",
    "TEST_COMMANDS.sh",
    "change.bundle",
    "PATCH_COMMIT.txt",
    "UPDATE_PACKAGE_MANIFEST.json",
}


def load_core():
    spec = importlib.util.spec_from_file_location("drpo_update_verifier_core", CORE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load updater core: {CORE_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def safe_relative(path: str) -> PurePosixPath:
    value = PurePosixPath(path)
    if value.is_absolute() or not value.parts or ".." in value.parts:
        raise ValueError(f"unsafe manifest path: {path}")
    return value


def verify(repo: Path, source: Path) -> dict[str, object]:
    core = load_core()
    temp = Path(tempfile.mkdtemp(prefix="drpo-package-verify-"))
    try:
        package = core.extract_package(source, temp)
        root = package.extracted_root
        missing = sorted(name for name in REQUIRED if not (root / name).is_file())
        if missing:
            raise core.UpdateError(
                "new production packages must be bundle-backed; missing: "
                + ", ".join(missing)
            )
        modified_root = root / "modified_files"
        if not modified_root.is_dir():
            raise core.UpdateError("modified_files/ is missing")
        base_requested = core.read_full_sha(package.base_file, "BASE_COMMIT.txt")
        base = core.resolve_commit(repo, base_requested, "BASE_COMMIT.txt")
        patch_commit = core.verify_bundle_and_patch(repo, package, base, temp)
        manifest = json.loads((root / "UPDATE_PACKAGE_MANIFEST.json").read_text())
        if manifest.get("schema_version") != 1:
            raise core.UpdateError("unsupported UPDATE_PACKAGE_MANIFEST schema")
        if manifest.get("package_format") != "bundle-backed-v1":
            raise core.UpdateError("manifest package_format must be bundle-backed-v1")
        if manifest.get("base_commit") != base:
            raise core.UpdateError("manifest base_commit does not match BASE_COMMIT.txt")
        if manifest.get("patch_commit") != patch_commit:
            raise core.UpdateError("manifest patch_commit does not match PATCH_COMMIT.txt")

        listed = manifest.get("files")
        if not isinstance(listed, list):
            raise core.UpdateError("manifest files must be a list")
        expected_paths = {
            path.relative_to(root).as_posix()
            for path in root.rglob("*")
            if path.is_file() and path.name != "UPDATE_PACKAGE_MANIFEST.json"
        }
        actual_paths: set[str] = set()
        listed_by_path: dict[str, dict[str, object]] = {}
        for item in listed:
            if not isinstance(item, dict) or not isinstance(item.get("path"), str):
                raise core.UpdateError("invalid manifest file entry")
            relative = safe_relative(item["path"]).as_posix()
            path = root / Path(relative)
            if not path.is_file():
                raise core.UpdateError(f"manifest file is missing: {relative}")
            if item.get("sha256") != sha256(path) or item.get("size_bytes") != path.stat().st_size:
                raise core.UpdateError(f"manifest checksum/size mismatch: {relative}")
            actual_paths.add(relative)
            listed_by_path[relative] = item
        if actual_paths != expected_paths:
            raise core.UpdateError(
                "manifest inventory mismatch; "
                f"missing={sorted(expected_paths - actual_paths)} "
                f"extra={sorted(actual_paths - expected_paths)}"
            )
        if listed_by_path.get("TEST_COMMANDS.sh", {}).get("executable") is not True:
            raise core.UpdateError("TEST_COMMANDS.sh is not executable in package manifest")
        if source.suffix.lower() == ".zip":
            with zipfile.ZipFile(source) as archive:
                infos = {info.filename: info for info in archive.infolist()}
            for relative, item in listed_by_path.items():
                info = infos.get(relative)
                if info is None:
                    raise core.UpdateError(f"ZIP member is missing: {relative}")
                executable = bool(stat.S_IMODE(info.external_attr >> 16) & 0o111)
                if executable != bool(item.get("executable")):
                    raise core.UpdateError(f"ZIP executable mode mismatch: {relative}")

        changed_rows = manifest.get("changed_files")
        if not isinstance(changed_rows, list):
            raise core.UpdateError("manifest changed_files must be a list")
        expected_after = {
            item["path"]
            for item in changed_rows
            if isinstance(item, dict) and item.get("status") != "D"
        }
        supplied = {
            path.relative_to(modified_root).as_posix()
            for path in modified_root.rglob("*")
            if path.is_file()
        }
        if supplied != expected_after:
            raise core.UpdateError("modified_files inventory does not match manifest")
        for item in changed_rows:
            if not isinstance(item, dict) or not isinstance(item.get("path"), str):
                raise core.UpdateError("invalid changed_files entry")
            if item.get("status") == "D":
                continue
            relative = safe_relative(item["path"]).as_posix()
            path = modified_root / Path(relative)
            if item.get("sha256") != sha256(path) or item.get("size_bytes") != path.stat().st_size:
                raise core.UpdateError(f"modified_files checksum/size mismatch: {relative}")
            package_item = listed_by_path.get(f"modified_files/{relative}")
            if package_item is None:
                raise core.UpdateError(f"modified_files manifest entry missing: {relative}")
            mode = "100755" if package_item.get("executable") is True else "100644"
            if item.get("git_mode") != mode:
                raise core.UpdateError(f"modified_files executable mode mismatch: {relative}")

        return {
            "status": "PASS",
            "base_commit": base,
            "patch_commit": patch_commit,
            "package_format": manifest["package_format"],
            "changed_files": len(changed_rows),
        }
    finally:
        shutil.rmtree(temp, ignore_errors=True)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--json", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        payload = verify(args.repo.resolve(), args.package.resolve())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print("Canonical update package: PASS")
        print(f"Base: {payload['base_commit']}")
        print(f"Patch commit: {payload['patch_commit']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
