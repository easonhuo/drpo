#!/usr/bin/env python3
"""Validate that a DRPO update package stays within its declared scope.

This is a producer-side scope checker for ChatGPT-generated update packages. It
is intentionally deterministic: it does not try to infer the user's intent from
free-form prose. Instead, the package must declare task metadata in
CHANGE_SUMMARY.md, and this script checks that the declared task type, claim,
modified-file list, and control-plane status match the actual patch payload.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import sys
import tempfile
import zipfile
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RULES = ROOT / "scripts" / "governance_scope_rules.yaml"
FIELD_ALIASES = {
    "task type": "task_type",
    "task_type": "task_type",
    "claim or experiment id": "claim_or_experiment_id",
    "claim/experiment id": "claim_or_experiment_id",
    "claim_or_experiment_id": "claim_or_experiment_id",
    "user-requested scope": "user_requested_scope",
    "user requested scope": "user_requested_scope",
    "user_requested_scope": "user_requested_scope",
    "first-failure classification": "first_failure_classification",
    "first failure classification": "first_failure_classification",
    "first_failure_classification": "first_failure_classification",
    "control-plane touched": "control_plane_touched",
    "control plane touched": "control_plane_touched",
    "control_plane_touched": "control_plane_touched",
}
FIELD_RE = re.compile(
    r"^\s*(?:[-*]\s*)?(?P<field>[A-Za-z0-9_ /-]+?)\s*:\s*(?P<value>.*?)\s*$"
)
HEADING_RE = re.compile(r"^#{1,6}\s+(?P<title>.+?)\s*$")
BULLET_PATH_RE = re.compile(r"^\s*[-*]\s+`?(?P<path>[^`\s]+)`?\s*$")
FULL_SHA_RE = re.compile(r"^[0-9a-f]{40}$")


class ScopeError(ValueError):
    """Raised when package structure is invalid enough to stop checking."""


@dataclass(frozen=True)
class PatchFile:
    path: str
    status: str


@dataclass
class ScopeReport:
    status: str = "PASS"
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    changed_files: list[str] = field(default_factory=list)
    task_type: str | None = None
    claim_or_experiment_id: str | None = None

    def fail(self, message: str) -> None:
        self.errors.append(message)
        self.status = "FAIL"

    def warn(self, message: str) -> None:
        self.warnings.append(message)
        if self.status == "PASS":
            self.status = "PASS_WITH_WARNINGS"

    def to_json(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "errors": self.errors,
            "warnings": self.warnings,
            "changed_files": self.changed_files,
            "task_type": self.task_type,
            "claim_or_experiment_id": self.claim_or_experiment_id,
        }


def safe_relative(value: str, *, label: str = "path") -> PurePosixPath:
    path = PurePosixPath(value)
    if path.is_absolute() or not path.parts or ".." in path.parts:
        raise ScopeError(f"unsafe {label}: {value!r}")
    return path


def read_yaml_mapping(path: Path) -> dict[str, Any]:
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise ScopeError(f"could not read rules file {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ScopeError(f"rules file must contain one YAML mapping: {path}")
    if payload.get("schema_version") != 1:
        raise ScopeError("unsupported governance scope rules schema_version")
    return payload


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def extract_zip_safe(source: Path) -> Path:
    temp = Path(tempfile.mkdtemp(prefix="drpo-scope-package-"))
    try:
        with zipfile.ZipFile(source) as archive:
            names = archive.namelist()
            if not names:
                raise ScopeError("package ZIP is empty")
            seen: set[str] = set()
            for info in archive.infolist():
                if info.filename in seen:
                    raise ScopeError(f"duplicate ZIP member: {info.filename}")
                seen.add(info.filename)
                member = safe_relative(info.filename, label="ZIP member")
                if info.is_dir():
                    continue
                mode = (info.external_attr >> 16) & 0o170000
                if mode == 0o120000:
                    raise ScopeError(f"symlink ZIP member is forbidden: {info.filename}")
                target = temp / Path(member.as_posix())
                target.parent.mkdir(parents=True, exist_ok=True)
                with archive.open(info) as src, target.open("wb") as dst:
                    shutil.copyfileobj(src, dst)
    except Exception:
        shutil.rmtree(temp, ignore_errors=True)
        raise
    return temp


def package_root_from_args(args: argparse.Namespace) -> tuple[Path, Path | None]:
    if bool(args.package) == bool(args.package_root):
        raise ScopeError("pass exactly one of --package or --package-root")
    if args.package:
        temp = extract_zip_safe(args.package.resolve())
        return temp, temp
    return args.package_root.resolve(), None


def required_file(root: Path, name: str) -> Path:
    path = root / name
    if not path.is_file():
        raise ScopeError(f"required package file is missing: {name}")
    return path


def parse_patch_files(patch_text: str) -> list[PatchFile]:
    files: list[PatchFile] = []
    current_old: str | None = None
    current_new: str | None = None
    current_deleted = False
    current_new_file = False

    def flush() -> None:
        nonlocal current_old, current_new, current_deleted, current_new_file
        if current_new is None and current_old is None:
            return
        if current_deleted:
            path = current_old
            status = "D"
        else:
            path = current_new or current_old
            status = "A" if current_new_file else "M"
        if path is None:
            raise ScopeError("could not determine path from patch hunk")
        safe_relative(path, label="patch path")
        files.append(PatchFile(path=path, status=status))
        current_old = None
        current_new = None
        current_deleted = False
        current_new_file = False

    for line in patch_text.splitlines():
        if line.startswith("diff --git "):
            flush()
            parts = line.split()
            if len(parts) < 4 or not parts[2].startswith("a/") or not parts[3].startswith("b/"):
                raise ScopeError(f"unsupported diff header: {line}")
            current_old = parts[2][2:]
            current_new = parts[3][2:]
            continue
        if line.startswith("new file mode "):
            current_new_file = True
        elif line.startswith("deleted file mode "):
            current_deleted = True
        elif line.startswith("--- ") and line[4:] == "/dev/null":
            current_new_file = True
        elif line.startswith("+++ ") and line[4:] == "/dev/null":
            current_deleted = True
    flush()
    if not files:
        raise ScopeError("update.patch does not contain any changed files")
    return files


def normalize_field_name(raw: str) -> str | None:
    key = " ".join(raw.strip().lower().replace("_", "_").split())
    return FIELD_ALIASES.get(key)


def parse_change_summary(path: Path) -> tuple[dict[str, str], set[str]]:
    fields: dict[str, str] = {}
    declared_files: set[str] = set()
    current_heading: str | None = None
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        heading = HEADING_RE.match(line)
        if heading:
            current_heading = heading.group("title").strip().lower()
            continue
        field = FIELD_RE.match(line)
        if field:
            canonical = normalize_field_name(field.group("field"))
            if canonical is not None:
                fields[canonical] = field.group("value").strip()
            continue
        if current_heading == "modified files":
            bullet = BULLET_PATH_RE.match(line)
            if bullet:
                declared_files.add(safe_relative(bullet.group("path"), label="summary path").as_posix())
    return fields, declared_files


def load_manifest(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ScopeError(f"could not parse UPDATE_PACKAGE_MANIFEST.json: {exc}") from exc
    if not isinstance(payload, dict):
        raise ScopeError("UPDATE_PACKAGE_MANIFEST.json must contain one object")
    return payload


def manifest_changed_files(manifest: dict[str, Any]) -> set[str]:
    rows = manifest.get("changed_files")
    if not isinstance(rows, list):
        raise ScopeError("UPDATE_PACKAGE_MANIFEST.json changed_files must be a list")
    paths: set[str] = set()
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get("path"), str):
            raise ScopeError("manifest changed_files contains a malformed entry")
        paths.add(safe_relative(row["path"], label="manifest changed path").as_posix())
    return paths


def manifest_expected_after_files(manifest: dict[str, Any]) -> set[str]:
    rows = manifest.get("changed_files")
    if not isinstance(rows, list):
        raise ScopeError("UPDATE_PACKAGE_MANIFEST.json changed_files must be a list")
    paths: set[str] = set()
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get("path"), str):
            raise ScopeError("manifest changed_files contains a malformed entry")
        if row.get("status") != "D":
            paths.add(safe_relative(row["path"], label="manifest after path").as_posix())
    return paths


def path_matches_prefix(path: str, prefix: str) -> bool:
    prefix_path = safe_relative(prefix.rstrip("/") if prefix != "AGENTS.md" else prefix).as_posix()
    if prefix.endswith("/"):
        return path.startswith(prefix)
    return path == prefix_path


def any_prefix(path: str, prefixes: list[str]) -> bool:
    return any(path_matches_prefix(path, prefix) for prefix in prefixes)


def top_level(path: str) -> str:
    return path.split("/", 1)[0]


def check_manifest_inventory(report: ScopeReport, root: Path, manifest: dict[str, Any]) -> None:
    expected_after = manifest_expected_after_files(manifest)
    supplied = {
        path.relative_to(root / "modified_files").as_posix()
        for path in (root / "modified_files").rglob("*")
        if path.is_file()
    }
    if supplied != expected_after:
        report.fail(
            "modified_files inventory mismatch; "
            f"missing={sorted(expected_after - supplied)} extra={sorted(supplied - expected_after)}"
        )

    rows = manifest.get("changed_files", [])
    if isinstance(rows, list):
        for row in rows:
            if not isinstance(row, dict) or row.get("status") == "D":
                continue
            path = row.get("path")
            if not isinstance(path, str):
                continue
            supplied_path = root / "modified_files" / path
            if not supplied_path.is_file():
                continue
            expected_hash = row.get("sha256")
            expected_size = row.get("size_bytes")
            if isinstance(expected_hash, str) and expected_hash != sha256(supplied_path):
                report.fail(f"modified_files checksum mismatch: {path}")
            if isinstance(expected_size, int) and expected_size != supplied_path.stat().st_size:
                report.fail(f"modified_files size mismatch: {path}")


def check_summary_metadata(
    report: ScopeReport,
    fields: dict[str, str],
    declared_files: set[str],
    changed_files: set[str],
    rules: dict[str, Any],
) -> None:
    for required_field in rules.get("required_summary_fields", []):
        if not fields.get(required_field):
            report.fail(f"CHANGE_SUMMARY.md missing required field: {required_field}")
    task_type = fields.get("task_type")
    report.task_type = task_type
    report.claim_or_experiment_id = fields.get("claim_or_experiment_id")
    allowed = set(rules.get("allowed_task_types", []))
    if task_type and task_type not in allowed:
        report.fail(f"unsupported task_type: {task_type}")
    claim = fields.get("claim_or_experiment_id", "")
    if claim and claim.lower() in {"none", "n/a", "na"}:
        report.fail("claim_or_experiment_id must be a real claim or experiment ID")
    classification = fields.get("first_failure_classification")
    allowed_classifications = set(rules.get("first_failure_classifications", []))
    if classification and classification not in allowed_classifications:
        report.fail(f"unsupported first_failure_classification: {classification}")
    if not declared_files:
        report.fail("CHANGE_SUMMARY.md must list actual paths under '## Modified files'")
    missing = changed_files - declared_files
    extra = declared_files - changed_files
    if missing or extra:
        report.fail(
            "CHANGE_SUMMARY.md modified-file list does not match update.patch; "
            f"missing={sorted(missing)} extra={sorted(extra)}"
        )


def check_task_scope(
    report: ScopeReport,
    fields: dict[str, str],
    changed_files: set[str],
    rules: dict[str, Any],
) -> None:
    task_type = fields.get("task_type")
    if not task_type:
        return
    task_rules = rules.get("task_rules", {}).get(task_type, {})
    if not isinstance(task_rules, dict):
        report.fail(f"missing task rule for task_type: {task_type}")
        return

    control_paths = rules.get("control_plane_paths", [])
    touched_control = sorted(path for path in changed_files if any_prefix(path, control_paths))
    declared_control = fields.get("control_plane_touched", "").strip().lower()
    control_declared_yes = declared_control in {"yes", "true", "y"}
    control_declared_no = declared_control in {"no", "false", "n"}
    if declared_control and not (control_declared_yes or control_declared_no):
        report.fail("control_plane_touched must be yes or no")
    if touched_control and control_declared_no:
        report.fail(f"control_plane_touched is no but control-plane paths changed: {touched_control}")
    if not touched_control and control_declared_yes and not task_rules.get("require_control_plane_touched"):
        report.warn("control_plane_touched is yes but no configured control-plane path changed")
    if touched_control and not task_rules.get("allow_control_plane"):
        report.fail(f"task_type {task_type} may not modify control-plane paths: {touched_control}")
    if task_rules.get("require_control_plane_touched") and not touched_control:
        report.fail(f"task_type {task_type} requires an explicit control-plane change")

    allowed_prefixes = task_rules.get("allowed_prefixes")
    if isinstance(allowed_prefixes, list) and allowed_prefixes:
        outside = sorted(path for path in changed_files if not any_prefix(path, allowed_prefixes))
        if outside:
            report.fail(f"task_type {task_type} changed paths outside allowed prefixes: {outside}")

    forbidden_prefixes = task_rules.get("forbidden_prefixes")
    if isinstance(forbidden_prefixes, list) and forbidden_prefixes:
        forbidden = sorted(path for path in changed_files if any_prefix(path, forbidden_prefixes))
        if forbidden:
            report.fail(f"task_type {task_type} changed forbidden paths: {forbidden}")

    protected_core = task_rules.get("forbid_existing_pipeline_core_paths")
    if isinstance(protected_core, list) and protected_core:
        blocked = sorted(path for path in changed_files if any_prefix(path, protected_core))
        if blocked:
            report.fail(f"task_type {task_type} may not change existing pipeline core paths: {blocked}")

    if task_rules.get("require_claim_or_experiment_id") and not fields.get("claim_or_experiment_id"):
        report.fail(f"task_type {task_type} requires claim_or_experiment_id")

    if task_rules.get("warn_if_crosses_top_level_dirs"):
        companions = set(task_rules.get("companion_top_level_dirs", []))
        primary_dirs = {top_level(path) for path in changed_files if top_level(path) not in companions}
        if len(primary_dirs) > 1:
            report.warn(
                f"bug-fix scope crosses multiple primary top-level directories: {sorted(primary_dirs)}"
            )


def validate_package(root: Path, rules_path: Path = DEFAULT_RULES) -> ScopeReport:
    rules = read_yaml_mapping(rules_path)
    patch_path = required_file(root, "update.patch")
    summary_path = required_file(root, "CHANGE_SUMMARY.md")
    manifest_path = required_file(root, "UPDATE_PACKAGE_MANIFEST.json")
    required_file(root, "BASE_COMMIT.txt")
    modified_root = root / "modified_files"
    if not modified_root.is_dir():
        raise ScopeError("modified_files/ is missing")

    base = (root / "BASE_COMMIT.txt").read_text(encoding="utf-8").strip()
    if not FULL_SHA_RE.fullmatch(base):
        raise ScopeError("BASE_COMMIT.txt must contain one full lowercase SHA")

    patch_files = parse_patch_files(patch_path.read_text(encoding="utf-8", errors="replace"))
    changed_files = {item.path for item in patch_files}
    report = ScopeReport(changed_files=sorted(changed_files))
    manifest = load_manifest(manifest_path)
    manifest_files = manifest_changed_files(manifest)
    if manifest_files != changed_files:
        report.fail(
            "UPDATE_PACKAGE_MANIFEST.json changed_files does not match update.patch; "
            f"missing={sorted(changed_files - manifest_files)} extra={sorted(manifest_files - changed_files)}"
        )
    check_manifest_inventory(report, root, manifest)
    fields, declared_files = parse_change_summary(summary_path)
    check_summary_metadata(report, fields, declared_files, changed_files, rules)
    check_task_scope(report, fields, changed_files, rules)
    return report


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", type=Path, help="Path to a drpo-update ZIP package")
    parser.add_argument("--package-root", type=Path, help="Path to an extracted package root")
    parser.add_argument("--rules", type=Path, default=DEFAULT_RULES)
    parser.add_argument("--json", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    temp: Path | None = None
    try:
        root, temp = package_root_from_args(args)
        report = validate_package(root, args.rules.resolve())
    except Exception as exc:
        payload = {"status": "FAIL", "errors": [str(exc)], "warnings": []}
        if args.json:
            print(json.dumps(payload, indent=2, sort_keys=True))
        else:
            print(f"Scope gate: FAIL\nERROR: {exc}", file=sys.stderr)
        return 1
    finally:
        if temp is not None:
            shutil.rmtree(temp, ignore_errors=True)

    if args.json:
        print(json.dumps(report.to_json(), indent=2, sort_keys=True))
    else:
        print(f"Scope gate: {report.status}")
        if report.task_type:
            print(f"Task type: {report.task_type}")
        if report.claim_or_experiment_id:
            print(f"Claim/experiment: {report.claim_or_experiment_id}")
        for warning in report.warnings:
            print(f"WARN: {warning}")
        for error in report.errors:
            print(f"ERROR: {error}")
    return 1 if report.errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
