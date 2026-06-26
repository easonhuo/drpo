#!/usr/bin/env python3
"""Validate the canonical DRPO formal-experiment execution channel.

This gate is deliberately independent of experiment science. It checks that every
registry entry explicitly declares an execution class and that every active or
blocked formal experiment is routed through the one hardened guard/package/verify
channel. It also rejects new formal entrypoints that create archives directly,
unless an exact legacy recovery-checkpoint exception is registered.
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import stat
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import yaml


CHANNEL_ID = "hardened-v1"
POLICY_ID = "GOV-FORMAL-ENTRYPOINT-01"
ALLOWED_EXECUTION_CLASSES = {"formal", "pilot", "historical_formal", "superseded"}
ALLOWED_ACTIVATION_STATES = {"active", "blocked"}
ALLOWED_ENTRYPOINT_STATES = {"implemented", "planned"}
ALLOWED_ARCHIVE_MODES = {"forbid", "legacy_exception"}
SHA_RE = re.compile(r"^[0-9a-f]{40}$")
EXPERIMENT_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")

CANONICAL_FIELDS = {
    "guard_entrypoint": "scripts/run_experiment_guard_hardened.py",
    "package_entrypoint": "scripts/package_experiment_hardened.py",
    "verify_entrypoint": "scripts/verify_experiment_package_hardened.py",
    "hardened_core": "scripts/artifact_protocol_hardened.py",
    "artifact_protocol": "docs/formal_experiment_artifact_protocol.md",
    "validator_entrypoint": "scripts/validate_formal_execution_channel.py",
}

WRAPPER_IMPORTS = {
    "guard_entrypoint": "from artifact_protocol_hardened import guard_main",
    "package_entrypoint": "from artifact_protocol_hardened import package_main",
    "verify_entrypoint": "from artifact_protocol_hardened import verify_main",
}

REQUIRED_FORMAL_REQUIREMENTS = {
    "require_explicit_execution_class": True,
    "require_canonical_channel_ref": True,
    "require_guarded_launch": True,
    "require_canonical_artifact_owner": True,
    "forbid_new_runner_archive_writes": True,
    "fail_closed_on_missing_core": True,
}


class ChannelError(ValueError):
    """Raised when the formal execution channel is missing or bypassed."""


@dataclass(frozen=True)
class ArchiveWrite:
    line: int
    kind: str


def load_yaml_mapping(path: Path) -> dict[str, Any]:
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise ChannelError(f"Could not read {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ChannelError(f"{path} must contain one YAML mapping")
    return payload


def safe_repo_path(
    repo_root: Path, value: str, label: str, *, must_exist: bool = True
) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ChannelError(f"{label} must be a non-empty repository-relative path")
    path = Path(value)
    if path.is_absolute() or ".." in path.parts or not path.parts:
        raise ChannelError(
            f"{label} must be a safe repository-relative path: {value!r}"
        )
    resolved = repo_root / path
    if must_exist and not resolved.is_file():
        raise ChannelError(f"{label} does not exist as a file: {value}")
    return resolved


def _literal_string(node: ast.AST | None) -> str | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return None


def _call_name(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        base = _call_name(node.value)
        return f"{base}.{node.attr}" if base else node.attr
    return None


def _mode_from_call(node: ast.Call, positional_index: int, default: str) -> str:
    for keyword in node.keywords:
        if keyword.arg == "mode":
            return _literal_string(keyword.value) or "<dynamic>"
    if len(node.args) > positional_index:
        return _literal_string(node.args[positional_index]) or "<dynamic>"
    return default


def _archive_command_name(node: ast.AST) -> str | None:
    if not isinstance(node, (ast.List, ast.Tuple)) or not node.elts:
        return None
    first = _literal_string(node.elts[0])
    if first is None:
        return None
    return Path(first).name


def detect_archive_writes(path: Path) -> list[ArchiveWrite]:
    """Return direct archive-writing operations in one Python entrypoint.

    Reading ZIP/TAR files is allowed. Dynamic archive modes are rejected because
    a formal entrypoint must not hide package ownership behind runtime values.
    """

    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except (OSError, SyntaxError) as exc:
        raise ChannelError(f"Could not parse formal entrypoint {path}: {exc}") from exc

    findings: list[ArchiveWrite] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        name = _call_name(node.func)
        if name in {"zipfile.ZipFile", "ZipFile"}:
            mode = _mode_from_call(node, 1, "r")
            if mode == "<dynamic>" or any(flag in mode for flag in "wax"):
                findings.append(ArchiveWrite(node.lineno, f"{name}(mode={mode})"))
        elif name in {"shutil.make_archive", "make_archive"}:
            findings.append(ArchiveWrite(node.lineno, name))
        elif name in {"tarfile.open", "TarFile.open"}:
            mode = _mode_from_call(node, 1, "r")
            if mode == "<dynamic>" or any(flag in mode for flag in "wax"):
                findings.append(ArchiveWrite(node.lineno, f"{name}(mode={mode})"))
        elif (
            name
            in {
                "subprocess.run",
                "subprocess.Popen",
                "subprocess.check_call",
                "subprocess.check_output",
            }
            and node.args
        ):
            command = _archive_command_name(node.args[0])
            if command in {"zip", "tar", "7z"}:
                findings.append(ArchiveWrite(node.lineno, f"{name}({command})"))
    return findings


def _require_executable(path: Path, label: str) -> None:
    if not (path.stat().st_mode & stat.S_IXUSR):
        raise ChannelError(f"{label} must preserve executable mode: {path}")


def validate_channel_definition(
    repo_root: Path, registry: dict[str, Any]
) -> dict[str, Any]:
    channel = registry.get("formal_execution_channel")
    if not isinstance(channel, dict):
        raise ChannelError("registry must define formal_execution_channel")
    if channel.get("schema_version") != 1:
        raise ChannelError("formal_execution_channel.schema_version must be 1")
    if channel.get("policy_id") != POLICY_ID:
        raise ChannelError(f"formal_execution_channel.policy_id must be {POLICY_ID}")
    if channel.get("channel_id") != CHANNEL_ID:
        raise ChannelError(f"formal_execution_channel.channel_id must be {CHANNEL_ID}")
    base = channel.get("registration_base_commit")
    if not isinstance(base, str) or not SHA_RE.fullmatch(base):
        raise ChannelError(
            "formal_execution_channel.registration_base_commit must be a full SHA"
        )

    canonical_paths: dict[str, Path] = {}
    for field, expected in CANONICAL_FIELDS.items():
        if channel.get(field) != expected:
            raise ChannelError(f"formal_execution_channel.{field} must be {expected}")
        canonical_paths[field] = safe_repo_path(
            repo_root, expected, f"canonical {field}"
        )

    for field, import_text in WRAPPER_IMPORTS.items():
        wrapper = canonical_paths[field]
        _require_executable(wrapper, field)
        content = wrapper.read_text(encoding="utf-8")
        if import_text not in content:
            raise ChannelError(
                f"{field} must import the shared hardened core via {import_text!r}"
            )
        if "ModuleNotFoundError" not in content or "SystemExit(2)" not in content:
            raise ChannelError(
                f"{field} must fail closed when the hardened core is missing"
            )

    classes = channel.get("allowed_execution_classes")
    if set(classes or []) != ALLOWED_EXECUTION_CLASSES:
        raise ChannelError(
            "formal_execution_channel.allowed_execution_classes must exactly match "
            f"{sorted(ALLOWED_EXECUTION_CLASSES)}"
        )

    defaults = channel.get("default_artifact_budget")
    if not isinstance(defaults, dict):
        raise ChannelError(
            "formal_execution_channel.default_artifact_budget is required"
        )
    if defaults.get("main_package_hard_limit_mib") != 25:
        raise ChannelError("default main package limit must remain 25 MiB")
    if defaults.get("single_file_main_limit_mib") != 10:
        raise ChannelError("default single-file main package limit must remain 10 MiB")
    if defaults.get("large_file_storage") != "persistent_local_index":
        raise ChannelError(
            "default large-file storage must remain persistent_local_index"
        )
    if defaults.get("sidecar_default_enabled") is not False:
        raise ChannelError("sidecars must remain disabled by default")

    requirements = channel.get("formal_requirements")
    if requirements != REQUIRED_FORMAL_REQUIREMENTS:
        raise ChannelError(
            "formal_execution_channel.formal_requirements must exactly preserve the "
            "registered fail-closed requirements"
        )

    exceptions = channel.get("legacy_runner_archive_exceptions", [])
    if not isinstance(exceptions, list):
        raise ChannelError("legacy_runner_archive_exceptions must be a list")
    by_id: dict[str, dict[str, Any]] = {}
    for item in exceptions:
        if not isinstance(item, dict):
            raise ChannelError("Each legacy archive exception must be a mapping")
        exception_id = item.get("exception_id")
        if not isinstance(exception_id, str) or not exception_id:
            raise ChannelError("legacy archive exception_id is required")
        if exception_id in by_id:
            raise ChannelError(f"Duplicate legacy archive exception: {exception_id}")
        if item.get("no_new_entrypoints") is not True:
            raise ChannelError(f"{exception_id}: no_new_entrypoints must be true")
        if item.get("scope") != "recovery_checkpoint_only":
            raise ChannelError(
                f"{exception_id}: scope must be recovery_checkpoint_only"
            )
        safe_repo_path(repo_root, item.get("entrypoint"), f"{exception_id} entrypoint")
        experiment_ids = item.get("allowed_experiment_ids")
        if not isinstance(experiment_ids, list) or not experiment_ids:
            raise ChannelError(
                f"{exception_id}: allowed_experiment_ids must be non-empty"
            )
        by_id[exception_id] = item

    return {
        "channel_id": CHANNEL_ID,
        "canonical_paths": {
            key: str(path.relative_to(repo_root))
            for key, path in canonical_paths.items()
        },
        "legacy_exceptions": by_id,
    }


def _declared_entrypoints(experiment: dict[str, Any]) -> set[str]:
    values: set[str] = set()
    for key in ("code_entrypoint", "one_click_entrypoint"):
        value = experiment.get(key)
        if isinstance(value, str) and value:
            values.add(value)
    shared = experiment.get("shared_implementation")
    if isinstance(shared, dict):
        value = shared.get("experiment_entrypoint")
        if isinstance(value, str) and value:
            values.add(value)
    return values


def _require_canonical_formal_fields(
    experiment_id: str, formal: dict[str, Any]
) -> None:
    if formal.get("channel_ref") != CHANNEL_ID:
        raise ChannelError(f"{experiment_id}: channel_ref must be {CHANNEL_ID}")
    if formal.get("artifact_owner") != "canonical_channel":
        raise ChannelError(f"{experiment_id}: artifact_owner must be canonical_channel")
    if formal.get("launch_mode") != "canonical_guard":
        raise ChannelError(f"{experiment_id}: launch_mode must be canonical_guard")
    if formal.get("activation_state") not in ALLOWED_ACTIVATION_STATES:
        raise ChannelError(f"{experiment_id}: invalid activation_state")
    if formal.get("entrypoint_status") not in ALLOWED_ENTRYPOINT_STATES:
        raise ChannelError(f"{experiment_id}: invalid entrypoint_status")
    for field, expected in CANONICAL_FIELDS.items():
        if field == "validator_entrypoint":
            continue
        if formal.get(field) != expected:
            raise ChannelError(
                f"{experiment_id}: formal_execution.{field} must be {expected}"
            )


def _validate_archive_policy(
    experiment_id: str,
    entrypoint: str,
    policy: Any,
    findings: list[ArchiveWrite],
    legacy_exceptions: dict[str, dict[str, Any]],
) -> None:
    if not isinstance(policy, dict) or policy.get("mode") not in ALLOWED_ARCHIVE_MODES:
        raise ChannelError(
            f"{experiment_id}: runner_archive_policy must declare forbid or legacy_exception"
        )
    mode = policy["mode"]
    if mode == "forbid":
        if findings:
            detail = ", ".join(f"line {item.line}: {item.kind}" for item in findings)
            raise ChannelError(
                f"{experiment_id}: formal entrypoint creates archives directly ({detail}); "
                "canonical packaging owns formal artifacts"
            )
        return

    exception_id = policy.get("exception_id")
    exception = legacy_exceptions.get(exception_id)
    if exception is None:
        raise ChannelError(
            f"{experiment_id}: unregistered legacy archive exception {exception_id!r}"
        )
    if exception.get("entrypoint") != entrypoint:
        raise ChannelError(f"{experiment_id}: legacy exception entrypoint mismatch")
    if experiment_id not in exception.get("allowed_experiment_ids", []):
        raise ChannelError(f"{experiment_id}: not allowed by legacy archive exception")
    if not findings:
        raise ChannelError(
            f"{experiment_id}: legacy archive exception is unnecessary; use mode=forbid"
        )


def validate_formal_experiment(
    repo_root: Path,
    experiment: dict[str, Any],
    legacy_exceptions: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    experiment_id = experiment["id"]
    formal = experiment.get("formal_execution")
    if not isinstance(formal, dict):
        raise ChannelError(
            f"{experiment_id}: formal experiment requires formal_execution"
        )
    _require_canonical_formal_fields(experiment_id, formal)

    entrypoint_status = formal["entrypoint_status"]
    activation_state = formal["activation_state"]
    entrypoint = formal.get("entrypoint")
    if entrypoint_status == "planned":
        if activation_state != "blocked":
            raise ChannelError(
                f"{experiment_id}: planned entrypoint must remain blocked"
            )
        if experiment.get("implementation_state") != "not_implemented":
            raise ChannelError(
                f"{experiment_id}: planned entrypoint requires implementation_state=not_implemented"
            )
        if entrypoint not in (None, ""):
            raise ChannelError(f"{experiment_id}: planned entrypoint must be null")
        return {"id": experiment_id, "class": "formal", "state": "planned_blocked"}

    if not isinstance(entrypoint, str):
        raise ChannelError(
            f"{experiment_id}: implemented formal entrypoint is required"
        )
    path = safe_repo_path(repo_root, entrypoint, f"{experiment_id} formal entrypoint")
    declared = _declared_entrypoints(experiment)
    if entrypoint not in declared:
        raise ChannelError(
            f"{experiment_id}: formal entrypoint {entrypoint} is not declared by the experiment"
        )

    if formal.get("inherit_default_artifact_budget") is not True and not isinstance(
        experiment.get("artifact_budget"), dict
    ):
        raise ChannelError(
            f"{experiment_id}: formal experiment must inherit or declare an artifact budget"
        )

    if activation_state == "active":
        launch = experiment.get("formal_launch_template")
        if (
            not isinstance(launch, str)
            or CANONICAL_FIELDS["guard_entrypoint"] not in launch
        ):
            raise ChannelError(
                f"{experiment_id}: active formal experiment needs a canonical formal_launch_template"
            )
        if f"--experiment-id {experiment_id}" not in " ".join(launch.split()):
            raise ChannelError(
                f"{experiment_id}: formal_launch_template must bind its experiment ID"
            )

    findings = detect_archive_writes(path)
    _validate_archive_policy(
        experiment_id,
        entrypoint,
        formal.get("runner_archive_policy"),
        findings,
        legacy_exceptions,
    )
    return {
        "id": experiment_id,
        "class": "formal",
        "state": activation_state,
        "entrypoint": entrypoint,
        "archive_writes": [item.__dict__ for item in findings],
        "archive_policy": formal["runner_archive_policy"]["mode"],
    }


def validate_experiments(
    repo_root: Path,
    registry: dict[str, Any],
    channel_report: dict[str, Any],
) -> list[dict[str, Any]]:
    experiments = registry.get("experiments")
    if not isinstance(experiments, list) or not experiments:
        raise ChannelError("registry experiments must be a non-empty list")
    reports: list[dict[str, Any]] = []
    seen: set[str] = set()
    for experiment in experiments:
        if not isinstance(experiment, dict):
            raise ChannelError("Every experiment must be a mapping")
        experiment_id = experiment.get("id")
        if not isinstance(experiment_id, str) or not EXPERIMENT_ID_RE.fullmatch(
            experiment_id
        ):
            raise ChannelError(f"Invalid experiment id: {experiment_id!r}")
        if experiment_id in seen:
            raise ChannelError(f"Duplicate experiment id: {experiment_id}")
        seen.add(experiment_id)

        execution_class = experiment.get("execution_class")
        if execution_class not in ALLOWED_EXECUTION_CLASSES:
            raise ChannelError(
                f"{experiment_id}: execution_class must be one of "
                f"{sorted(ALLOWED_EXECUTION_CLASSES)}"
            )
        if execution_class == "formal":
            reports.append(
                validate_formal_experiment(
                    repo_root,
                    experiment,
                    channel_report["legacy_exceptions"],
                )
            )
        elif execution_class == "historical_formal":
            historical = experiment.get("historical_formal_execution")
            if not isinstance(historical, dict):
                raise ChannelError(
                    f"{experiment_id}: historical_formal requires historical_formal_execution"
                )
            if historical.get("future_rerun_requires_channel") != CHANNEL_ID:
                raise ChannelError(
                    f"{experiment_id}: future rerun must require {CHANNEL_ID}"
                )
            if (experiment.get("execution") or {}).get("state") != "delivered":
                raise ChannelError(
                    f"{experiment_id}: historical formal entry must be delivered"
                )
            if experiment.get("preserved_history") is not True:
                raise ChannelError(
                    f"{experiment_id}: historical formal entry must preserve history"
                )
            reports.append(
                {
                    "id": experiment_id,
                    "class": execution_class,
                    "state": "grandfathered",
                }
            )
        elif execution_class == "superseded":
            if experiment.get("status") != "superseded":
                raise ChannelError(
                    f"{experiment_id}: superseded class requires superseded status"
                )
            reports.append(
                {"id": experiment_id, "class": execution_class, "state": "preserved"}
            )
        else:
            if "formal_execution" in experiment:
                raise ChannelError(
                    f"{experiment_id}: pilot must not masquerade as formal"
                )
            pilot = experiment.get("pilot_execution")
            if pilot is not None:
                if not isinstance(pilot, dict):
                    raise ChannelError(
                        f"{experiment_id}: pilot_execution must be a mapping"
                    )
                if pilot.get("channel_ref") != CHANNEL_ID:
                    raise ChannelError(
                        f"{experiment_id}: pilot channel_ref must be {CHANNEL_ID}"
                    )
                if pilot.get("launch_mode") != "guarded_orchestrator":
                    raise ChannelError(
                        f"{experiment_id}: pilot launch_mode must be guarded_orchestrator"
                    )
                if pilot.get("guard_required") is not True:
                    raise ChannelError(
                        f"{experiment_id}: guarded pilot must require the guard"
                    )
                safe_repo_path(
                    repo_root,
                    pilot.get("operator_entrypoint"),
                    f"{experiment_id} pilot operator",
                )
            reports.append(
                {"id": experiment_id, "class": execution_class, "state": "nonformal"}
            )
    return reports


def validate_registry(repo_root: Path, registry_path: Path) -> dict[str, Any]:
    repo_root = repo_root.resolve()
    registry = load_yaml_mapping(registry_path)
    channel = validate_channel_definition(repo_root, registry)
    experiment_reports = validate_experiments(repo_root, registry, channel)
    counts = {key: 0 for key in sorted(ALLOWED_EXECUTION_CLASSES)}
    for item in experiment_reports:
        counts[item["class"]] += 1
    return {
        "matched": True,
        "policy_id": POLICY_ID,
        "channel_id": CHANNEL_ID,
        "execution_class_counts": counts,
        "formal_experiments": [
            item["id"] for item in experiment_reports if item["class"] == "formal"
        ],
        "legacy_archive_exceptions": sorted(channel["legacy_exceptions"]),
        "experiment_reports": experiment_reports,
    }


def compact_summary(report: dict[str, Any], report_out: Path | None) -> str:
    counts = report["execution_class_counts"]
    lines = [
        "Formal execution channel validation: PASS",
        f"Channel: {report['channel_id']} ({report['policy_id']})",
        (
            "Experiments: "
            f"formal={counts['formal']}, historical_formal={counts['historical_formal']}, "
            f"pilot={counts['pilot']}, superseded={counts['superseded']}"
        ),
        f"Legacy runner archive exceptions: {len(report['legacy_archive_exceptions'])}",
    ]
    if report_out is not None:
        lines.append(f"Full report: {report_out}")
    return "\n".join(lines)


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--registry",
        type=Path,
        default=Path("experiments/registry.yaml"),
    )
    parser.add_argument("--report-out", type=Path)
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    registry_path = args.registry
    if not registry_path.is_absolute():
        registry_path = args.repo_root / registry_path
    report_out = args.report_out
    if report_out is not None and not report_out.is_absolute():
        report_out = args.repo_root / report_out
    try:
        report = validate_registry(args.repo_root, registry_path)
    except ChannelError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    rendered = json.dumps(report, indent=2, sort_keys=True)
    if report_out is not None:
        report_out.parent.mkdir(parents=True, exist_ok=True)
        report_out.write_text(rendered + "\n", encoding="utf-8")
    print(rendered if args.verbose else compact_summary(report, report_out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
