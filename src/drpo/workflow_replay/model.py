"""Strict, side-effect-free replay-case manifest validation."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Any, Mapping

import yaml

SHA40 = re.compile(r"[0-9a-f]{40}")
SHA256 = re.compile(r"[0-9a-f]{64}")
CASE_ID = re.compile(r"[A-Z0-9][A-Z0-9._-]{2,127}")
HASH_KEY = re.compile(r"[a-z][a-z0-9_.-]{1,63}")
TASK_CLASSES = frozenset(
    {"code_only", "add_registration", "replace_registration", "result_closure", "stale_recovery", "gate_failure"}
)
FIELDS = {
    "manifest": {"schema_version", "case_id", "task_class", "historical_task", "benchmark"},
    "historical_task": {
        "base_sha", "frozen_implementation_sha", "source_prs", "source_commits",
        "historical_real_time_evidence",
    },
    "benchmark": {
        "toolchain_sha", "input_spec_sha256", "expected_terminal_state",
        "expected_safety_boundary", "expected_changed_paths",
        "expected_final_tree_or_semantic_hashes", "required_gates", "environment_id",
        "cache_policy", "replayability", "predeclared_exclusions",
    },
}


class ManifestError(ValueError):
    """The replay-case contract is unsafe or ambiguous."""


@dataclass(frozen=True)
class CaseManifest:
    """Immutable validated case contract."""

    data: Mapping[str, Any]

    @property
    def case_id(self) -> str:
        return self.data["case_id"]

    @property
    def task_class(self) -> str:
        return self.data["task_class"]

    @property
    def historical_task(self) -> Mapping[str, Any]:
        return self.data["historical_task"]

    @property
    def benchmark(self) -> Mapping[str, Any]:
        return self.data["benchmark"]


def _freeze(value: Any) -> Any:
    if isinstance(value, dict):
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_freeze(item) for item in value)
    return value


def _strict_map(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ManifestError(f"{label} must be a mapping")
    allowed = FIELDS[label]
    if unknown := sorted(set(value) - allowed):
        raise ManifestError(f"{label} has unknown keys: {', '.join(unknown)}")
    if missing := sorted(allowed - set(value)):
        raise ManifestError(f"{label} is missing keys: {', '.join(missing)}")
    return value


def _text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ManifestError(f"{label} must be a non-empty trimmed string")
    return value


def _choice(value: Any, label: str, allowed: set[str] | frozenset[str]) -> str:
    value = _text(value, label)
    if value not in allowed:
        raise ManifestError(f"{label} must be one of {sorted(allowed)}")
    return value


def _digest(value: Any, label: str, pattern: re.Pattern[str]) -> str:
    value = _text(value, label)
    if pattern.fullmatch(value) is None:
        raise ManifestError(f"{label} has invalid hash syntax")
    return value


def _string_list(value: Any, label: str, required: bool = False) -> tuple[str, ...]:
    if not isinstance(value, list) or (required and not value):
        qualifier = "non-empty " if required else ""
        raise ManifestError(f"{label} must be a {qualifier}list")
    result = tuple(_text(item, f"{label}[]") for item in value)
    if len(result) != len(set(result)):
        raise ManifestError(f"{label} contains duplicates")
    return result


def validate_case_manifest(payload: Any) -> CaseManifest:
    """Validate parsed YAML without running commands or writing repository state."""
    root = _strict_map(payload, "manifest")
    if isinstance(root["schema_version"], bool) or root["schema_version"] != 1:
        raise ManifestError("schema_version must equal integer 1")
    if CASE_ID.fullmatch(_text(root["case_id"], "case_id")) is None:
        raise ManifestError("case_id has invalid syntax")
    _choice(root["task_class"], "task_class", TASK_CLASSES)

    historical = _strict_map(root["historical_task"], "historical_task")
    _digest(historical["base_sha"], "historical_task.base_sha", SHA40)
    if historical["frozen_implementation_sha"] is not None:
        _digest(
            historical["frozen_implementation_sha"],
            "historical_task.frozen_implementation_sha",
            SHA40,
        )
    prs = historical["source_prs"]
    if not isinstance(prs, list) or any(
        isinstance(pr, bool) or not isinstance(pr, int) or pr <= 0 for pr in prs
    ) or len(prs) != len(set(prs)):
        raise ManifestError("historical_task.source_prs must contain unique positive integers")
    for commit in _string_list(historical["source_commits"], "historical_task.source_commits", True):
        _digest(commit, "historical_task.source_commits[]", SHA40)
    _string_list(historical["historical_real_time_evidence"], "historical_task.historical_real_time_evidence")

    benchmark = _strict_map(root["benchmark"], "benchmark")
    _digest(benchmark["toolchain_sha"], "benchmark.toolchain_sha", SHA40)
    _digest(benchmark["input_spec_sha256"], "benchmark.input_spec_sha256", SHA256)
    terminal = _choice(
        benchmark["expected_terminal_state"], "benchmark.expected_terminal_state",
        {"READY", "BLOCKED", "STALE"},
    )
    boundary = benchmark["expected_safety_boundary"]
    if boundary is not None:
        _text(boundary, "benchmark.expected_safety_boundary")
    paths = _string_list(benchmark["expected_changed_paths"], "benchmark.expected_changed_paths")
    for raw in paths:
        path = PurePosixPath(raw)
        if raw.startswith(("/", "-")) or "\\" in raw or path.as_posix() != raw or ".." in path.parts:
            raise ManifestError(f"unsafe repository path: {raw}")

    hashes = benchmark["expected_final_tree_or_semantic_hashes"]
    if not isinstance(hashes, dict):
        raise ManifestError("benchmark.expected_final_tree_or_semantic_hashes must be a mapping")
    for key, value in hashes.items():
        if not isinstance(key, str) or HASH_KEY.fullmatch(key) is None:
            raise ManifestError("benchmark outcome hash key has invalid syntax")
        _digest(value, f"benchmark outcome hash {key}", SHA40 if key.endswith("tree_sha") else SHA256)
    if terminal == "READY" and (boundary is not None or not paths or not hashes):
        raise ManifestError("READY outcome requires changed paths and hashes, with no safety boundary")
    if terminal != "READY" and (boundary is None or paths or hashes):
        raise ManifestError("BLOCKED/STALE outcome requires a safety boundary and no repository outcome")

    _string_list(benchmark["required_gates"], "benchmark.required_gates", True)
    _text(benchmark["environment_id"], "benchmark.environment_id")
    _choice(benchmark["cache_policy"], "benchmark.cache_policy", {"cold", "fixed_warm"})
    replayability = _choice(
        benchmark["replayability"], "benchmark.replayability",
        {"complete", "reconstructed", "partial"},
    )
    exclusions = _string_list(benchmark["predeclared_exclusions"], "benchmark.predeclared_exclusions")
    if exclusions and replayability != "partial":
        raise ManifestError("predeclared exclusions require partial replayability")
    return CaseManifest(_freeze(root))


def load_case_manifest(path: str | Path) -> CaseManifest:
    """Load one UTF-8 YAML contract without following path symlinks."""
    manifest_path = Path(path)
    if manifest_path.is_symlink() or any(parent.is_symlink() for parent in manifest_path.parents):
        raise ManifestError("manifest path must not contain symlinks")
    if not manifest_path.is_file():
        raise ManifestError("manifest path must be a regular file")
    try:
        return validate_case_manifest(yaml.safe_load(manifest_path.read_text(encoding="utf-8")))
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        raise ManifestError(f"cannot read manifest: {exc}") from exc
