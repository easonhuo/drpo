from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import yaml

INITIATIVE = "PAPER-WRITING-LOGIC-FIRST-01"
LEVELS = {"wording", "paragraph", "section"}
ACTIONS = {"KEEP", "TRIM", "REVISE", "MOVE", "ADD"}
CLAIM_IMPACTS = {"none", "weaken", "strengthen"}
EXPECTED_ARTIFACTS = {
    "wording": ["paragraph_logic", "source_mapping", "candidate"],
    "paragraph": ["section_logic", "paragraph_logic", "source_mapping", "candidate"],
    "section": ["section_logic", "paragraph_logic", "source_mapping", "candidate"],
}


class GateError(RuntimeError):
    """Expected fail-closed validation error."""


def read_yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise GateError(f"expected YAML mapping: {path}")
    return value


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def inside(repo: Path, value: str) -> Path:
    path = (repo / value).resolve()
    try:
        path.relative_to(repo.resolve())
    except ValueError as exc:
        raise GateError(f"path escapes repository: {value}") from exc
    return path


def text(row: dict[str, Any], key: str, context: str) -> str:
    value = row.get(key)
    if not isinstance(value, str) or not value.strip():
        raise GateError(f"{context} requires non-empty string {key}")
    return value.strip()


def strings(row: dict[str, Any], key: str, context: str) -> list[str]:
    value = row.get(key)
    if not isinstance(value, list) or not value:
        raise GateError(f"{context} requires non-empty list {key}")
    result = [item.strip() for item in value if isinstance(item, str) and item.strip()]
    if len(result) != len(value) or len(result) != len(set(result)):
        raise GateError(f"{context} field {key} must contain unique non-empty strings")
    return result


def file_ref(repo: Path, row: Any, name: str) -> Path:
    if not isinstance(row, dict):
        raise GateError(f"manifest requires artifact reference {name}")
    path = inside(repo, text(row, "path", name))
    expected = text(row, "sha256", name)
    if not path.is_file():
        raise GateError(f"{name} artifact is missing: {path.relative_to(repo)}")
    actual = digest(path)
    if actual != expected:
        raise GateError(f"{name} artifact checksum mismatch: {actual} != {expected}")
    return path


def approval(row: dict[str, Any], context: str) -> None:
    value = row.get("approval")
    if not isinstance(value, dict):
        raise GateError(f"{context} requires approval metadata")
    text(value, "approved_by", f"{context}.approval")
    text(value, "approved_at", f"{context}.approval")


def load_policy(repo: Path, path: Path) -> dict[str, Any]:
    policy = read_yaml(path)
    if policy.get("schema_version") != 1 or policy.get("initiative") != INITIATIVE:
        raise GateError("paper logic gate policy identity mismatch")
    guidance = inside(repo, text(policy, "guidance_path", "policy"))
    playbook = inside(repo, text(policy, "playbook_path", "policy"))
    if not guidance.is_file() or not playbook.is_file():
        raise GateError("policy guidance/playbook source is missing")
    levels = policy.get("levels")
    if not isinstance(levels, dict) or set(levels) != LEVELS:
        raise GateError("policy must define wording, paragraph, and section exactly")
    guidance_text = guidance.read_text(encoding="utf-8")
    playbook_lines = {line.strip() for line in playbook.read_text(encoding="utf-8").splitlines()}
    for level, contract in levels.items():
        if not isinstance(contract, dict):
            raise GateError(f"policy level {level} must be a mapping")
        rules = strings(contract, "guidance_rules", f"policy level {level}")
        modules = strings(contract, "playbook_modules", f"policy level {level}")
        required = strings(contract, "required_artifacts", f"policy level {level}")
        if required != EXPECTED_ARTIFACTS[level]:
            raise GateError(
                f"policy level {level} required_artifacts changed: "
                f"{required} != {EXPECTED_ARTIFACTS[level]}"
            )
        missing_rules = [rule for rule in rules if f"### {rule}." not in guidance_text]
        missing_modules = [module for module in modules if module not in playbook_lines]
        if missing_rules or missing_modules:
            raise GateError(
                f"policy level {level} has missing references: "
                f"rules={missing_rules} modules={missing_modules}"
            )
    return policy
