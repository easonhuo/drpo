#!/usr/bin/env python3
"""Validate and route shared manuscript writing skills.

The library is intentionally report-only: it loads writing obligations for the
manuscript pipeline and interactive editor without changing scientific facts or
claim authority. Project-specific terminology and claim boundaries remain in the
project profile.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

import yaml


class SkillLibraryError(RuntimeError):
    pass


SKILL_FILES = {
    "core": "core_paper_writing_skills.yaml",
    "domain": "rl_paper_writing_skills.yaml",
}
ROUTER_FILE = "writing_task_router.yaml"
SCHEMA_FILE = "skill_schema.yaml"
EXPERIMENT_ID_RE = re.compile(r"\b[A-Z]-[A-Z0-9]+-E\d+(?:-[A-Z0-9]+)*\b")
DOMAIN_FORBIDDEN_EXAMPLES = ("C-U1", "D-U1", "Hopper", "Countdown")


def read_yaml(path: Path) -> dict[str, Any]:
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise SkillLibraryError(f"cannot read YAML {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise SkillLibraryError(f"YAML root must be a mapping: {path}")
    return payload


def _require_non_empty_string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise SkillLibraryError(f"{label} must be a non-empty string")
    return value.strip()


def _require_string_list(value: Any, label: str) -> list[str]:
    if not isinstance(value, list) or not value:
        raise SkillLibraryError(f"{label} must be a non-empty list")
    result: list[str] = []
    for item in value:
        if not isinstance(item, str) or not item.strip():
            raise SkillLibraryError(f"{label} contains an empty or non-string item")
        result.append(item.strip())
    return result


def _skill_text(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True).lower()


def load_project_leak_terms(project_profile: Path | None) -> list[str]:
    if project_profile is None or not project_profile.is_file():
        return []
    profile = read_yaml(project_profile)
    terms = profile.get("business_terms_for_core_leak_check", [])
    if not isinstance(terms, list):
        raise SkillLibraryError("project business_terms_for_core_leak_check must be a list")
    return [str(term).strip() for term in terms if str(term).strip()]


def load_schema(skills_root: Path) -> dict[str, Any]:
    schema = read_yaml(skills_root / SCHEMA_FILE)
    for key in ("required_fields", "allowed_scopes", "allowed_severities", "task_types"):
        _require_string_list(schema.get(key), f"schema {key}")
    return schema


def validate_skill(skill: dict[str, Any], *, expected_scope: str, schema: dict[str, Any]) -> None:
    required = _require_string_list(schema.get("required_fields"), "schema required_fields")
    for field in required:
        if field not in skill:
            raise SkillLibraryError(f"{skill.get('skill_id', '<missing-id>')}: missing {field}")
    skill_id = _require_non_empty_string(skill.get("skill_id"), "skill_id")
    scope = _require_non_empty_string(skill.get("scope"), f"{skill_id}.scope")
    if scope != expected_scope:
        raise SkillLibraryError(f"{skill_id}: scope {scope!r} != file scope {expected_scope!r}")
    if scope == "core" and not skill_id.startswith("core."):
        raise SkillLibraryError(f"{skill_id}: core skill_id must start with core.")
    if scope == "domain" and not skill_id.startswith(("domain.", "rl.", "ml.")):
        raise SkillLibraryError(
            f"{skill_id}: domain skill_id must start with domain., rl., or ml."
        )
    allowed_scopes = set(_require_string_list(schema.get("allowed_scopes"), "schema scopes"))
    if scope not in allowed_scopes:
        raise SkillLibraryError(f"{skill_id}: unknown scope {scope}")
    allowed_tasks = set(_require_string_list(schema.get("task_types"), "schema task_types"))
    for task in _require_string_list(skill.get("applies_to"), f"{skill_id}.applies_to"):
        if task not in allowed_tasks:
            raise SkillLibraryError(f"{skill_id}: unknown applies_to task {task}")
    rule = skill.get("rule")
    if not isinstance(rule, dict):
        raise SkillLibraryError(f"{skill_id}: rule must be a mapping")
    _require_non_empty_string(rule.get("summary"), f"{skill_id}.rule.summary")
    _require_string_list(skill.get("required_steps"), f"{skill_id}.required_steps")
    _require_string_list(skill.get("checks"), f"{skill_id}.checks")
    _require_string_list(skill.get("failure_modes"), f"{skill_id}.failure_modes")
    severity = _require_non_empty_string(skill.get("severity"), f"{skill_id}.severity")
    allowed_severities = set(
        _require_string_list(schema.get("allowed_severities"), "schema severities")
    )
    if severity not in allowed_severities:
        raise SkillLibraryError(f"{skill_id}: unknown severity {severity}")


def load_skill_files(skills_root: Path, schema: dict[str, Any]) -> dict[str, dict[str, Any]]:
    skills_by_id: dict[str, dict[str, Any]] = {}
    for scope, filename in SKILL_FILES.items():
        payload = read_yaml(skills_root / filename)
        if payload.get("scope") != scope:
            raise SkillLibraryError(f"{filename}: top-level scope must be {scope}")
        skills = payload.get("skills")
        if not isinstance(skills, list) or not skills:
            raise SkillLibraryError(f"{filename}: skills must be a non-empty list")
        for skill in skills:
            if not isinstance(skill, dict):
                raise SkillLibraryError(f"{filename}: skill entries must be mappings")
            validate_skill(skill, expected_scope=scope, schema=schema)
            skill_id = str(skill["skill_id"])
            if skill_id in skills_by_id:
                raise SkillLibraryError(f"duplicate skill_id: {skill_id}")
            skills_by_id[skill_id] = skill
    return skills_by_id


def validate_core_leakage(skills_by_id: dict[str, dict[str, Any]], leak_terms: list[str]) -> None:
    core_text = "\n".join(
        _skill_text(skill) for skill in skills_by_id.values() if skill.get("scope") == "core"
    )
    leaked = [term for term in leak_terms if term.lower() in core_text]
    if leaked:
        raise SkillLibraryError("core skills contain project-specific terms: " + ", ".join(leaked))


def validate_domain_genericity(skills_by_id: dict[str, dict[str, Any]]) -> None:
    domain_text = "\n".join(
        _skill_text(skill) for skill in skills_by_id.values() if skill.get("scope") == "domain"
    )
    matches = sorted(set(EXPERIMENT_ID_RE.findall(domain_text.upper())))
    leaked_examples = [term for term in DOMAIN_FORBIDDEN_EXAMPLES if term.lower() in domain_text]
    if matches or leaked_examples:
        raise SkillLibraryError(
            "domain skills contain project/example identifiers: "
            + ", ".join(matches + leaked_examples)
        )


def load_router(
    skills_root: Path,
    schema: dict[str, Any],
    skills_by_id: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    router = read_yaml(skills_root / ROUTER_FILE)
    routing = router.get("routing")
    if not isinstance(routing, dict) or not routing:
        raise SkillLibraryError("router routing must be a non-empty mapping")
    allowed_tasks = set(_require_string_list(schema.get("task_types"), "schema task_types"))
    for task, config in routing.items():
        if task not in allowed_tasks:
            raise SkillLibraryError(f"router references unknown task type: {task}")
        if not isinstance(config, dict):
            raise SkillLibraryError(f"router task {task} must be a mapping")
        for scope in ("core", "domain"):
            ids = config.get(scope, [])
            if ids is None:
                ids = []
            if not isinstance(ids, list) or not all(isinstance(item, str) for item in ids):
                raise SkillLibraryError(f"router {task}.{scope} must be a list of skill ids")
            for skill_id in ids:
                if skill_id not in skills_by_id:
                    raise SkillLibraryError(f"router {task} references missing skill {skill_id}")
                if skills_by_id[skill_id].get("scope") != scope:
                    raise SkillLibraryError(f"router {task} places {skill_id} under wrong scope")
        obligations = config.get("project_obligations", [])
        if obligations is None:
            obligations = []
        if not isinstance(obligations, list) or not all(
            isinstance(item, str) for item in obligations
        ):
            raise SkillLibraryError(f"router {task}.project_obligations must be a list")
    return router


def validate_library(skills_root: Path, project_profile: Path | None = None) -> dict[str, Any]:
    if not skills_root.is_dir():
        raise SkillLibraryError(f"skills root is missing: {skills_root}")
    schema = load_schema(skills_root)
    skills_by_id = load_skill_files(skills_root, schema)
    validate_core_leakage(skills_by_id, load_project_leak_terms(project_profile))
    validate_domain_genericity(skills_by_id)
    router = load_router(skills_root, schema, skills_by_id)
    return {
        "schema_version": 1,
        "status": "PASS",
        "skills_root": str(skills_root),
        "skill_count": len(skills_by_id),
        "core_skill_count": sum(
            1 for skill in skills_by_id.values() if skill.get("scope") == "core"
        ),
        "domain_skill_count": sum(
            1 for skill in skills_by_id.values() if skill.get("scope") == "domain"
        ),
        "task_count": len(router["routing"]),
        "tasks": sorted(router["routing"].keys()),
    }


def _compact_skill(skill: dict[str, Any]) -> dict[str, Any]:
    return {
        "skill_id": skill["skill_id"],
        "name": skill["name"],
        "scope": skill["scope"],
        "severity": skill["severity"],
        "summary": skill["rule"]["summary"],
        "required_steps": skill["required_steps"],
        "checks": skill["checks"],
    }


def build_obligations_report(
    skills_root: Path,
    project_profile: Path | None = None,
    task_types: list[str] | None = None,
) -> dict[str, Any]:
    schema = load_schema(skills_root)
    skills_by_id = load_skill_files(skills_root, schema)
    validate_core_leakage(skills_by_id, load_project_leak_terms(project_profile))
    validate_domain_genericity(skills_by_id)
    router = load_router(skills_root, schema, skills_by_id)
    requested = task_types or sorted(router["routing"].keys())
    obligations: dict[str, Any] = {}
    for task in requested:
        if task not in router["routing"]:
            raise SkillLibraryError(f"unknown task type requested: {task}")
        config = router["routing"][task]
        obligations[task] = {
            "core": [_compact_skill(skills_by_id[skill_id]) for skill_id in config.get("core", [])],
            "domain": [
                _compact_skill(skills_by_id[skill_id]) for skill_id in config.get("domain", [])
            ],
            "project_obligations": list(config.get("project_obligations", [])),
        }
    return {
        "schema_version": 1,
        "status": "PASS",
        "mode": "report_only",
        "skills_root": str(skills_root),
        "task_types": requested,
        "obligations_by_task": obligations,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("validate", "obligations"))
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--skills-root", type=Path, default=Path("docs/manuscript/skills"))
    parser.add_argument(
        "--project-profile", type=Path, default=Path("docs/manuscript/projects/drpo_profile.yaml")
    )
    parser.add_argument("--task-type", action="append", dest="task_types")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    repo = args.repo_root.resolve()
    skills_root = (repo / args.skills_root).resolve()
    project_profile = (repo / args.project_profile).resolve() if args.project_profile else None
    try:
        if args.command == "validate":
            result = validate_library(skills_root, project_profile)
        else:
            result = build_obligations_report(skills_root, project_profile, args.task_types)
        print(json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True))
        return 0
    except SkillLibraryError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
