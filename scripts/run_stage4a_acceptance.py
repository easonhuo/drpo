#!/usr/bin/env python3
"""Run the fail-closed Stage 4A final acceptance and write auditable evidence.

The runner validates the existing Stage 4A static inventory, dynamic semantic
shadow graph, and minimal-context core.  It does not create Stage 4B output,
change research authority, or authorize Stage 4B implementation.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Callable

import yaml

POLICY_ID = "GOV-HANDOFF-INDEX-01"
BASE_COMMIT = "9674cb167080dfdeecb353c9f328ad86b74f87c5"
AUTHORITY = "shadow_only"
OUTPUT_REL = Path("docs/governance_stage4a_acceptance")

EXPECTED_CONTRACTS = {
    "global_core_governance": {
        "unique_master_document",
        "document_before_experiment",
        "non_destructive_history",
        "terminal_audit_governance",
        "controlled_external_validity_boundary",
    },
    "execution_status_gates": {
        "formal_vs_development_evidence",
        "single_registered_execution_order",
        "blocked_requires_protocol_or_predecessor",
        "current_formal_route",
        "no_unregistered_experiment",
    },
    "terminal_audit": {
        "convergence_or_persistent_drift",
        "two_x_continuation",
        "false_plateau_checks",
        "task_performance_collapse",
        "support_or_variance_boundary",
        "nan_inf_numerical_failure",
        "separate_failure_reporting",
    },
}
EXPECTED_TARGETS = {
    "continuous_e4_extrapolation",
    "continuous_e4_taper",
    "categorical_e6_generalization",
    "hopper_e7",
    "countdown_e8",
    "paper_rewrite",
}
FORBIDDEN_SUGGESTIONS = {
    "candidate_add_or_map_module",
    "candidate_map_development_formal_registration",
    "candidate_map_development_registration",
}
AUTHORITATIVE_INPUTS = (
    "docs/handoff.md",
    "experiments/registry.yaml",
    "AGENTS.md",
)
AFTER_IMAGE_PATHS = (
    "docs/governance_stage4a_acceptance_spec.md",
    "docs/governance_stage4_semantic_context_spec.md",
    "docs/handoff_shadow/stage4",
    "scripts/build_stage4_context.py",
    "scripts/build_stage4_semantic_graph.py",
    "scripts/validate_stage4_context.py",
    "scripts/validate_stage4_semantic_graph.py",
    "scripts/validate_stage4a_inventory.py",
    "scripts/run_stage4a_acceptance.py",
    "tests/test_stage4_context.py",
    "tests/test_stage4_semantic_graph.py",
    "tests/test_stage4a_inventory.py",
    "tests/test_stage4a_acceptance.py",
)


class AcceptanceError(ValueError):
    """Raised when a Stage 4A hard acceptance condition fails."""


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json(payload: Any) -> bytes:
    return (json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")


def load_python(path: Path, name: str):
    unique_name = f"{name}_{hashlib.sha1(str(path).encode()).hexdigest()[:12]}_{os.getpid()}"
    spec = importlib.util.spec_from_file_location(unique_name, path)
    if spec is None or spec.loader is None:
        raise AcceptanceError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[unique_name] = module
    spec.loader.exec_module(module)
    return module


def load_yaml(path: Path) -> dict[str, Any]:
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise AcceptanceError(f"cannot read YAML {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise AcceptanceError(f"YAML root must be a mapping: {path}")
    return payload


def write_yaml(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(yaml.safe_dump(payload, sort_keys=False, allow_unicode=True), encoding="utf-8")


def snapshot_files(root: Path, relative_root: Path) -> dict[str, str]:
    base = root / relative_root
    if not base.is_dir():
        raise AcceptanceError(f"missing generated directory: {relative_root}")
    return {
        path.relative_to(root).as_posix(): sha256_path(path)
        for path in sorted(base.rglob("*"))
        if path.is_file() and not path.is_symlink()
    }


def copy_repository(source: Path, destination: Path) -> None:
    shutil.copytree(
        source,
        destination,
        ignore=shutil.ignore_patterns(
            ".git", "__pycache__", ".pytest_cache", ".ruff_cache", "*.pyc",
            "outputs", "wandb", ".venv", "venv"
        ),
    )


def subprocess_json(command: list[str], cwd: Path) -> dict[str, Any]:
    proc = subprocess.run(command, cwd=cwd, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    if proc.returncode != 0:
        raise AcceptanceError(
            f"command failed ({proc.returncode}): {' '.join(command)}\n"
            f"stdout={proc.stdout[-4000:]}\nstderr={proc.stderr[-4000:]}"
        )
    try:
        payload = json.loads(proc.stdout)
    except json.JSONDecodeError as exc:
        raise AcceptanceError(f"command did not emit JSON: {' '.join(command)}") from exc
    if not isinstance(payload, dict) or payload.get("status") not in {None, "PASS"}:
        raise AcceptanceError(f"command report is not PASS: {' '.join(command)}")
    return payload


def run_core_acceptance(repo_root: Path, *, check_determinism: bool) -> dict[str, Any]:
    repo_root = repo_root.resolve()
    for value in AUTHORITATIVE_INPUTS:
        if not (repo_root / value).is_file():
            raise AcceptanceError(f"authoritative input missing: {value}")

    inventory = subprocess_json(
        [sys.executable, "scripts/validate_stage4a_inventory.py", "--repo-root", str(repo_root), "--json"],
        repo_root,
    )
    semantic = subprocess_json(
        [sys.executable, "scripts/validate_stage4_semantic_graph.py", "--repo-root", str(repo_root), "--json"],
        repo_root,
    )
    context = subprocess_json(
        [sys.executable, "scripts/validate_stage4_context.py", "--repo-root", str(repo_root), "--json"],
        repo_root,
    )

    if inventory.get("authority_cutover_allowed") is not False or inventory.get("manual_handoff_remains_authoritative") is not True:
        raise AcceptanceError("Stage 4A inventory violated the shadow authority boundary")
    if semantic.get("review_queue") != 0 or semantic.get("rejected_candidates") != 0:
        raise AcceptanceError("dynamic semantic graph has unresolved or rejected candidates")
    if semantic.get("authority_cutover_allowed") is not False or semantic.get("manual_handoff_remains_authoritative") is not True:
        raise AcceptanceError("dynamic semantic graph violated the shadow authority boundary")
    if context.get("authority_cutover_allowed") is not False or context.get("manual_handoff_remains_authoritative") is not True:
        raise AcceptanceError("minimal context violated the shadow authority boundary")

    builder = load_python(repo_root / "scripts/build_stage4_context.py", "stage4_acceptance_builder")
    try:
        plan = builder.build_plan(repo_root)
    except Exception as exc:  # builder owns a purpose-specific error type
        raise AcceptanceError(str(exc)) from exc

    contract_report: dict[str, Any] = {}
    for module_id, expected_topics in EXPECTED_CONTRACTS.items():
        snapshot = plan.snapshots.get(module_id)
        if snapshot is None:
            raise AcceptanceError(f"required semantic-contract module missing: {module_id}")
        actual = set(snapshot.contract_topics)
        if actual != expected_topics:
            raise AcceptanceError(
                f"semantic contract {module_id} topic mismatch; "
                f"missing={sorted(expected_topics - actual)}, extra={sorted(actual - expected_topics)}"
            )
        evidence_topics = {item["topic_id"] for item in snapshot.contract_evidence}
        if evidence_topics != expected_topics:
            raise AcceptanceError(
                f"semantic contract {module_id} evidence mismatch; "
                f"missing={sorted(expected_topics - evidence_topics)}, extra={sorted(evidence_topics - expected_topics)}"
            )
        contract_report[module_id] = {
            "required_topics": len(expected_topics),
            "satisfied_topics": len(evidence_topics),
            "topic_ids": sorted(expected_topics),
            "evidence": list(snapshot.contract_evidence),
        }

    suggestion_kinds = {item["kind"] for item in plan.suggestions}
    forbidden = sorted(suggestion_kinds & FORBIDDEN_SUGGESTIONS)
    if forbidden:
        raise AcceptanceError(f"unmapped registry objects remain: {forbidden}")
    for item in plan.suggestions:
        if item.get("automatic_action") is not False:
            raise AcceptanceError("structure suggestion attempted an automatic action")

    targets = {item["target"] for item in plan.acceptance_results}
    if targets != EXPECTED_TARGETS:
        raise AcceptanceError(
            f"acceptance target set mismatch; missing={sorted(EXPECTED_TARGETS-targets)}, "
            f"extra={sorted(targets-EXPECTED_TARGETS)}"
        )
    if any(item.get("status") != "pass" for item in plan.acceptance_results):
        raise AcceptanceError("one or more dependency-closure acceptance targets failed")

    generated_dir = Path("docs/handoff_shadow/stage4/minimal/generated")
    generated_before = snapshot_files(repo_root, generated_dir)
    authority_before = {value: sha256_path(repo_root / value) for value in AUTHORITATIVE_INPUTS}
    determinism: dict[str, Any] = {"checked": check_determinism}
    if check_determinism:
        with tempfile.TemporaryDirectory(prefix="drpo-stage4a-determinism-") as temp:
            candidate = Path(temp) / "repo"
            copy_repository(repo_root, candidate)
            first = subprocess_json(
                [sys.executable, "scripts/build_stage4_context.py", "--repo-root", str(candidate), "--json", "build"],
                candidate,
            )
            bytes_after_first = snapshot_files(candidate, generated_dir)
            second = subprocess_json(
                [sys.executable, "scripts/build_stage4_context.py", "--repo-root", str(candidate), "--json", "build"],
                candidate,
            )
            bytes_after_second = snapshot_files(candidate, generated_dir)
            if bytes_after_first != bytes_after_second or bytes_after_second != generated_before:
                raise AcceptanceError("repeated context builds were not byte deterministic")
            if second.get("refreshed_modules") or second.get("refreshed_supporting_files") or second.get("removed_stale_files"):
                raise AcceptanceError("second no-op build did not fully reuse generated state")
            if sorted(second.get("reused_modules", [])) != sorted(plan.module_order):
                raise AcceptanceError("second no-op build did not reuse every module")
            authority_after = {value: sha256_path(candidate / value) for value in AUTHORITATIVE_INPUTS}
            if authority_before != authority_after:
                raise AcceptanceError("context build modified an authoritative input")
            determinism = {
                "checked": True,
                "first_graph_hash": first.get("graph_hash"),
                "second_graph_hash": second.get("graph_hash"),
                "byte_identical": True,
                "second_noop_reused_all_modules": True,
                "authoritative_inputs_unchanged": True,
            }

    return {
        "inventory": inventory,
        "semantic_graph": semantic,
        "minimal_context": context,
        "semantic_contracts": contract_report,
        "acceptance_targets": list(plan.acceptance_results),
        "suggestions": list(plan.suggestions),
        "mapping": {
            "unmapped_objects": [],
            "canonical_experiments": inventory.get("experiment_count"),
            "development_registrations": len(
                load_yaml(repo_root / "experiments/registry.yaml").get("development_experiment_registrations", [])
            ),
        },
        "determinism": determinism,
        "generated_file_count": len(generated_before),
    }


def _module(payload: dict[str, Any], module_id: str) -> dict[str, Any]:
    for item in payload["modules"]:
        if item.get("module_id") == module_id:
            return item
    raise AcceptanceError(f"test fixture module missing: {module_id}")


def _targeted_context_check(repo: Path) -> Any:
    plan = _run_build(repo)
    for module_id, expected_topics in EXPECTED_CONTRACTS.items():
        snapshot = plan.snapshots.get(module_id)
        if snapshot is None or set(snapshot.contract_topics) != expected_topics:
            actual = set() if snapshot is None else set(snapshot.contract_topics)
            raise AcceptanceError(
                f"semantic contract {module_id} topic mismatch; "
                f"missing={sorted(expected_topics-actual)}, extra={sorted(actual-expected_topics)}"
            )
        if {item["topic_id"] for item in snapshot.contract_evidence} != expected_topics:
            raise AcceptanceError(f"semantic contract {module_id} source-scoped evidence mismatch")
    forbidden = sorted({item["kind"] for item in plan.suggestions} & FORBIDDEN_SUGGESTIONS)
    if forbidden:
        raise AcceptanceError(f"unmapped registry objects remain: {forbidden}")
    targets = {item["target"] for item in plan.acceptance_results}
    if targets != EXPECTED_TARGETS:
        raise AcceptanceError("acceptance target set mismatch")
    return plan


def _expect_failure(repo: Path) -> str:
    try:
        _targeted_context_check(repo)
    except Exception as exc:
        return str(exc)
    raise AcceptanceError("fault injection unexpectedly passed")


def _run_build(repo: Path) -> Any:
    builder = load_python(repo / "scripts/build_stage4_context.py", "stage4_fault_builder")
    return builder.build_plan(repo)


def run_fault_injections(repo_root: Path) -> dict[str, Any]:
    cases: list[tuple[str, str, Callable[[Path], str]]] = []

    def yaml_fault(name: str, expected: str, mutator: Callable[[Path], None]) -> None:
        def run(repo: Path) -> str:
            mutator(repo)
            return _expect_failure(repo)
        cases.append((name, expected, run))

    def edit_modules(repo: Path, fn: Callable[[dict[str, Any]], None]) -> None:
        path = repo / "docs/handoff_shadow/stage4/minimal/MODULES.yaml"
        payload = load_yaml(path)
        fn(payload)
        write_yaml(path, payload)

    def edit_deps(repo: Path, fn: Callable[[dict[str, Any]], None]) -> None:
        path = repo / "docs/handoff_shadow/stage4/minimal/DEPENDENCIES.yaml"
        payload = load_yaml(path)
        fn(payload)
        write_yaml(path, payload)

    yaml_fault("missing_terminal_topic", "missing topic", lambda repo: edit_modules(repo, lambda p: _module(p, "terminal_audit")["content_contract"]["required_topics"].pop()))
    yaml_fault("terminal_evidence_wrong_source", "source-scoped evidence failure", lambda repo: edit_modules(repo, lambda p: _module(p, "terminal_audit")["content_contract"]["required_topics"][-1].update({"source_label_any": ["impossible-source-label"]})))
    yaml_fault("authority_promoted", "shadow authority boundary failure", lambda repo: edit_modules(repo, lambda p: p.update({"authority": "authoritative"})))
    yaml_fault("automatic_structure_policy", "automatic structure mutation failure", lambda repo: edit_modules(repo, lambda p: p.update({"structure_change_policy": "automatic"})))
    yaml_fault("unknown_dependency", "unknown dependency failure", lambda repo: edit_deps(repo, lambda p: p["depends_on"]["hopper_e7"].append("unknown_module")))
    yaml_fault("self_dependency", "self dependency failure", lambda repo: edit_deps(repo, lambda p: p["depends_on"]["hopper_e7"].append("hopper_e7")))
    yaml_fault("dependency_cycle", "cycle failure", lambda repo: edit_deps(repo, lambda p: p["depends_on"]["global_core_governance"].append("hopper_e7")))
    yaml_fault("hopper_leaks_countdown", "closure leakage failure", lambda repo: edit_deps(repo, lambda p: p["depends_on"]["hopper_e7"].append("countdown_e8")))
    yaml_fault("countdown_leaks_hopper", "closure leakage failure", lambda repo: edit_deps(repo, lambda p: p["depends_on"]["countdown_e8"].append("hopper_e7")))

    def add_registry(repo: Path, collection: str, item: dict[str, Any]) -> None:
        path = repo / "experiments/registry.yaml"
        payload = load_yaml(path)
        payload[collection].append(item)
        write_yaml(path, payload)

    yaml_fault("unmapped_canonical_experiment", "unmapped canonical failure", lambda repo: add_registry(repo, "experiments", {"id": "C-U1-ACCEPTANCE-UNMAPPED", "execution_class": "formal"}))
    yaml_fault("unmapped_formal_development", "unmapped formal development failure", lambda repo: add_registry(repo, "development_experiment_registrations", {"id": "DEV-ACCEPTANCE-FORMAL-UNMAPPED", "execution_class": "formal"}))
    yaml_fault("unmapped_pilot_development", "unmapped pilot development failure", lambda repo: add_registry(repo, "development_experiment_registrations", {"id": "DEV-ACCEPTANCE-PILOT-UNMAPPED", "execution_class": "pilot"}))

    def stale_registry(repo: Path) -> None:
        modules_path = repo / "docs/handoff_shadow/stage4/minimal/MODULES.yaml"
        modules = load_yaml(modules_path)
        target_id = None
        for module in modules["modules"]:
            for source in module.get("sources", []):
                ids = source.get("experiment_ids", [])
                if source.get("kind") == "registry_entries" and ids:
                    target_id = ids[0]
                    break
            if target_id:
                break
        registry_path = repo / "experiments/registry.yaml"
        registry = load_yaml(registry_path)
        registry["experiments"] = [item for item in registry["experiments"] if item.get("id") != target_id]
        registry["development_experiment_registrations"] = [item for item in registry["development_experiment_registrations"] if item.get("id") != target_id]
        write_yaml(registry_path, registry)
    yaml_fault("stale_registry_mapping", "stale mapping failure", stale_registry)

    def partial_overlap(repo: Path) -> None:
        def mutate(payload: dict[str, Any]) -> None:
            _module(payload, "global_core_governance")["sources"].append({
                "kind": "markdown_range", "path": "docs/handoff.md",
                "start": '## 0.1 当前执行门禁', "end": '# 2. 环境登记表（锁定)'
            })
        edit_modules(repo, mutate)
    # Use a guaranteed real end marker and overlap a subset plus following section.
    def partial_overlap_real(repo: Path) -> None:
        def mutate(payload: dict[str, Any]) -> None:
            _module(payload, "global_core_governance")["sources"].append({
                "kind": "markdown_range", "path": "docs/handoff.md",
                "start": '## 0.1 当前执行门禁', "end": '# 2. 环境登记表（锁定）'
            })
        edit_modules(repo, mutate)
    yaml_fault("partial_source_overlap", "partial overlap failure", partial_overlap_real)

    def duplicate_marker(repo: Path) -> str:
        handoff = repo / "docs/handoff.md"
        text = handoff.read_text(encoding="utf-8")
        start = text.find("<!-- HANDOFF-DELTA-BLOCK:")
        if start < 0:
            raise AcceptanceError("test fixture has no handoff delta marker")
        start_line_end = text.find("-->", start) + 3
        start_marker = text[start:start_line_end]
        if not start_marker.endswith(":START -->"):
            raise AcceptanceError("test fixture first handoff marker is not a START marker")
        end_marker = start_marker.replace(":START -->", ":END -->")
        end_start = text.find(end_marker, start_line_end)
        if end_start < 0:
            raise AcceptanceError("test fixture has no matching handoff END marker")
        end = end_start + len(end_marker)
        handoff.write_text(text + "\n\n" + text[start:end] + "\n", encoding="utf-8")
        return _expect_failure(repo)
    cases.append(("duplicate_marker_block", "duplicate marker failure", duplicate_marker))

    def stale_generated(repo: Path) -> str:
        path = repo / "docs/handoff_shadow/stage4/minimal/generated/MODULE_INDEX.json"
        path.write_bytes(path.read_bytes() + b"\n")
        proc = subprocess.run(
            [sys.executable, "scripts/validate_stage4_context.py", "--repo-root", str(repo), "--json"],
            cwd=repo, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False,
        )
        if proc.returncode == 0:
            raise AcceptanceError("tampered generated output was not rejected")
        return (proc.stdout + proc.stderr)[-1200:]
    cases.append(("tampered_generated_output", "stale output failure", stale_generated))

    def missing_generated(repo: Path) -> str:
        (repo / "docs/handoff_shadow/stage4/minimal/generated/modules/terminal_audit.md").unlink()
        proc = subprocess.run(
            [sys.executable, "scripts/validate_stage4_context.py", "--repo-root", str(repo), "--json"],
            cwd=repo, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False,
        )
        if proc.returncode == 0:
            raise AcceptanceError("missing generated output was not rejected")
        return (proc.stdout + proc.stderr)[-1200:]
    cases.append(("missing_generated_output", "missing output failure", missing_generated))

    def full_coverage_dedup(repo: Path) -> str:
        def mutate(payload: dict[str, Any]) -> None:
            source = dict(_module(payload, "global_core_governance")["sources"][0])
            _module(payload, "global_core_governance")["sources"].append(source)
        edit_modules(repo, mutate)
        plan = _run_build(repo)
        dropped = plan.snapshots["global_core_governance"].deduplicated_source_labels
        if not dropped:
            raise AcceptanceError("fully covered duplicate was not provenance-deduplicated")
        return f"deduplicated={len(dropped)}"
    cases.append(("fully_covered_source_dedup", "dedup with provenance", full_coverage_dedup))

    def builder_side_effect(repo: Path) -> str:
        before = {value: sha256_path(repo / value) for value in AUTHORITATIVE_INPUTS}
        subprocess_json([sys.executable, "scripts/build_stage4_context.py", "--repo-root", str(repo), "--json", "build"], repo)
        after = {value: sha256_path(repo / value) for value in AUTHORITATIVE_INPUTS}
        if before != after:
            raise AcceptanceError("builder modified authoritative input")
        return "authoritative hashes unchanged"
    cases.append(("builder_authority_isolation", "no authoritative side effect", builder_side_effect))

    results: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="drpo-stage4a-faults-") as temp:
        temp_root = Path(temp)
        for index, (case_id, expected, runner) in enumerate(cases):
            print(f"[fault {index + 1}/{len(cases)}] {case_id}", file=sys.stderr, flush=True)
            candidate = temp_root / f"case-{index:02d}"
            copy_repository(repo_root, candidate)
            try:
                diagnostic = runner(candidate)
            except Exception as exc:
                raise AcceptanceError(f"fault-injection case {case_id} failed its harness: {exc}") from exc
            results.append({"case_id": case_id, "expected": expected, "status": "PASS", "diagnostic": diagnostic[:1200]})
    return {"status": "PASS", "total": len(results), "passed": len(results), "cases": results}


def collect_after_image(repo_root: Path) -> dict[str, Any]:
    files: dict[str, str] = {}
    for value in AFTER_IMAGE_PATHS:
        path = repo_root / value
        if not path.exists():
            raise AcceptanceError(f"after-image path is missing: {value}")
        candidates = [path] if path.is_file() else sorted(path.rglob("*"))
        for candidate in candidates:
            if candidate.is_symlink():
                raise AcceptanceError(f"after-image path may not be a symlink: {candidate}")
            if candidate.is_file() and "docs/governance_stage4a_acceptance/" not in candidate.relative_to(repo_root).as_posix():
                files[candidate.relative_to(repo_root).as_posix()] = sha256_path(candidate)
    entries = [{"path": path, "sha256": digest} for path, digest in sorted(files.items())]
    return {
        "schema_version": 1,
        "policy_id": POLICY_ID,
        "authority": AUTHORITY,
        "base_commit": BASE_COMMIT,
        "files": entries,
        "file_count": len(entries),
        "tree_hash": sha256_bytes(canonical_json(entries)),
    }


def load_gate_evidence(path: Path | None) -> list[dict[str, Any]]:
    if path is None:
        raise AcceptanceError("--gate-evidence is required when writing final acceptance evidence")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise AcceptanceError(f"invalid gate evidence: {exc}") from exc
    if not isinstance(payload, list) or not payload:
        raise AcceptanceError("gate evidence must be a non-empty JSON list")
    for item in payload:
        if not isinstance(item, dict) or item.get("status") != "PASS" or not item.get("command"):
            raise AcceptanceError("every repository gate evidence entry must be PASS with a command")
    required_tokens = ("pytest -q", "ruff check .", "compileall")
    joined = "\n".join(str(item.get("command")) for item in payload)
    missing = [token for token in required_tokens if token not in joined]
    if missing:
        raise AcceptanceError(f"gate evidence is missing required commands: {missing}")
    return payload


def write_outputs(repo_root: Path, core: dict[str, Any], faults: dict[str, Any], gates: list[dict[str, Any]]) -> dict[str, Any]:
    output = repo_root / OUTPUT_REL
    output.mkdir(parents=True, exist_ok=True)
    after_image = collect_after_image(repo_root)
    report = {
        "schema_version": 1,
        "policy_id": POLICY_ID,
        "claim_id": POLICY_ID,
        "evaluated_base_commit": BASE_COMMIT,
        "status": "PASS",
        "authority": AUTHORITY,
        "manual_handoff_remains_authoritative": True,
        "authority_cutover_allowed": False,
        "stage_4a_state": "accepted",
        "stage_4b_state": "ready_for_separate_authorization",
        "stage_4c_state": "blocked_by_stage_4b_acceptance",
        "hard_blockers": [],
        "module_count": core["minimal_context"]["module_count"],
        "dependency_edge_count": core["minimal_context"]["edge_count"],
        "minimal_graph_hash": core["minimal_context"]["graph_hash"],
        "semantic_graph_hash": core["semantic_graph"]["graph_hash"],
        "mapping": core["mapping"],
        "semantic_contracts": core["semantic_contracts"],
        "acceptance_targets": core["acceptance_targets"],
        "determinism": core["determinism"],
        "fault_injection": {"status": faults["status"], "total": faults["total"], "passed": faults["passed"]},
        "repository_gates": gates,
        "after_image_tree_hash": after_image["tree_hash"],
        "remaining_advisories": core["suggestions"],
    }
    fault_path = output / "FAULT_INJECTION_REPORT.json"
    after_path = output / "AFTER_IMAGE.json"
    report_path = output / "ACCEPTANCE_REPORT.json"
    fault_path.write_bytes(json.dumps(faults, ensure_ascii=False, indent=2, sort_keys=True).encode("utf-8") + b"\n")
    after_path.write_bytes(json.dumps(after_image, ensure_ascii=False, indent=2, sort_keys=True).encode("utf-8") + b"\n")
    report_path.write_bytes(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True).encode("utf-8") + b"\n")
    summary = f"""# Stage 4A Final Acceptance Summary

- Policy / claim: `{POLICY_ID}`
- Evaluated base commit: `{BASE_COMMIT}`
- Result: **PASS**
- Authority: **shadow only**; `docs/handoff.md` remains authoritative.
- Modules / dependency edges: `{report['module_count']}` / `{report['dependency_edge_count']}`
- Semantic contracts: `{len(report['semantic_contracts'])}` modules, `{sum(v['required_topics'] for v in report['semantic_contracts'].values())}` required topics, all source-scoped evidence satisfied.
- Fault injection: `{faults['passed']}/{faults['total']}` cases passed.
- Stage 4B: ready for a separate authorization, not active.
- Stage 4C, Stage 5, and authority cutover: blocked.

Length-only structure suggestions remain advisory and do not weaken semantic acceptance.
"""
    (output / "ACCEPTANCE_SUMMARY.md").write_text(summary, encoding="utf-8")
    checksum_targets = [report_path, fault_path, after_path, output / "ACCEPTANCE_SUMMARY.md", repo_root / "docs/governance_stage4a_acceptance_spec.md"]
    lines = [f"{sha256_path(path)}  {path.relative_to(repo_root).as_posix()}" for path in checksum_targets]
    (output / "CHECKSUMS.sha256").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--write", action="store_true")
    parser.add_argument("--skip-fault-injection", action="store_true")
    parser.add_argument("--gate-evidence", type=Path)
    parser.add_argument("--json", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    repo_root = args.repo_root.resolve()
    try:
        core = run_core_acceptance(repo_root, check_determinism=True)
        faults = ({"status": "SKIPPED", "total": 0, "passed": 0, "cases": []}
                  if args.skip_fault_injection else run_fault_injections(repo_root))
        if args.write and faults["status"] != "PASS":
            raise AcceptanceError("final evidence cannot be written with fault injection skipped")
        if args.write:
            gates = load_gate_evidence(args.gate_evidence)
            report = write_outputs(repo_root, core, faults, gates)
        else:
            report = {
                "status": "PASS", "policy_id": POLICY_ID, "authority": AUTHORITY,
                "module_count": core["minimal_context"]["module_count"],
                "dependency_edge_count": core["minimal_context"]["edge_count"],
                "fault_injection": {"status": faults["status"], "total": faults["total"], "passed": faults["passed"]},
                "hard_blockers": [],
            }
    except (AcceptanceError, OSError, RuntimeError) as exc:
        if args.json:
            print(json.dumps({"status": "FAIL", "error": str(exc)}, ensure_ascii=False, indent=2, sort_keys=True))
        else:
            print(f"Stage 4A final acceptance: FAIL: {exc}", file=sys.stderr)
        return 2
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    else:
        print(
            "Stage 4A final acceptance: PASS "
            f"(modules={report['module_count']}, edges={report['dependency_edge_count']}, "
            f"faults={report['fault_injection']['passed']}/{report['fault_injection']['total']})"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
