from __future__ import annotations

import json
import os
import sys
from copy import deepcopy
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from prepare_dev_pilot_registration import (  # noqa: E402
    PreparationError,
    prepare,
    sha256_file,
)
from validate_dev_integration import json_hash  # noqa: E402

MAIN_SHA = "1" * 40
DEV_SHA = "2" * 40
BLOB_A = "3" * 40
BLOB_B = "4" * 40


def write_yaml(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(payload, sort_keys=False, allow_unicode=True, width=1000),
        encoding="utf-8",
    )


def registry(experiments: list[dict] | None = None) -> dict:
    return {
        "schema_version": 2,
        "project": {
            "name": "drpo",
            "repository": "easonhuo/drpo",
            "default_branch": "main",
            "research_source_of_truth": "docs/handoff.md",
        },
        "allowed_statuses": ["pilot", "not_run"],
        "rules": {"document_before_experiment": True},
        "experiments": experiments
        or [
            {
                "id": "EXISTING-EXPERIMENT-01",
                "environment": "test",
                "name": "existing",
                "status": "pilot",
                "claim": "fixture",
                "role": "diagnostic",
            }
        ],
    }


def approved_decision() -> dict:
    return {
        "approved": True,
        "code_integration_eligible": True,
        "evidence_level": "pilot",
        "result_status": "pilot",
        "claim_support_level": "diagnostic",
        "terminal_audit": "partial",
        "task_performance_collapse": "inconclusive",
        "support_boundary": "inconclusive",
        "numerical_failure": "none",
    }


def base_spec(experiment_id: str = "EXAMPLE-EXPERIMENT-01") -> dict:
    return {
        "schema_version": 1,
        "preparation_id": "GOV-EXAMPLE-PILOT-PREP-01",
        "source": {
            "repository": "easonhuo/drpo",
            "remote": "origin",
            "main_ref": "refs/heads/main",
            "expected_main_sha": MAIN_SHA,
            "dev_branch": "dev/example-pilot",
            "expected_dev_sha": DEV_SHA,
            "result_commit_sha": DEV_SHA,
            "result_git_dirty": False,
        },
        "subject": {
            "experiment_id": experiment_id,
            "governance_claims": [],
        },
        "implementation": {
            "operations": [
                {
                    "op": "add",
                    "source_path": "src/drpo/example_pilot.py",
                    "destination_path": "src/drpo/example_pilot.py",
                    "expected_blob_sha": BLOB_A,
                    "expected_old_blob_sha": None,
                    "expected_mode": "100644",
                }
            ]
        },
        "review": {
            "reviewer_id": "chatgpt-reviewer",
            "decision_token": "review-token-001",
            "decision": approved_decision(),
            "limitations": ["development pilot only"],
            "unresolved": ["terminal result unavailable"],
        },
        "registration": {
            "mode": "none",
            "update_id": None,
            "expected_before_semantic_sha256": None,
            "experiment": None,
            "handoff_operations": [],
            "registry_changes": [],
        },
    }


def registration_spec(
    *,
    experiment_id: str = "EXAMPLE-EXPERIMENT-01",
    mode: str = "add_experiment",
    existing: dict | None = None,
) -> dict:
    spec = base_spec(experiment_id)
    experiment = existing or {
        "id": experiment_id,
        "environment": "test",
        "name": "prepared_pilot",
        "status": "pilot",
        "claim": "Fixture code-first pilot registration.",
        "role": "external_validity_tuning",
        "result_status": "not_run",
    }
    spec["registration"] = {
        "mode": mode,
        "update_id": f"{experiment_id}-REGISTRATION-2026-07-14",
        "expected_before_semantic_sha256": (
            json_hash(existing) if mode == "replace_experiment" and existing else None
        ),
        "experiment": experiment,
        "handoff_operations": [
            {
                "operation_id": "append-pilot-fixture",
                "op": "append_to_section",
                "heading_path": [
                    "0. 研究与执行原则（每次新会话首先阅读）",
                    "0.1 当前执行门禁",
                ],
                "block_id": "pilot-fixture",
                "content": f"- Register `{experiment_id}` as a development pilot.",
            }
        ],
        "registry_changes": [
            {
                "change_id": "register-pilot-fixture",
                "kind": "add_entity" if mode == "add_experiment" else "replace_entity",
                "entity_id": experiment_id,
                "evidence": ["experiments/registry.yaml"],
            }
        ],
    }
    return spec


def make_repo(tmp_path: Path, experiments: list[dict] | None = None) -> Path:
    repo = tmp_path / "repo"
    write_yaml(repo / "experiments" / "registry.yaml", registry(experiments))
    return repo


def run_prepare(tmp_path: Path, spec: dict, repo: Path | None = None) -> tuple[dict, Path]:
    repo = repo or make_repo(tmp_path)
    spec_path = tmp_path / "spec.yaml"
    write_yaml(spec_path, spec)
    output_root = tmp_path / "prepared"
    result = prepare(repo, spec_path, output_root)
    return result, output_root / spec["preparation_id"]


def test_code_only_preparation_is_v1_compatible(tmp_path: Path) -> None:
    result, prepared = run_prepare(tmp_path, base_spec())
    assert result["state"] == "PREPARED_INPUTS"
    integration = prepared / "repository_overlay" / "docs" / "integrations" / result["preparation_id"]
    assert (integration / "INTEGRATION_REQUEST.yaml").is_file()
    assert (integration / "REVIEW_DECISION.yaml").is_file()
    assert not (prepared / "transaction_inputs").exists()
    report = json.loads((prepared / "PREPARATION_REPORT.json").read_text(encoding="utf-8"))
    assert report["mode"] == "none"
    assert report["network_used"] is False
    assert report["repository_modified"] is False


def test_add_registration_compiles_hash_bound_transaction_inputs(tmp_path: Path) -> None:
    spec = registration_spec()
    result, prepared = run_prepare(tmp_path, spec)
    tx = prepared / "transaction_inputs"
    intent = tx / "REGISTRATION_INTENT.yaml"
    approval = yaml.safe_load((tx / "REGISTRATION_APPROVAL.yaml").read_text(encoding="utf-8"))
    overlay = prepared / "repository_overlay" / "docs" / "integrations" / result["preparation_id"]
    assert approval["intent_sha256"] == sha256_file(intent)
    assert approval["request_sha256"] == sha256_file(overlay / "INTEGRATION_REQUEST.yaml")
    assert approval["review_decision_sha256"] == sha256_file(
        overlay / "REVIEW_DECISION.yaml"
    )
    assert approval["reviewer"] == {
        "id": "chatgpt-reviewer",
        "decision_token": "review-token-001",
    }


def test_replace_registration_checks_exact_semantic_before_image(tmp_path: Path) -> None:
    existing = {
        "id": "EXISTING-EXPERIMENT-01",
        "environment": "test",
        "name": "existing",
        "status": "pilot",
        "claim": "fixture",
        "role": "diagnostic",
    }
    repo = make_repo(tmp_path, [existing])
    updated = {**existing, "claim": "reviewed replacement"}
    spec = registration_spec(
        experiment_id=existing["id"],
        mode="replace_experiment",
        existing=updated,
    )
    spec["registration"]["expected_before_semantic_sha256"] = json_hash(existing)
    result, prepared = run_prepare(tmp_path, spec, repo)
    assert result["status"] == "PASS"
    intent = yaml.safe_load(
        (prepared / "transaction_inputs" / "REGISTRATION_INTENT.yaml").read_text(
            encoding="utf-8"
        )
    )
    assert intent["registry_mutation"]["kind"] == "replace_experiment"
    assert intent["registry_mutation"]["expected_before_semantic_sha256"] == json_hash(
        existing
    )


def test_identical_rerun_is_idempotent(tmp_path: Path) -> None:
    repo = make_repo(tmp_path)
    spec = base_spec()
    spec_path = tmp_path / "spec.yaml"
    write_yaml(spec_path, spec)
    output_root = tmp_path / "prepared"
    first = prepare(repo, spec_path, output_root)
    second = prepare(repo, spec_path, output_root)
    assert first["idempotent_reuse"] is False
    assert second["idempotent_reuse"] is True
    assert first["manifest_sha256"] == second["manifest_sha256"]


def test_conflicting_existing_output_is_rejected(tmp_path: Path) -> None:
    repo = make_repo(tmp_path)
    spec = base_spec()
    spec_path = tmp_path / "spec.yaml"
    write_yaml(spec_path, spec)
    output_root = tmp_path / "prepared"
    prepare(repo, spec_path, output_root)
    decision = (
        output_root
        / spec["preparation_id"]
        / "repository_overlay"
        / "docs"
        / "integrations"
        / spec["preparation_id"]
        / "REVIEW_DECISION.yaml"
    )
    decision.write_text("tampered: true\n", encoding="utf-8")
    with pytest.raises(PreparationError, match="OUTPUT_CONFLICT"):
        prepare(repo, spec_path, output_root)


def test_unknown_spec_key_fails_closed(tmp_path: Path) -> None:
    spec = base_spec()
    spec["unexpected"] = True
    with pytest.raises(PreparationError, match="unknown keys"):
        run_prepare(tmp_path, spec)


def test_forbidden_import_path_fails_before_publish(tmp_path: Path) -> None:
    spec = base_spec()
    operation = spec["implementation"]["operations"][0]
    operation["source_path"] = "docs/handoff.md"
    operation["destination_path"] = "docs/handoff.md"
    repo = make_repo(tmp_path)
    spec_path = tmp_path / "spec.yaml"
    write_yaml(spec_path, spec)
    output_root = tmp_path / "prepared"
    with pytest.raises(PreparationError, match="system-forbidden"):
        prepare(repo, spec_path, output_root)
    assert not (output_root / spec["preparation_id"]).exists()


def test_casefold_target_collision_fails_closed(tmp_path: Path) -> None:
    spec = base_spec()
    spec["implementation"]["operations"] = [
        {
            "op": "add",
            "source_path": "src/drpo/Foo.py",
            "destination_path": "src/drpo/Foo.py",
            "expected_blob_sha": BLOB_A,
            "expected_old_blob_sha": None,
            "expected_mode": "100644",
        },
        {
            "op": "add",
            "source_path": "src/drpo/foo.py",
            "destination_path": "src/drpo/foo.py",
            "expected_blob_sha": BLOB_B,
            "expected_old_blob_sha": None,
            "expected_mode": "100644",
        },
    ]
    with pytest.raises(PreparationError, match="target collision"):
        run_prepare(tmp_path, spec)


def test_registration_target_mismatch_fails_closed(tmp_path: Path) -> None:
    spec = registration_spec()
    spec["registration"]["experiment"]["id"] = "OTHER-EXPERIMENT-01"
    with pytest.raises(PreparationError, match="experiment ID mismatch"):
        run_prepare(tmp_path, spec)


def test_stale_replace_before_image_fails_closed(tmp_path: Path) -> None:
    existing = registry()["experiments"][0]
    repo = make_repo(tmp_path, [existing])
    spec = registration_spec(
        experiment_id=existing["id"],
        mode="replace_experiment",
        existing={**existing, "claim": "new"},
    )
    spec["registration"]["expected_before_semantic_sha256"] = "0" * 64
    with pytest.raises(PreparationError, match="before-image"):
        run_prepare(tmp_path, spec, repo)


def test_unapproved_registration_is_not_promoted(tmp_path: Path) -> None:
    spec = registration_spec()
    spec["review"]["decision"]["approved"] = False
    with pytest.raises(PreparationError, match="SCIENTIFIC_REVIEW_MISSING"):
        run_prepare(tmp_path, spec)


def test_dirty_formal_evidence_is_rejected_by_existing_validator(tmp_path: Path) -> None:
    spec = base_spec()
    spec["source"]["result_git_dirty"] = True
    spec["review"]["decision"]["evidence_level"] = "formal"
    with pytest.raises(PreparationError, match="dirty-worktree"):
        run_prepare(tmp_path, spec)


def test_injected_write_failure_publishes_no_partial_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = make_repo(tmp_path)
    spec = registration_spec()
    spec_path = tmp_path / "spec.yaml"
    write_yaml(spec_path, spec)
    output_root = tmp_path / "prepared"
    monkeypatch.setenv("DRPO_PREPARATION_INJECT_FAILURE_AFTER_FILES", "1")
    with pytest.raises(PreparationError, match="INJECTED_FAILURE"):
        prepare(repo, spec_path, output_root)
    assert not (output_root / spec["preparation_id"]).exists()
    assert not list(output_root.glob(f".{spec['preparation_id']}.*"))


def test_historical_e8_style_registration_replay_fixture(tmp_path: Path) -> None:
    experiment_id = "EXT-C-E8-ORACLE-OFFLINE-V2-CONTINUOUS-EXP-GRID-0.5B-01"
    spec = registration_spec(experiment_id=experiment_id)
    spec["preparation_id"] = "EXT-C-E8-CONTINUOUS-EXP-REPLAY-2026-07-14"
    spec["source"]["dev_branch"] = "dev/e8-continuous-exp-grid-pilot"
    spec["registration"]["update_id"] = (
        "EXT-C-E8-V2-CONTINUOUS-EXP-GRID-REGISTRATION-2026-07-13"
    )
    spec["registration"]["experiment"].update(
        {
            "environment": "Countdown",
            "name": "continuous_exp_grid_0p5b",
            "status": "pilot",
            "claim": "Development tuning only; no formal ranking or convergence claim.",
            "role": "external_validity_tuning",
            "result_status": "not_run",
        }
    )
    spec["registration"]["registry_changes"][0]["entity_id"] = experiment_id
    result, prepared = run_prepare(tmp_path, spec)
    assert result["status"] == "PASS"
    intent = yaml.safe_load(
        (prepared / "transaction_inputs" / "REGISTRATION_INTENT.yaml").read_text(
            encoding="utf-8"
        )
    )
    assert intent["registry_mutation"]["experiment_id"] == experiment_id
    assert intent["registry_mutation"]["experiment"]["result_status"] == "not_run"
    assert "ranking" in intent["registry_mutation"]["experiment"]["claim"]


def test_input_spec_is_not_mutated(tmp_path: Path) -> None:
    spec = registration_spec()
    before = deepcopy(spec)
    run_prepare(tmp_path, spec)
    assert spec == before


def test_failure_injection_environment_is_optional(tmp_path: Path) -> None:
    assert "DRPO_PREPARATION_INJECT_FAILURE_AFTER_FILES" not in os.environ
    result, _ = run_prepare(tmp_path, base_spec())
    assert result["status"] == "PASS"
