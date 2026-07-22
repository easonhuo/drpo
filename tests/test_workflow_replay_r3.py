from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import pytest
import yaml

from drpo.workflow_replay.evidence import RunIdentity, canonical_sha256
from drpo.workflow_replay.trajectory import (
    RESOURCE_DIMENSIONS,
    TrajectoryError,
    load_r3_run_artifact,
    summarize_trajectory,
    validate_attempt_record,
    validate_r3_run_artifact,
)

ROOT = Path(__file__).resolve().parents[1]
INVENTORY = ROOT / "docs/development_workflow_optimization/REPLAYAB_R3_CALIBRATION_INVENTORY.yaml"
INPUTS = ROOT / "tests/fixtures/workflow_replay/r3/calibration_cases.yaml"
OBSERVED = ("command_count", "active_ns", "retained_bytes")


def _identity() -> RunIdentity:
    return RunIdentity.build("REPLAYAB-R3-CAL-01", "A", "pair-0", 0, 0, "fixture-r3-v1")


def _identity_payload(identity: RunIdentity) -> dict:
    return asdict(identity)


def _write(root: Path, relative_path: str, raw: bytes, kind: str) -> dict:
    path = root / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)
    return {
        "kind": kind,
        "relative_path": relative_path,
        "sha256": hashlib.sha256(raw).hexdigest(),
        "byte_size": len(raw),
    }


def _json_locator(root: Path, relative_path: str, value: dict, kind: str) -> dict:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return _write(root, relative_path, raw, kind)


def _capabilities() -> dict[str, str]:
    return {
        name: "OBSERVED" if name in OBSERVED else "UNAVAILABLE"
        for name in RESOURCE_DIMENSIONS
    }


def _attempt(
    root: Path,
    identity: RunIdentity,
    ordinal: int,
    terminal: str,
    disposition: str,
    parent_attempt_id: str | None,
    feedback_class: str = "NONE",
) -> dict:
    attempt_id = canonical_sha256({"run_id": identity.run_id, "ordinal": ordinal})
    candidate_produced = terminal in {"SUCCEEDED", "FAILED", "TIMED_OUT"}
    event = {
        "schema_version": 1,
        "run_id": identity.run_id,
        "attempt_id": attempt_id,
        "ordinal": ordinal,
        "terminal": terminal,
        "candidate_artifact_produced": candidate_produced,
    }
    event_locator = _json_locator(
        root,
        f"attempts/{ordinal}/events.json",
        event,
        f"journal-{attempt_id}",
    )
    output_locator = None
    if candidate_produced:
        output_locator = _write(
            root,
            f"attempts/{ordinal}/candidate.bin",
            f"candidate-{ordinal}-{terminal}".encode(),
            f"candidate-{attempt_id}",
        )
    feedback_locator = None
    if feedback_class != "NONE":
        feedback_locator = _write(
            root,
            f"attempts/{ordinal}/feedback.txt",
            f"feedback-for-{ordinal}".encode(),
            "feedback-"
            + canonical_sha256(
                {"parent_attempt_id": parent_attempt_id, "repair_attempt_id": attempt_id}
            ),
        )
    resources = {
        "command_count": ordinal + 1,
        "active_ns": (ordinal + 1) * 100,
        "retained_bytes": (ordinal + 1) * 10,
    }
    payload = {
        "attempt_id": attempt_id,
        "ordinal": ordinal,
        "kind": "INITIAL" if ordinal == 0 else "REPAIR",
        "parent_attempt_id": parent_attempt_id,
        "terminal": terminal,
        "disposition": disposition,
        "input_artifact_locator": None,
        "output_artifact_locator": output_locator,
        "event_journal_locator": event_locator,
        "feedback_class": feedback_class,
        "feedback_locator": feedback_locator,
        "diagnostic_codes": [] if terminal == "SUCCEEDED" else [f"R3_{terminal}"],
        "observed_resources": resources,
        "attempt_sha256": "",
    }
    payload["attempt_sha256"] = canonical_sha256(
        {key: value for key, value in payload.items() if key != "attempt_sha256"}
    )
    return payload


def _binding(identity: RunIdentity, final_attempt_id: str) -> dict:
    return {
        "schema_version": 1,
        "case_id": identity.case_id,
        "arm": identity.arm,
        "run_id": identity.run_id,
        "final_attempt_id": final_attempt_id,
    }


def _artifact(
    root: Path,
    attempts_spec: list[tuple[str, str, str]],
    final_acceptance: str,
) -> dict:
    identity = _identity()
    attempts: list[dict] = []
    parent = None
    for ordinal, (terminal, disposition, feedback_class) in enumerate(attempts_spec):
        current = _attempt(
            root,
            identity,
            ordinal,
            terminal,
            disposition,
            parent,
            feedback_class,
        )
        attempts.append(current)
        parent = current["attempt_id"]
    final_attempt_id = attempts[-1]["attempt_id"]
    final_outcome = None
    acceptance = None
    if final_acceptance in {"PASS", "REJECTED"}:
        final_outcome = _json_locator(
            root,
            "final/outcome.json",
            _binding(identity, final_attempt_id),
            "r3_final_outcome_binding",
        )
        acceptance_payload = _binding(identity, final_attempt_id)
        acceptance_payload["final_acceptance"] = final_acceptance
        acceptance = _json_locator(
            root,
            "final/acceptance.json",
            acceptance_payload,
            "r2_acceptance_binding",
        )
    aggregate = {
        name: sum(attempt["observed_resources"][name] for attempt in attempts)
        for name in OBSERVED
    }
    payload = {
        "schema_version": 1,
        "run_identity": _identity_payload(identity),
        "base_sha": "1" * 40,
        "toolchain_sha": "2" * 40,
        "environment_id": "linux-py311-fixture-v1",
        "cache_policy": "cold",
        "backend_id": identity.backend_id,
        "resource_capabilities": _capabilities(),
        "attempts": attempts,
        "first_attempt_id": attempts[0]["attempt_id"],
        "final_attempt_id": final_attempt_id,
        "final_outcome_locator": final_outcome,
        "final_acceptance": final_acceptance,
        "acceptance_evidence_locator": acceptance,
        "aggregate_observed_resources": aggregate,
        "run_artifact_sha256": "",
    }
    _resign_run(payload)
    return payload


def _resign_attempt(attempt: dict) -> None:
    attempt["attempt_sha256"] = canonical_sha256(
        {key: value for key, value in attempt.items() if key != "attempt_sha256"}
    )


def _resign_run(payload: dict) -> None:
    payload["run_artifact_sha256"] = canonical_sha256(
        {key: value for key, value in payload.items() if key != "run_artifact_sha256"}
    )


def _rewrite_json_locator(root: Path, locator: dict, value: dict) -> None:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    (root / locator["relative_path"]).write_bytes(raw)
    locator["sha256"] = hashlib.sha256(raw).hexdigest()
    locator["byte_size"] = len(raw)


def _summary_dict(summary) -> dict:
    return asdict(summary)


def _inventory() -> dict:
    return yaml.safe_load(INVENTORY.read_text(encoding="utf-8"))


def _calibration_inputs() -> dict[str, dict]:
    payload = yaml.safe_load(INPUTS.read_text(encoding="utf-8"))
    assert payload["schema_version"] == 1
    return {item["case_id"]: item for item in payload["cases"]}


def _valid_cases() -> dict[str, dict]:
    return {
        case_id: item
        for case_id, item in _calibration_inputs().items()
        if item["mutation"] is None
    }


@pytest.mark.parametrize("case_id", sorted(_valid_cases()))
def test_frozen_valid_calibration_cases(case_id: str, tmp_path: Path) -> None:
    inventory = {item["case_id"]: item for item in _inventory()["cases"]}
    case = _valid_cases()[case_id]
    specs = [tuple(item) for item in case["attempts"]]
    payload = _artifact(tmp_path, specs, case["final_acceptance"])
    artifact = validate_r3_run_artifact(payload, tmp_path)
    assert _summary_dict(summarize_trajectory(artifact)) == inventory[case_id]["expected_summary"]


def _invalid_case(case_id: str, root: Path) -> dict:
    case = _calibration_inputs()[case_id]
    payload = _artifact(
        root,
        [tuple(item) for item in case["attempts"]],
        case["final_acceptance"],
    )
    mutation = case["mutation"]
    if mutation == "remove_initial":
        payload["attempts"] = payload["attempts"][1:]
        payload["first_attempt_id"] = payload["attempts"][0]["attempt_id"]
        _resign_run(payload)
    elif mutation == "ordinal_gap":
        attempt = payload["attempts"][1]
        attempt["ordinal"] = 2
        attempt["attempt_id"] = canonical_sha256(
            {"run_id": payload["run_identity"]["run_id"], "ordinal": 2}
        )
        _resign_attempt(attempt)
        payload["final_attempt_id"] = attempt["attempt_id"]
        _resign_run(payload)
    elif mutation == "wrong_parent":
        attempt = payload["attempts"][1]
        attempt["parent_attempt_id"] = "f" * 64
        attempt["feedback_locator"]["kind"] = "feedback-" + canonical_sha256(
            {
                "parent_attempt_id": attempt["parent_attempt_id"],
                "repair_attempt_id": attempt["attempt_id"],
            }
        )
        _resign_attempt(attempt)
        _resign_run(payload)
    elif mutation == "wrong_feedback_binding":
        attempt = payload["attempts"][1]
        attempt["feedback_locator"]["kind"] = "feedback-" + "0" * 64
        _resign_attempt(attempt)
        _resign_run(payload)
    elif mutation == "tamper_output":
        locator = payload["attempts"][0]["output_artifact_locator"]
        (root / locator["relative_path"]).write_bytes(b"tampered")
    elif mutation == "wrong_final_pointer":
        payload["final_attempt_id"] = payload["attempts"][0]["attempt_id"]
        _resign_run(payload)
    elif mutation == "wrong_resource_aggregate":
        payload["aggregate_observed_resources"]["command_count"] += 1
        _resign_run(payload)
    elif mutation == "missing_resource_capability":
        payload["resource_capabilities"].pop("token_count")
        _resign_run(payload)
    else:
        raise AssertionError(f"unknown calibration mutation {mutation}")
    return payload


INVALID_CASES = tuple(
    case_id
    for case_id, item in _calibration_inputs().items()
    if item["mutation"] is not None
)


@pytest.mark.parametrize("case_id", INVALID_CASES)
def test_frozen_invalid_calibration_cases_fail_closed(case_id: str, tmp_path: Path) -> None:
    inventory = {item["case_id"]: item for item in _inventory()["cases"]}
    payload = _invalid_case(case_id, tmp_path)
    with pytest.raises(TrajectoryError) as caught:
        validate_r3_run_artifact(payload, tmp_path)
    assert caught.value.code == inventory[case_id]["expected_error_code"]
    assert str(caught.value).startswith(f"{caught.value.code}:")


def test_calibration_inventory_is_complete_and_frozen() -> None:
    inventory = _inventory()
    cases = {item["case_id"]: item for item in inventory["cases"]}
    assert set(cases) == set(_valid_cases()) | set(INVALID_CASES)
    assert len(cases) == 16
    assert all(cases[case_id]["expected_ingestion"] == "ACCEPT" for case_id in _valid_cases())
    assert all(cases[case_id]["expected_ingestion"] == "REJECT" for case_id in INVALID_CASES)


def test_run_and_attempt_digests_are_deterministic(tmp_path: Path) -> None:
    first_root = tmp_path / "first"
    second_root = tmp_path / "second"
    first = _artifact(first_root, [("SUCCEEDED", "NONE", "NONE")], "PASS")
    second = _artifact(second_root, [("SUCCEEDED", "NONE", "NONE")], "PASS")
    assert first == second
    assert first["attempts"][0]["attempt_sha256"] == second["attempts"][0]["attempt_sha256"]
    assert first["run_artifact_sha256"] == second["run_artifact_sha256"]


def test_loader_rejects_unknown_fields_and_never_returns_partial_summary(tmp_path: Path) -> None:
    payload = _artifact(tmp_path, [("SUCCEEDED", "NONE", "NONE")], "PASS")
    payload["unexpected"] = True
    artifact_path = tmp_path / "run-artifact.json"
    artifact_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(TrajectoryError, match="SCHEMA_INVALID"):
        load_r3_run_artifact(artifact_path, tmp_path)


def test_direct_validation_rejects_oversized_artifact(tmp_path: Path) -> None:
    payload = _artifact(tmp_path, [("SUCCEEDED", "NONE", "NONE")], "PASS")
    payload["attempts"][0]["diagnostic_codes"] = ["X" * 262144]
    _resign_attempt(payload["attempts"][0])
    _resign_run(payload)
    with pytest.raises(TrajectoryError, match="LIMIT_EXCEEDED"):
        validate_r3_run_artifact(payload, tmp_path)


def test_loader_rejects_symlink_and_oversized_json(tmp_path: Path) -> None:
    target = tmp_path / "target.json"
    target.write_text("{}", encoding="utf-8")
    link = tmp_path / "link.json"
    try:
        link.symlink_to(target)
    except (OSError, NotImplementedError):
        pytest.skip("symlinks unavailable")
    with pytest.raises(TrajectoryError, match="SCHEMA_INVALID"):
        load_r3_run_artifact(link, tmp_path)
    oversized = tmp_path / "oversized.json"
    oversized.write_bytes(b"{" + b" " * (262144 + 1) + b"}")
    with pytest.raises(TrajectoryError, match="LIMIT_EXCEEDED"):
        load_r3_run_artifact(oversized, tmp_path)


def test_binding_evidence_cannot_point_to_a_different_attempt(tmp_path: Path) -> None:
    payload = _artifact(
        tmp_path,
        [
            ("FAILED", "CANDIDATE", "NONE"),
            ("SUCCEEDED", "NONE", "EVALUATOR"),
        ],
        "PASS",
    )
    locator = payload["final_outcome_locator"]
    wrong = _binding(_identity(), payload["attempts"][0]["attempt_id"])
    _rewrite_json_locator(tmp_path, locator, wrong)
    _resign_run(payload)
    with pytest.raises(TrajectoryError) as caught:
        validate_r3_run_artifact(payload, tmp_path)
    assert caught.value.code == "FINAL_POINTER_MISMATCH"


@pytest.mark.parametrize(
    ("field", "wrong_value"),
    [
        ("run_id", "0" * 64),
        ("attempt_id", "0" * 64),
        ("ordinal", 1),
        ("terminal", "FAILED"),
        ("candidate_artifact_produced", False),
    ],
)
def test_attempt_journal_semantic_binding_survives_outer_resigning(
    field: str,
    wrong_value: object,
    tmp_path: Path,
) -> None:
    payload = _artifact(tmp_path, [("SUCCEEDED", "NONE", "NONE")], "PASS")
    attempt = payload["attempts"][0]
    locator = attempt["event_journal_locator"]
    journal_path = tmp_path / locator["relative_path"]
    journal = json.loads(journal_path.read_text(encoding="utf-8"))
    journal[field] = wrong_value
    _rewrite_json_locator(tmp_path, locator, journal)
    _resign_attempt(attempt)
    _resign_run(payload)

    with pytest.raises(TrajectoryError) as caught:
        validate_r3_run_artifact(payload, tmp_path)
    assert caught.value.code == "ATTEMPT_LINEAGE_INVALID"


def test_environment_events_do_not_count_as_candidate_failures(tmp_path: Path) -> None:
    payload = _artifact(
        tmp_path,
        [
            ("TIMED_OUT", "ENVIRONMENT", "NONE"),
            ("INVALIDATED", "ENVIRONMENT", "EXECUTION"),
        ],
        "NOT_AVAILABLE",
    )
    artifact = validate_r3_run_artifact(payload, tmp_path)
    summary = summarize_trajectory(artifact)
    assert summary.candidate_failure_count == 0
    assert summary.timeout_count == 1
    assert summary.invalidation_count == 1


def test_standalone_attempt_validation_enforces_kind_and_parent(tmp_path: Path) -> None:
    identity = _identity()
    initial = _attempt(
        tmp_path,
        identity,
        0,
        "SUCCEEDED",
        "NONE",
        None,
    )
    initial["ordinal"] = 1
    initial["attempt_id"] = canonical_sha256({"run_id": identity.run_id, "ordinal": 1})
    initial["event_journal_locator"]["kind"] = f"journal-{initial['attempt_id']}"
    initial["output_artifact_locator"]["kind"] = f"candidate-{initial['attempt_id']}"
    _resign_attempt(initial)
    with pytest.raises(TrajectoryError) as caught:
        validate_attempt_record(
            initial,
            run_id=identity.run_id,
            resource_capabilities=_capabilities(),
            evidence_root=tmp_path,
        )
    assert caught.value.code == "ATTEMPT_LINEAGE_INVALID"

    repair = _attempt(
        tmp_path,
        identity,
        1,
        "SUCCEEDED",
        "NONE",
        None,
    )
    with pytest.raises(TrajectoryError) as caught:
        validate_attempt_record(
            repair,
            run_id=identity.run_id,
            resource_capabilities=_capabilities(),
            evidence_root=tmp_path,
        )
    assert caught.value.code == "ATTEMPT_LINEAGE_INVALID"


def test_attempt_and_run_digest_mismatches_fail_closed(tmp_path: Path) -> None:
    payload = _artifact(tmp_path, [("SUCCEEDED", "NONE", "NONE")], "PASS")
    payload["attempts"][0]["attempt_sha256"] = "0" * 64
    _resign_run(payload)
    with pytest.raises(TrajectoryError) as caught:
        validate_r3_run_artifact(payload, tmp_path)
    assert caught.value.code == "ATTEMPT_DIGEST_MISMATCH"

    payload = _artifact(tmp_path / "run", [("SUCCEEDED", "NONE", "NONE")], "PASS")
    payload["run_artifact_sha256"] = "0" * 64
    with pytest.raises(TrajectoryError) as caught:
        validate_r3_run_artifact(payload, tmp_path / "run")
    assert caught.value.code == "RUN_ARTIFACT_DIGEST_MISMATCH"


def test_terminal_disposition_compatibility_is_fail_closed(tmp_path: Path) -> None:
    payload = _artifact(tmp_path, [("SUCCEEDED", "NONE", "NONE")], "PASS")
    payload["attempts"][0]["disposition"] = "CANDIDATE"
    _resign_attempt(payload["attempts"][0])
    _resign_run(payload)
    with pytest.raises(TrajectoryError) as caught:
        validate_r3_run_artifact(payload, tmp_path)
    assert caught.value.code == "TERMINAL_DISPOSITION_INVALID"


def test_non_evaluable_final_attempt_cannot_claim_r2_rejection(tmp_path: Path) -> None:
    payload = _artifact(tmp_path, [("INVALIDATED", "ENVIRONMENT", "NONE")], "REJECTED")
    with pytest.raises(TrajectoryError) as caught:
        validate_r3_run_artifact(payload, tmp_path)
    assert caught.value.code == "FINAL_POINTER_MISMATCH"


def test_pass_requires_a_succeeded_final_attempt(tmp_path: Path) -> None:
    payload = _artifact(tmp_path, [("FAILED", "CANDIDATE", "NONE")], "PASS")
    with pytest.raises(TrajectoryError) as caught:
        validate_r3_run_artifact(payload, tmp_path)
    assert caught.value.code == "FINAL_POINTER_MISMATCH"


def test_unavailable_resources_are_explicit_and_never_materialized(tmp_path: Path) -> None:
    payload = _artifact(tmp_path, [("SUCCEEDED", "NONE", "NONE")], "PASS")
    artifact = validate_r3_run_artifact(payload, tmp_path)
    assert dict(artifact.aggregate_observed_resources) == payload["aggregate_observed_resources"]
    assert set(dict(artifact.aggregate_observed_resources)) == set(OBSERVED)
    assert all(
        name not in dict(artifact.aggregate_observed_resources)
        for name in RESOURCE_DIMENSIONS
        if name not in OBSERVED
    )


def test_non_json_direct_payload_is_wrapped_as_trajectory_error(tmp_path: Path) -> None:
    payload = _artifact(tmp_path, [("SUCCEEDED", "NONE", "NONE")], "PASS")
    payload["attempts"][0]["diagnostic_codes"] = {"not-json"}
    with pytest.raises(TrajectoryError) as caught:
        validate_r3_run_artifact(payload, tmp_path)
    assert caught.value.code == "SCHEMA_INVALID"
