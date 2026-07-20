from __future__ import annotations

import hashlib
import json
import shutil
import statistics
import sys
import time
from dataclasses import replace
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from drpo.workflow_replay.compare import EquivalenceError  # noqa: E402
from drpo.workflow_replay.evidence import (  # noqa: E402
    EvidenceError,
    EvidenceLocator,
    build_opposite_order_schedule,
    build_semantic_opposite_order_schedule,
    compare_normalized_runs,
    compare_semantic_runs,
    load_acceptance_contract,
    load_r1_case_contract,
    load_run_artifact,
    release_bound_efficiency,
    release_semantic_efficiency,
    validate_r1_case_contract,
)
from drpo.workflow_replay.model import ManifestError, load_case_manifest  # noqa: E402

R1 = ROOT / "tests" / "fixtures" / "workflow_replay" / "r1"
CAL = ROOT / "docs" / "development_workflow_optimization" / "replayab_r1_calibration"
SOURCES = (
    "tests/fixtures/workflow_replay/valid_code_only.yaml",
    "tests/fixtures/workflow_replay/invalid_unknown_key.yaml",
)
EXPECTED_PATH = CAL / "EXPECTED_VERDICTS.yaml"
CONTRACT_PATH = CAL.parent / "R1_IMPLEMENTATION_CONTRACT.md"


def expected(case_id: str) -> dict[str, str]:
    return yaml.safe_load(EXPECTED_PATH.read_text(encoding="utf-8"))["verdicts"][case_id]


def git_blob_sha(path: Path) -> str:
    raw = path.read_bytes()
    return hashlib.sha1(f"blob {len(raw)}\0".encode() + raw).hexdigest()


def load_pair(case: str, repo: Path = ROOT):
    folder = repo / "tests" / "fixtures" / "workflow_replay" / "r1" / case
    manifest = load_r1_case_contract(folder / "manifest.yaml")
    arm_a = load_run_artifact(folder / "run-a.json", repo, manifest)
    arm_b = load_run_artifact(folder / "run-b.json", repo, manifest)
    return manifest, arm_a, arm_b


def copied_case(tmp_path: Path, case: str) -> Path:
    for source in SOURCES:
        target = tmp_path / source
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / source, target)
    target_case = tmp_path / "tests" / "fixtures" / "workflow_replay" / "r1" / case
    target_case.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(R1 / case, target_case)
    return target_case


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def locator(path: Path, repo: Path, kind: str) -> dict[str, object]:
    raw = path.read_bytes()
    return {
        "kind": kind,
        "relative_path": path.relative_to(repo).as_posix(),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "byte_size": len(raw),
    }


def mutate_outcome(repo: Path, case: str, arms: tuple[str, ...], mutate) -> None:
    folder = repo / "tests" / "fixtures" / "workflow_replay" / "r1" / case
    source = json.loads((folder / "outcome.json").read_text(encoding="utf-8"))
    mutate(source)
    for arm in arms:
        path = folder / f"outcome-{arm}.json"
        write_json(path, source)
        run_path = folder / f"run-{arm}.json"
        run = json.loads(run_path.read_text(encoding="utf-8"))
        run["evidence"]["outcome"] = locator(path, repo, "outcome")
        write_json(run_path, run)


def replace_terminal(repo: Path, case: str, arm: str, terminal: str, event: str) -> None:
    folder = repo / "tests" / "fixtures" / "workflow_replay" / "r1" / case
    run_path = folder / f"run-{arm}.json"
    run = json.loads(run_path.read_text(encoding="utf-8"))
    identity = run["run_identity"]
    rows = [
        {
            "run_id": identity["run_id"],
            "sequence": 0,
            "event": "run_started",
            "monotonic_ns": 100,
            "payload": {"arm": identity["arm"], "case_id": identity["case_id"]},
        },
        {
            "run_id": identity["run_id"],
            "sequence": 1,
            "event": event,
            "monotonic_ns": 200,
            "payload": {"terminal_state": terminal},
        },
    ]
    event_path = folder / f"events-{arm}-{terminal.lower()}.jsonl"
    event_path.write_text(
        "".join(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n" for row in rows),
        encoding="utf-8",
    )
    run["execution_terminal"] = terminal
    run["evidence"]["event_log"] = locator(event_path, repo, "event_log")
    run["evidence"]["outcome"] = None
    write_json(run_path, run)


def test_calibration_authority_was_frozen_before_behavior_results() -> None:
    inventory = yaml.safe_load((CAL / "INVENTORY.yaml").read_text(encoding="utf-8"))
    expected = yaml.safe_load((CAL / "EXPECTED_VERDICTS.yaml").read_text(encoding="utf-8"))
    ids = {item["case_id"] for item in inventory["cases"]}
    assert inventory["selection_frozen_before_behavior_results"] is True
    assert inventory["post_selection_rule"]["no_case_removal_after_results"] is True
    assert ids == set(expected["verdicts"])
    assert len(ids) == 10


def test_frozen_authority_digests_match_actual_files_and_source_blobs() -> None:
    inventory = yaml.safe_load((CAL / "INVENTORY.yaml").read_text(encoding="utf-8"))
    ready = load_r1_case_contract(R1 / "ready" / "manifest.yaml")
    failure = load_r1_case_contract(R1 / "failure" / "manifest.yaml")
    evaluator_sha = hashlib.sha256(EXPECTED_PATH.read_bytes()).hexdigest()
    assert ready.r1["evaluator_sha256"] == failure.r1["evaluator_sha256"] == evaluator_sha
    if CONTRACT_PATH.exists():
        schema_sha = hashlib.sha256(CONTRACT_PATH.read_bytes()).hexdigest()
        assert ready.r1["evidence_schema_sha256"] == failure.r1["evidence_schema_sha256"] == schema_sha
    for source in inventory["real_repository_sources"].values():
        assert git_blob_sha(ROOT / source["path"]) == source["source_blob_sha"]


def test_real_failure_source_is_rejected_by_existing_manifest_validator() -> None:
    with pytest.raises(ManifestError, match="unknown keys"):
        load_case_manifest(ROOT / SOURCES[1])


def test_schema_v1_remains_readable_and_schema_v2_is_explicit() -> None:
    legacy = load_case_manifest(ROOT / SOURCES[0])
    current = load_r1_case_contract(R1 / "ready" / "manifest.yaml")
    assert legacy.case_id == "GOV-CODE-ONLY-01"
    assert current.r1["comparison_mode"] == "exact_artifact"
    payload = yaml.safe_load((R1 / "ready" / "manifest.yaml").read_text(encoding="utf-8"))
    payload["r1"]["comparison_mode"] = "semantic_acceptance"
    with pytest.raises(EvidenceError, match="comparison_mode"):
        validate_r1_case_contract(payload)


def test_real_ready_and_failure_boundary_artifacts_match_frozen_verdicts() -> None:
    ready, ready_a, ready_b = load_pair("ready")
    failure, failure_a, failure_b = load_pair("failure")
    ready_report = compare_normalized_runs(ready, ready_a, ready_b)
    failure_report = compare_normalized_runs(failure, failure_a, failure_b)
    assert ready_report.equivalent
    assert failure_report.equivalent
    assert ready_a.execution_valid and failure_a.execution_valid
    payload = {
        "run_ids": ready_report.run_ids,
        "evidence_sha256": ready_report.evidence_sha256,
        "timing": ready_report.timing,
    }
    assert release_bound_efficiency(ready_report, payload) == ready_report.timing
    assert expected("R1-CAL-READY-EXACT") == {
        "expected": "equivalent", "efficiency_release": "allowed"
    }
    assert expected("R1-CAL-FAILURE-BOUNDARY") == {
        "expected": "equivalent_expected_stop", "efficiency_release": "allowed"
    }


def test_one_arm_hash_mismatch_is_rejected(tmp_path: Path) -> None:
    copied_case(tmp_path, "ready")
    mutate_outcome(
        tmp_path,
        "ready",
        ("b",),
        lambda outcome: outcome["output_hashes"].update(artifact_sha256="0" * 64),
    )
    manifest, arm_a, arm_b = load_pair("ready", tmp_path)
    report = compare_normalized_runs(manifest, arm_a, arm_b)
    assert not report.equivalent
    assert "B.output_hashes" in report.mismatches
    assert expected("R1-CAL-ONE-ARM-HASH-MISMATCH")["expected"] == "reject_b"


def test_both_arms_same_wrong_are_rejected(tmp_path: Path) -> None:
    copied_case(tmp_path, "ready")
    mutate_outcome(
        tmp_path,
        "ready",
        ("a", "b"),
        lambda outcome: outcome["output_hashes"].update(artifact_sha256="f" * 64),
    )
    manifest, arm_a, arm_b = load_pair("ready", tmp_path)
    report = compare_normalized_runs(manifest, arm_a, arm_b)
    assert {"A.output_hashes", "B.output_hashes"} <= set(report.mismatches)
    assert not report.equivalent
    assert expected("R1-CAL-BOTH-SAME-WRONG")["expected"] == "reject_both"


def test_wrong_file_mode_is_rejected_even_when_pair_peer_is_correct(tmp_path: Path) -> None:
    copied_case(tmp_path, "ready")
    mutate_outcome(
        tmp_path,
        "ready",
        ("b",),
        lambda outcome: outcome["file_modes"].update(
            {"tests/fixtures/workflow_replay/valid_code_only.yaml": "100755"}
        ),
    )
    manifest, arm_a, arm_b = load_pair("ready", tmp_path)
    report = compare_normalized_runs(manifest, arm_a, arm_b)
    assert "B.r1_contract" in report.mismatches
    assert expected("R1-CAL-WRONG-MODE")["expected"] == "reject_wrong_mode"


def test_interrupted_run_is_retained_but_cannot_compare_or_release(tmp_path: Path) -> None:
    copied_case(tmp_path, "ready")
    replace_terminal(tmp_path, "ready", "a", "INTERRUPTED", "run_interrupted")
    manifest, arm_a, arm_b = load_pair("ready", tmp_path)
    assert arm_a.execution_terminal == "INTERRUPTED"
    assert not arm_a.execution_valid and arm_a.outcome is None
    report = compare_normalized_runs(manifest, arm_a, arm_b)
    assert "A.execution_invalid" in report.mismatches
    with pytest.raises(EquivalenceError):
        release_bound_efficiency(report, {})
    assert expected("R1-CAL-INTERRUPTED")["expected"] == "execution_invalid"


def test_failure_boundary_with_workspace_mutation_is_invalid(tmp_path: Path) -> None:
    copied_case(tmp_path, "failure")
    run_path = tmp_path / "tests/fixtures/workflow_replay/r1/failure/run-a.json"
    run = json.loads(run_path.read_text(encoding="utf-8"))
    run["workspace_after_sha256"] = "0" * 64
    write_json(run_path, run)
    manifest, arm_a, arm_b = load_pair("failure", tmp_path)
    assert not arm_a.execution_valid
    report = compare_normalized_runs(manifest, arm_a, arm_b)
    assert "A.execution_invalid" in report.mismatches
    assert expected("R1-CAL-PARTIAL-MUTATION")["expected"] == "execution_invalid_partial_mutation"


def test_digest_identity_and_journal_tampering_fail_closed(tmp_path: Path) -> None:
    folder = copied_case(tmp_path, "ready")
    event = folder / "events-a.jsonl"
    event.write_text(event.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    manifest = load_r1_case_contract(folder / "manifest.yaml")
    with pytest.raises(EvidenceError, match="mismatch"):
        load_run_artifact(folder / "run-a.json", tmp_path, manifest)

    folder = copied_case(tmp_path / "identity", "ready")
    run_path = folder / "run-a.json"
    run = json.loads(run_path.read_text(encoding="utf-8"))
    run["run_identity"]["backend_id"] = "other"
    write_json(run_path, run)
    manifest = load_r1_case_contract(folder / "manifest.yaml")
    with pytest.raises(EvidenceError, match="run_id"):
        load_run_artifact(run_path, tmp_path / "identity", manifest)
    assert expected("R1-CAL-EVIDENCE-DIGEST-MISMATCH")["expected"] == "evidence_invalid"


def test_subject_substitution_fails_even_with_updated_locator(tmp_path: Path) -> None:
    folder = copied_case(tmp_path, "ready")
    subject = tmp_path / SOURCES[0]
    subject.write_text(subject.read_text(encoding="utf-8") + "# mutation\n", encoding="utf-8")
    run_path = folder / "run-a.json"
    run = json.loads(run_path.read_text(encoding="utf-8"))
    run["evidence"]["subject"] = locator(subject, tmp_path, "subject")
    write_json(run_path, run)
    manifest = load_r1_case_contract(folder / "manifest.yaml")
    with pytest.raises(EvidenceError, match="frozen input"):
        load_run_artifact(run_path, tmp_path, manifest)


def test_schedule_is_exactly_a_b_then_b_a_and_is_deterministic() -> None:
    manifest = load_r1_case_contract(R1 / "ready" / "manifest.yaml")
    first = build_opposite_order_schedule(manifest, "artifact-ingest")
    second = build_opposite_order_schedule(manifest, "artifact-ingest")
    assert first == second
    observed = [
        (item.pair_id, item.arm, item.order_position)
        for item in first
    ]
    assert observed == [
        ("pair-0", "A", 0),
        ("pair-0", "B", 1),
        ("pair-1", "B", 0),
        ("pair-1", "A", 1),
    ]
    assert len({item.run_id for item in first}) == 4
    assert expected("R1-CAL-ORDER-BALANCE")["expected"] == "schedule_exactly_a_b_then_b_a"


def test_timing_must_be_bound_to_exact_run_and_evidence_identities() -> None:
    manifest, arm_a, arm_b = load_pair("ready")
    report = compare_normalized_runs(manifest, arm_a, arm_b)
    good = {
        "run_ids": report.run_ids,
        "evidence_sha256": report.evidence_sha256,
        "timing": report.timing,
    }
    assert release_bound_efficiency(report, good) == report.timing
    bad = dict(good, run_ids=("wrong", "wrong"))
    with pytest.raises(EquivalenceError, match="run identities"):
        release_bound_efficiency(report, bad)
    with pytest.raises(EquivalenceError, match="timing"):
        release_bound_efficiency(report, dict(good, timing=()))
    with pytest.raises(EquivalenceError, match="report digest"):
        release_bound_efficiency(replace(report, timing=()), good)
    assert expected("R1-CAL-TIMING-BINDING-MISMATCH")["expected"] == (
        "comparison_may_pass_but_efficiency_blocked"
    )


@pytest.mark.parametrize(
    "mutation",
    [
        lambda payload: payload["r1"].update(expected_authority_result="READY"),
        lambda payload: payload["r1"]["expected_gate_results"].update(artifact_digest="MAYBE"),
        lambda payload: payload["r1"].update(expected_diagnostic_codes=["DUP", "DUP"]),
    ],
)
def test_r1_contract_rejects_invalid_expected_classes(mutation) -> None:
    payload = yaml.safe_load((R1 / "ready" / "manifest.yaml").read_text(encoding="utf-8"))
    mutation(payload)
    with pytest.raises(EvidenceError):
        validate_r1_case_contract(payload)


def test_evidence_locator_kind_must_match_its_declared_role(tmp_path: Path) -> None:
    folder = copied_case(tmp_path, "ready")
    run_path = folder / "run-a.json"
    run = json.loads(run_path.read_text(encoding="utf-8"))
    run["evidence"]["subject"]["kind"] = "result"
    write_json(run_path, run)
    manifest = load_r1_case_contract(folder / "manifest.yaml")
    with pytest.raises(EvidenceError, match="kind"):
        load_run_artifact(run_path, tmp_path, manifest)


def test_locator_rejects_path_escape_and_symlink(tmp_path: Path) -> None:
    outside = tmp_path.parent / "outside.txt"
    outside.write_text("x", encoding="utf-8")
    with pytest.raises(EvidenceError, match="unsafe"):
        EvidenceLocator("subject", "../outside.txt", hashlib.sha256(b"x").hexdigest(), 1)
    target = tmp_path / "target.txt"
    target.write_text("x", encoding="utf-8")
    link = tmp_path / "link.txt"
    try:
        link.symlink_to(target)
    except (OSError, NotImplementedError):
        pytest.skip("symlinks unavailable")
    value = EvidenceLocator("subject", "link.txt", hashlib.sha256(b"x").hexdigest(), 1)
    with pytest.raises(EvidenceError, match="symlink"):
        value.verify(tmp_path)


def test_r1_loading_and_reporting_runtime_guardrail() -> None:
    samples = []
    for _ in range(100):
        start = time.perf_counter_ns()
        manifest, arm_a, arm_b = load_pair("ready")
        assert compare_normalized_runs(manifest, arm_a, arm_b).equivalent
        samples.append((time.perf_counter_ns() - start) / 1_000_000_000)
    ordered = sorted(samples)
    assert statistics.median(ordered) <= 0.250
    assert ordered[int(0.95 * (len(ordered) - 1))] <= 1.000


def test_manifest_and_run_artifact_digests_are_stable() -> None:
    manifest = load_r1_case_contract(R1 / "ready" / "manifest.yaml")
    run = json.loads((R1 / "ready" / "run-a.json").read_text(encoding="utf-8"))
    assert run["case_contract_sha256"] == manifest.sha256


R2 = ROOT / "tests" / "fixtures" / "workflow_replay" / "r2" / "semantic"
R2_CAL = ROOT / "docs" / "development_workflow_optimization" / "replayab_r2_calibration"
R2_EXPECTED = R2_CAL / "EXPECTED_VERDICTS.yaml"


def expected_r2(case_id: str) -> dict[str, str]:
    return yaml.safe_load(R2_EXPECTED.read_text(encoding="utf-8"))["verdicts"][case_id]


def load_r2_pair(repo: Path = ROOT):
    folder = repo / "tests" / "fixtures" / "workflow_replay" / "r2" / "semantic"
    contract = load_acceptance_contract(folder / "manifest.yaml")
    arm_a = load_run_artifact(folder / "run-a.json", repo, contract)
    arm_b = load_run_artifact(folder / "run-b.json", repo, contract)
    return contract, arm_a, arm_b


def copied_r2(tmp_path: Path) -> Path:
    source = tmp_path / SOURCES[0]
    source.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(ROOT / SOURCES[0], source)
    target = tmp_path / "tests" / "fixtures" / "workflow_replay" / "r2" / "semantic"
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(R2, target)
    return target


def mutate_r2_result(repo: Path, arms: tuple[str, ...], mutate) -> None:
    folder = repo / "tests" / "fixtures" / "workflow_replay" / "r2" / "semantic"
    for arm in arms:
        result_path = folder / f"acceptance-{arm}.json"
        result = json.loads(result_path.read_text(encoding="utf-8"))
        mutate(result)
        write_json(result_path, result)
        run_path = folder / f"run-{arm}.json"
        run = json.loads(run_path.read_text(encoding="utf-8"))
        run["evidence"]["result"] = locator(result_path, repo, "result")
        write_json(run_path, run)


def test_r2_calibration_authority_is_frozen() -> None:
    inventory = yaml.safe_load((R2_CAL / "INVENTORY.yaml").read_text(encoding="utf-8"))
    expected = yaml.safe_load(R2_EXPECTED.read_text(encoding="utf-8"))
    assert inventory["selection_frozen_before_implementation_results"] is True
    assert inventory["post_selection_rule"]["no_case_removal_after_results"] is True
    assert {item["case_id"] for item in inventory["cases"]} == set(expected["verdicts"])
    assert len(expected["verdicts"]) == 12


def test_r2_contract_and_opposite_order_schedule() -> None:
    contract = load_acceptance_contract(R2 / "manifest.yaml")
    assert contract.r1["comparison_mode"] == "semantic_acceptance"
    assert contract.mandatory_behaviors == ("api_contract", "core_behavior")
    schedule = build_semantic_opposite_order_schedule(contract, "fixture")
    assert tuple(item.arm for item in schedule) == ("A", "B", "B", "A")
    assert tuple(item.order_position for item in schedule) == (0, 1, 0, 1)


def test_r2_different_correct_implementations_are_both_accepted() -> None:
    contract, arm_a, arm_b = load_r2_pair()
    assert arm_a.outcome is not None and arm_b.outcome is not None
    assert arm_a.outcome.output_hashes != arm_b.outcome.output_hashes
    assert arm_a.acceptance is not None and arm_a.acceptance.accepted
    assert arm_b.acceptance is not None and arm_b.acceptance.accepted
    report = compare_semantic_runs(contract, arm_a, arm_b)
    assert report.acceptance_pattern == "BOTH_ACCEPTED"
    assert report.pair_comparable and report.efficiency_release_allowed
    payload = {"run_ids": report.run_ids, "evidence_sha256": report.evidence_sha256, "timing": report.timing}
    assert release_semantic_efficiency(report, payload) == report.timing
    assert expected_r2("R2-CAL-DIFFERENT-BOTH-CORRECT")["expected"] == "both_accepted"


def test_r2_missing_mandatory_rejects_only_that_arm(tmp_path: Path) -> None:
    copied_r2(tmp_path)
    mutate_r2_result(tmp_path, ("b",), lambda result: result["mandatory_results"].update(api_contract=False))
    contract, arm_a, arm_b = load_r2_pair(tmp_path)
    report = compare_semantic_runs(contract, arm_a, arm_b)
    assert report.acceptance_pattern == "A_ACCEPTED_B_REJECTED"
    assert dict(report.arm_failures)["B"] == ("mandatory.api_contract",)
    with pytest.raises(EquivalenceError):
        release_semantic_efficiency(report, {})
    assert expected_r2("R2-CAL-MISSING-MANDATORY")["expected"] == "reject_b"


def test_r2_forbidden_regression_rejects_only_that_arm(tmp_path: Path) -> None:
    copied_r2(tmp_path)
    mutate_r2_result(tmp_path, ("a",), lambda result: result["forbidden_results"].update(deletes_required_behavior=True))
    contract, arm_a, arm_b = load_r2_pair(tmp_path)
    report = compare_semantic_runs(contract, arm_a, arm_b)
    assert report.acceptance_pattern == "A_REJECTED_B_ACCEPTED"
    assert dict(report.arm_failures)["A"] == ("forbidden.deletes_required_behavior",)
    assert expected_r2("R2-CAL-FORBIDDEN-REGRESSION")["expected"] == "reject_a"


def test_r2_both_same_wrong_are_rejected(tmp_path: Path) -> None:
    copied_r2(tmp_path)
    mutate_r2_result(tmp_path, ("a", "b"), lambda result: result["mandatory_results"].update(core_behavior=False))
    contract, arm_a, arm_b = load_r2_pair(tmp_path)
    report = compare_semantic_runs(contract, arm_a, arm_b)
    assert report.acceptance_pattern == "BOTH_REJECTED"
    assert not report.pair_comparable
    assert expected_r2("R2-CAL-BOTH-SAME-WRONG")["expected"] == "reject_both"


def test_r2_inclusive_tolerance_boundaries_are_accepted(tmp_path: Path) -> None:
    copied_r2(tmp_path)
    mutate_r2_result(tmp_path, ("a",), lambda result: result["tolerance_values"].update(score=0.9))
    mutate_r2_result(tmp_path, ("b",), lambda result: result["tolerance_values"].update(latency_ms=100.0, score=1.0))
    contract, arm_a, arm_b = load_r2_pair(tmp_path)
    report = compare_semantic_runs(contract, arm_a, arm_b)
    assert report.acceptance_pattern == "BOTH_ACCEPTED"
    assert expected_r2("R2-CAL-LOWER-BOUND")["expected"] == "accepted"
    assert expected_r2("R2-CAL-UPPER-BOUND")["expected"] == "accepted"


def test_r2_out_of_bound_value_is_rejected(tmp_path: Path) -> None:
    copied_r2(tmp_path)
    mutate_r2_result(tmp_path, ("b",), lambda result: result["tolerance_values"].update(score=0.899))
    contract, arm_a, arm_b = load_r2_pair(tmp_path)
    report = compare_semantic_runs(contract, arm_a, arm_b)
    assert report.acceptance_pattern == "A_ACCEPTED_B_REJECTED"
    assert dict(report.arm_failures)["B"] == ("tolerance.score.minimum",)
    assert expected_r2("R2-CAL-TOLERANCE-VIOLATION")["expected"] == "reject_b"


@pytest.mark.parametrize(
    ("field", "value", "case_id"),
    [
        ("evaluator_sha256", "0" * 64, "R2-CAL-EVALUATOR-MISMATCH"),
        ("acceptance_contract_sha256", "0" * 64, "R2-CAL-CONTRACT-MISMATCH"),
        ("run_id", "0" * 64, "R2-CAL-WRONG-RUN-BINDING"),
    ],
)
def test_r2_identity_mismatches_fail_closed(tmp_path: Path, field: str, value: str, case_id: str) -> None:
    folder = copied_r2(tmp_path)
    mutate_r2_result(tmp_path, ("b",), lambda result: result.__setitem__(field, value))
    contract = load_acceptance_contract(folder / "manifest.yaml")
    with pytest.raises(EvidenceError, match="identity mismatch"):
        load_run_artifact(folder / "run-b.json", tmp_path, contract)
    assert expected_r2(case_id)["expected"] == "evidence_invalid"


def test_r2_protected_path_failure_rejects_arm(tmp_path: Path) -> None:
    copied_r2(tmp_path)
    mutate_r2_result(tmp_path, ("b",), lambda result: result.__setitem__("protected_paths_ok", False))
    contract, arm_a, arm_b = load_r2_pair(tmp_path)
    report = compare_semantic_runs(contract, arm_a, arm_b)
    assert report.acceptance_pattern == "A_ACCEPTED_B_REJECTED"
    assert dict(report.arm_failures)["B"] == ("protected_paths",)
    assert expected_r2("R2-CAL-PROTECTED-PATH")["expected"] == "reject_b"


def test_r2_rejected_arm_never_releases_efficiency(tmp_path: Path) -> None:
    copied_r2(tmp_path)
    mutate_r2_result(tmp_path, ("a",), lambda result: result["mandatory_results"].update(api_contract=False))
    contract, arm_a, arm_b = load_r2_pair(tmp_path)
    report = compare_semantic_runs(contract, arm_a, arm_b)
    with pytest.raises(EquivalenceError, match="two accepted arms"):
        release_semantic_efficiency(report, {"run_ids": report.run_ids, "evidence_sha256": report.evidence_sha256, "timing": report.timing})
    assert expected_r2("R2-CAL-EFFICIENCY-BLOCK")["expected"] == "blocked"
