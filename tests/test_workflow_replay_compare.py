from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest
import yaml

from drpo.workflow_replay.compare import (
    EquivalenceError,
    OutcomeSnapshot,
    compare_outcomes,
    release_efficiency_payload,
)
from drpo.workflow_replay.model import CaseManifest, validate_case_manifest

FIXTURE = Path(__file__).parent / "fixtures" / "workflow_replay" / "valid_code_only.yaml"


def manifest() -> CaseManifest:
    payload = yaml.safe_load(FIXTURE.read_text(encoding="utf-8"))
    payload["benchmark"]["expected_final_tree_or_semantic_hashes"].update(
        delta_semantic_sha256="d" * 64,
        handoff_materialized_sha256="b" * 64,
        registry_semantic_sha256="c" * 64,
    )
    return validate_case_manifest(payload)


def provenance(case: CaseManifest) -> tuple[tuple[str, str], ...]:
    historical = case.historical_task
    benchmark = case.benchmark
    values = {
        "benchmark_toolchain_sha": benchmark["toolchain_sha"],
        "cache_policy": benchmark["cache_policy"],
        "environment_id": benchmark["environment_id"],
        "historical_base_sha": historical["base_sha"],
        "input_spec_sha256": benchmark["input_spec_sha256"],
    }
    if historical["frozen_implementation_sha"] is not None:
        values["frozen_implementation_sha"] = historical["frozen_implementation_sha"]
    return tuple(sorted(values.items()))


def ready(case: CaseManifest) -> OutcomeSnapshot:
    gate = case.benchmark["required_gates"][0]
    return OutcomeSnapshot(
        case.case_id,
        "READY",
        None,
        tuple(sorted(case.benchmark["expected_changed_paths"])),
        (("docs/example.md", "100644"),),
        tuple(sorted(case.benchmark["expected_final_tree_or_semantic_hashes"].items())),
        "PASS",
        (gate,),
        ((gate, "PASS"),),
        provenance(case),
    )


def test_ready_equivalence_releases_efficiency_payload() -> None:
    case = manifest()
    report = compare_outcomes(case, ready(case), ready(case))
    assert report.equivalent
    assert release_efficiency_payload(report, {"wall_ns": 1}) == {"wall_ns": 1}


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("authority_result", "FAIL"),
        ("gate_results", (("python3 -m pytest tests/test_example.py -q", "FAIL"),)),
        ("file_modes", (("docs/example.md", "100755"),)),
    ],
)
def test_pair_mismatches_block_efficiency(field: str, value) -> None:
    case = manifest()
    report = compare_outcomes(case, ready(case), replace(ready(case), **{field: value}))
    assert not report.equivalent
    assert f"pair.{field}" in report.mismatches
    with pytest.raises(EquivalenceError):
        release_efficiency_payload(report, {})


def test_provenance_mismatch_blocks_efficiency() -> None:
    case = manifest()
    changed = dict(provenance(case))
    changed["input_spec_sha256"] = "0" * 64
    report = compare_outcomes(
        case,
        ready(case),
        replace(ready(case), provenance=tuple(sorted(changed.items()))),
    )
    assert not report.equivalent
    assert "B.provenance.input_spec_sha256" in report.mismatches


@pytest.mark.parametrize(
    "key",
    [
        "final_tree_sha",
        "registry_semantic_sha256",
        "handoff_materialized_sha256",
        "delta_semantic_sha256",
    ],
)
def test_protected_hash_mismatches_are_detected(key: str) -> None:
    case = manifest()
    changed = dict(ready(case).output_hashes)
    changed[key] = "0" * len(changed[key])
    report = compare_outcomes(
        case,
        ready(case),
        replace(ready(case), output_hashes=tuple(sorted(changed.items()))),
    )
    assert not report.equivalent
    assert "B.output_hashes" in report.mismatches


def test_terminal_state_mismatch_is_detected() -> None:
    case = manifest()
    gate = case.benchmark["required_gates"][0]
    blocked = replace(
        ready(case),
        terminal_state="BLOCKED",
        safety_boundary="gate",
        changed_paths=(),
        file_modes=(),
        output_hashes=(),
        authority_result="BLOCKED",
        gate_results=((gate, "FAIL"),),
        diagnostic_codes=("GATE_FAIL",),
        recovery_class="repair-input",
    )
    report = compare_outcomes(case, ready(case), blocked)
    assert not report.equivalent
    assert "B.terminal_state" in report.mismatches


def test_both_arms_equally_wrong_still_fail_frozen_manifest() -> None:
    case = manifest()
    changed = dict(ready(case).output_hashes)
    changed["final_tree_sha"] = "0" * 40
    wrong = replace(ready(case), output_hashes=tuple(sorted(changed.items())))
    report = compare_outcomes(case, wrong, wrong)
    assert not report.equivalent
    assert {"A.output_hashes", "B.output_hashes"} <= set(report.mismatches)


def test_both_arms_wrong_provenance_fail_manifest() -> None:
    case = manifest()
    changed = dict(provenance(case))
    changed["historical_base_sha"] = "0" * 40
    wrong = replace(ready(case), provenance=tuple(sorted(changed.items())))
    report = compare_outcomes(case, wrong, wrong)
    assert not report.equivalent
    assert "A.provenance.historical_base_sha" in report.mismatches


def failure_manifest() -> CaseManifest:
    payload = yaml.safe_load(FIXTURE.read_text(encoding="utf-8"))
    payload["case_id"] = "GOV-GATE-FAILURE-01"
    payload["task_class"] = "gate_failure"
    payload["benchmark"].update(
        expected_terminal_state="BLOCKED",
        expected_safety_boundary="focused_test_failure",
        expected_changed_paths=[],
        expected_final_tree_or_semantic_hashes={},
    )
    return validate_case_manifest(payload)


def failure(case: CaseManifest) -> OutcomeSnapshot:
    gate = case.benchmark["required_gates"][0]
    return OutcomeSnapshot(
        case.case_id,
        "BLOCKED",
        "focused_test_failure",
        (),
        (),
        (),
        "BLOCKED",
        (gate,),
        ((gate, "FAIL"),),
        provenance(case),
        ("TEST_FAIL",),
        False,
        "repair-input",
    )


def test_equivalent_fail_closed_outcomes_pass() -> None:
    case = failure_manifest()
    assert compare_outcomes(case, failure(case), failure(case)).equivalent


def test_failure_partial_mutation_and_recovery_mismatch_are_detected() -> None:
    case = failure_manifest()
    bad = replace(failure(case), partial_mutation=True, recovery_class="unsafe-retry")
    report = compare_outcomes(case, failure(case), bad)
    assert not report.equivalent
    assert "B.partial_mutation" in report.mismatches
    assert "pair.recovery_class" in report.mismatches


def test_malformed_duplicate_pairs_fail_closed() -> None:
    case = manifest()
    malformed = replace(ready(case), provenance=(("x", "1"), ("x", "2")))
    with pytest.raises(EquivalenceError, match="provenance"):
        compare_outcomes(case, ready(case), malformed)
