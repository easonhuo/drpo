from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from drpo.workflow_replay.compare import OutcomeSnapshot  # noqa: E402
from drpo.workflow_replay.evidence import (  # noqa: E402
    NormalizedRun,
    RunIdentity,
    _load_acceptance_result,
    canonical_sha256,
    compare_normalized_runs,
    compare_semantic_runs,
    validate_acceptance_contract,
    validate_r1_case_contract,
)

BENCHMARK_ID = "REPLAYAB-R1-R2-DISCRIMINATION-01"
BASE_COMMIT = "b18aea9186d7e3ccc5d43b456719cafc23761e03"
BENCHMARK_ROOT = (
    ROOT
    / "docs"
    / "development_workflow_optimization"
    / "benchmarks"
    / "REPLAYAB_R1_R2_DISCRIMINATION_01"
)
INVENTORY_PATH = BENCHMARK_ROOT / "CASE_INVENTORY.yaml"
EXPECTED_PATH = BENCHMARK_ROOT / "EXPECTED_VERDICTS.yaml"
RAW_RESULTS_PATH = BENCHMARK_ROOT / "RAW_RESULTS.jsonl"
COMPARISON_PATH = BENCHMARK_ROOT / "PAIRED_COMPARISON.json"
DECISION_PATH = BENCHMARK_ROOT / "DECISION.md"
R1_CONTRACT_PATH = (
    ROOT / "docs" / "development_workflow_optimization" / "R1_IMPLEMENTATION_CONTRACT.md"
)
R2_CONTRACT_PATH = (
    ROOT / "docs" / "development_workflow_optimization" / "R2_IMPLEMENTATION_CONTRACT.md"
)

PAIR_PATTERN = {
    (True, True): "both_accepted",
    (True, False): "a_accepted_b_rejected",
    (False, True): "a_rejected_b_accepted",
    (False, False): "both_rejected",
}


def _load_yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _subject(case_id: str) -> bytes:
    return f"{BENCHMARK_ID}:{case_id}:frozen-subject-v1\n".encode()


def _common_benchmark(case: dict[str, Any]) -> dict[str, Any]:
    return {
        "toolchain_sha": BASE_COMMIT,
        "input_spec_sha256": _sha256(_subject(case["case_id"])),
        "expected_terminal_state": "READY",
        "expected_safety_boundary": None,
        "required_gates": ["pytest", "ruff"],
        "environment_id": "replayab-r1-r2-discrimination-v1",
        "cache_policy": "cold",
        "replayability": "complete",
        "predeclared_exclusions": [],
    }


def _historical_task(case: dict[str, Any], family: dict[str, Any]) -> dict[str, Any]:
    return {
        "base_sha": BASE_COMMIT,
        "frozen_implementation_sha": None,
        "source_prs": [family["recent_context_pr"]],
        "source_commits": [BASE_COMMIT],
        "historical_real_time_evidence": [],
    }


def _r1_contract(
    case: dict[str, Any], family: dict[str, Any], evaluator_sha256: str
):
    canonical = case["arms"][case["canonical_arm"]]
    benchmark = _common_benchmark(case)
    benchmark.update(
        expected_changed_paths=[canonical["changed_path"]],
        expected_final_tree_or_semantic_hashes={
            "artifact_sha256": canonical["artifact_sha256"]
        },
    )
    payload = {
        "schema_version": 2,
        "case_id": case["case_id"],
        "task_class": "code_only",
        "historical_task": _historical_task(case, family),
        "benchmark": benchmark,
        "r1": {
            "comparison_mode": "exact_artifact",
            "expected_file_modes": {canonical["changed_path"]: "100644"},
            "expected_authority_result": "PASS",
            "expected_gate_results": {"pytest": "PASS", "ruff": "PASS"},
            "expected_diagnostic_codes": [],
            "expected_recovery_class": None,
            "workspace_rule": "changed_as_expected",
            "evaluator_sha256": evaluator_sha256,
            "evidence_schema_sha256": _sha256(R1_CONTRACT_PATH.read_bytes()),
            "order_policy": "two_opposite_pairs",
        },
    }
    return validate_r1_case_contract(payload)


def _r2_contract(
    case: dict[str, Any],
    family: dict[str, Any],
    shared: dict[str, Any],
    evaluator_sha256: str,
):
    acceptance = {
        "mandatory_behaviors": shared["mandatory_behaviors"],
        "forbidden_regressions": shared["forbidden_regressions"],
        "tolerances": shared["tolerances"],
        "protected_paths": shared["protected_paths"],
        "evaluator_sha256": evaluator_sha256,
        "evidence_schema_sha256": _sha256(R2_CONTRACT_PATH.read_bytes()),
        "order_policy": "two_opposite_pairs",
    }
    benchmark = _common_benchmark(case)
    benchmark.update(
        expected_changed_paths=[family["allowed_root"]],
        expected_final_tree_or_semantic_hashes={
            "acceptance_contract_sha256": canonical_sha256(acceptance),
            "evaluator_sha256": evaluator_sha256,
        },
    )
    payload = {
        "schema_version": 3,
        "case_id": case["case_id"],
        "task_class": "code_only",
        "historical_task": _historical_task(case, family),
        "benchmark": benchmark,
        "acceptance": acceptance,
    }
    return validate_acceptance_contract(payload)


def _outcome(case: dict[str, Any], arm: str, environment_id: str) -> OutcomeSnapshot:
    item = case["arms"][arm]
    path = item["changed_path"]
    provenance = {
        "benchmark_toolchain_sha": BASE_COMMIT,
        "cache_policy": "cold",
        "environment_id": environment_id,
        "historical_base_sha": BASE_COMMIT,
        "input_spec_sha256": _sha256(_subject(case["case_id"])),
    }
    return OutcomeSnapshot(
        case_id=case["case_id"],
        terminal_state="READY",
        safety_boundary=None,
        changed_paths=(path,),
        file_modes=((path, "100644"),),
        output_hashes=(("artifact_sha256", item["artifact_sha256"]),),
        authority_result="PASS",
        gate_plan=("pytest", "ruff"),
        gate_results=(("pytest", "PASS"), ("ruff", "PASS")),
        provenance=tuple(sorted(provenance.items())),
        diagnostic_codes=(),
        partial_mutation=False,
        recovery_class=None,
    )


def _identity(case_id: str, arm: str, backend_id: str) -> RunIdentity:
    return RunIdentity.build(
        case_id,
        arm,
        "pair-0",
        0,
        0 if arm == "A" else 1,
        backend_id,
    )


def _evidence(outcome: OutcomeSnapshot, arm: str) -> tuple[tuple[str, str], ...]:
    value = {
        "case_id": outcome.case_id,
        "arm": arm,
        "changed_paths": outcome.changed_paths,
        "output_hashes": outcome.output_hashes,
    }
    return (
        ("outcome", canonical_sha256(value)),
        ("subject", _sha256(_subject(outcome.case_id))),
    )


def _timing(arm: str) -> tuple[tuple[str, int], ...]:
    total = 100 if arm == "A" else 101
    return (("child_ns", total), ("self_overhead_ns", 0), ("total_ns", total))


def _r1_judgment(case: dict[str, Any], family: dict[str, Any], evaluator_sha256: str):
    contract = _r1_contract(case, family, evaluator_sha256)
    runs = []
    for arm in ("A", "B"):
        outcome = _outcome(case, arm, contract.base.benchmark["environment_id"])
        runs.append(
            NormalizedRun(
                identity=_identity(case["case_id"], arm, "r1-discrimination"),
                evidence_sha256=_evidence(outcome, arm),
                execution_terminal="READY",
                outcome=outcome,
                timing=_timing(arm),
                execution_valid=True,
                acceptance=None,
            )
        )
    report = compare_normalized_runs(contract, runs[0], runs[1])
    accepted = {
        arm: not any(item.startswith(f"{arm}.") for item in report.mismatches)
        for arm in ("A", "B")
    }
    return report, accepted


def _facts(inventory: dict[str, Any], arm: dict[str, Any]) -> dict[str, Any]:
    if arm["facts"] == "correct":
        return inventory["shared_semantic_contract"]["correct_facts"]
    return inventory["fact_overrides"][arm["facts"]]


def _r2_judgment(
    case: dict[str, Any],
    family: dict[str, Any],
    inventory: dict[str, Any],
    evaluator_sha256: str,
):
    contract = _r2_contract(
        case, family, inventory["shared_semantic_contract"], evaluator_sha256
    )
    runs = []
    for arm in ("A", "B"):
        identity = _identity(case["case_id"], arm, "r2-discrimination")
        outcome = _outcome(case, arm, contract.base.benchmark["environment_id"])
        outcome_sha256 = canonical_sha256(
            {
                "case_id": outcome.case_id,
                "arm": arm,
                "changed_paths": outcome.changed_paths,
                "output_hashes": outcome.output_hashes,
            }
        )
        facts = _facts(inventory, case["arms"][arm])
        acceptance_payload = {
            "schema_version": 1,
            "case_id": case["case_id"],
            "run_id": identity.run_id,
            "outcome_sha256": outcome_sha256,
            "acceptance_contract_sha256": contract.acceptance_sha256,
            "evaluator_sha256": contract.r1["evaluator_sha256"],
            "mandatory_results": facts["mandatory_results"],
            "forbidden_results": facts["forbidden_results"],
            "tolerance_values": facts["tolerance_values"],
            "protected_paths_ok": facts["protected_paths_ok"],
            "diagnostic_codes": [],
        }
        raw = json.dumps(
            acceptance_payload, sort_keys=True, separators=(",", ":")
        ).encode()
        acceptance = _load_acceptance_result(
            raw,
            contract,
            identity,
            _sha256(raw),
            outcome_sha256,
        )
        runs.append(
            NormalizedRun(
                identity=identity,
                evidence_sha256=_evidence(outcome, arm)
                + (("acceptance", acceptance.evidence_sha256),),
                execution_terminal="READY",
                outcome=outcome,
                timing=_timing(arm),
                execution_valid=True,
                acceptance=acceptance,
            )
        )
    report = compare_semantic_runs(contract, runs[0], runs[1])
    return report, dict(report.arm_acceptance)


def _judge_metrics(rows: list[dict[str, Any]], judge: str) -> dict[str, Any]:
    correct = 0
    incorrect = 0
    false_rejections = 0
    false_acceptances = 0
    arm_correct = 0
    pair_correct = 0
    released = 0
    released_eligible = 0
    eligible = 0
    for row in rows:
        pair_ok = True
        for arm in ("A", "B"):
            truth = row["ground_truth"][arm] == "correct"
            accepted = row[judge][arm] == "accept"
            correct += int(truth)
            incorrect += int(not truth)
            false_rejections += int(truth and not accepted)
            false_acceptances += int(not truth and accepted)
            arm_correct += int(truth == accepted)
            pair_ok &= truth == accepted
        pair_correct += int(pair_ok)
        is_eligible = row["efficiency_eligible"]
        is_released = row[judge]["efficiency_released"]
        eligible += int(is_eligible)
        released += int(is_released)
        released_eligible += int(is_released and is_eligible)
    return {
        "correct_arms": correct,
        "incorrect_arms": incorrect,
        "correct_arm_false_rejections": false_rejections,
        "incorrect_arm_false_acceptances": false_acceptances,
        "correct_arm_frr": false_rejections / correct,
        "incorrect_arm_far": false_acceptances / incorrect,
        "arm_accuracy": arm_correct / (correct + incorrect),
        "pair_accuracy": pair_correct / len(rows),
        "efficiency_released_pairs": released,
        "efficiency_eligible_pairs": eligible,
        "efficiency_release_precision": released_eligible / released,
        "efficiency_release_coverage": released_eligible / eligible,
    }


def execute_benchmark() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    inventory = _load_yaml(INVENTORY_PATH)
    expected = _load_yaml(EXPECTED_PATH)
    evaluator_sha256 = _sha256(EXPECTED_PATH.read_bytes())
    rows: list[dict[str, Any]] = []
    for case in inventory["cases"]:
        family = inventory["families"][case["family"]]
        r1_report, r1_acceptance = _r1_judgment(case, family, evaluator_sha256)
        r2_report, r2_acceptance = _r2_judgment(
            case, family, inventory, evaluator_sha256
        )
        expected_case = expected["verdicts"][case["case_id"]]
        ground_truth = {
            arm: case["arms"][arm]["ground_truth"] for arm in ("A", "B")
        }
        expected_pattern = expected_case["pair_pattern"]
        assert expected_pattern == PAIR_PATTERN[
            (ground_truth["A"] == "correct", ground_truth["B"] == "correct")
        ]
        r2_pattern = PAIR_PATTERN[(r2_acceptance["A"], r2_acceptance["B"])]
        rows.append(
            {
                "schema_version": 1,
                "benchmark_id": BENCHMARK_ID,
                "case_id": case["case_id"],
                "family": case["family"],
                "pair_class": case["pair_class"],
                "canonical_arm": case["canonical_arm"],
                "ground_truth": ground_truth,
                "ground_truth_pair_pattern": expected_pattern,
                "efficiency_eligible": expected_case["efficiency_eligible"],
                "r1": {
                    "A": "accept" if r1_acceptance["A"] else "reject",
                    "B": "accept" if r1_acceptance["B"] else "reject",
                    "pair_pattern": PAIR_PATTERN[
                        (r1_acceptance["A"], r1_acceptance["B"])
                    ],
                    "efficiency_released": r1_report.equivalent,
                    "mismatches": list(r1_report.mismatches),
                },
                "r2": {
                    "A": "accept" if r2_acceptance["A"] else "reject",
                    "B": "accept" if r2_acceptance["B"] else "reject",
                    "pair_pattern": r2_pattern,
                    "efficiency_released": r2_report.efficiency_release_allowed,
                    "arm_failures": {
                        arm: list(failures)
                        for arm, failures in r2_report.arm_failures
                    },
                    "issues": list(r2_report.issues),
                },
            }
        )
    r1 = _judge_metrics(rows, "r1")
    r2 = _judge_metrics(rows, "r2")
    gate = expected["success_gate"]
    checks = {
        "r2_frr_reduction": (
            r1["correct_arm_frr"] - r2["correct_arm_frr"]
            >= gate["minimum_r2_frr_reduction"]
        ),
        "r2_zero_incorrect_far": (
            r2["incorrect_arm_far"] <= gate["maximum_r2_incorrect_arm_far"]
        ),
        "r2_far_not_above_r1": r2["incorrect_arm_far"] <= r1["incorrect_arm_far"],
        "r2_arm_accuracy": r2["arm_accuracy"] >= gate["minimum_r2_arm_accuracy"],
        "r2_pair_accuracy": r2["pair_accuracy"] >= gate["minimum_r2_pair_accuracy"],
        "r2_efficiency_precision": (
            r2["efficiency_release_precision"]
            >= gate["minimum_r2_efficiency_precision"]
        ),
        "r2_efficiency_coverage_gain": (
            r2["efficiency_release_coverage"]
            - r1["efficiency_release_coverage"]
            >= gate["minimum_r2_efficiency_coverage_gain"]
        ),
        "r1_identical_correct_controls": all(
            row["r1"]["pair_pattern"] == "both_accepted"
            for row in rows
            if row["pair_class"] == "identical_correct"
        ),
        "both_wrong_controls_rejected": all(
            row["r1"]["pair_pattern"] == "both_rejected"
            and row["r2"]["pair_pattern"] == "both_rejected"
            for row in rows
            if row["pair_class"] == "both_wrong"
        ),
    }
    comparison = {
        "schema_version": 1,
        "benchmark_id": BENCHMARK_ID,
        "base_commit": BASE_COMMIT,
        "evidence_grade": "C2_controlled_discrimination",
        "case_count": len(rows),
        "arm_count": 2 * len(rows),
        "r1": r1,
        "r2": r2,
        "deltas": {
            "correct_arm_frr_reduction": r1["correct_arm_frr"]
            - r2["correct_arm_frr"],
            "efficiency_release_coverage_gain": r2[
                "efficiency_release_coverage"
            ]
            - r1["efficiency_release_coverage"],
        },
        "judge_gate_checks": checks,
        "judge_gate_pass": all(checks.values()),
        "r1_terminal_non_regression": "REQUIRES_FULL_PR_TEST_SUITE",
        "decision_before_terminal_non_regression": (
            "PASS_CONTROLLED_ADVANTAGE_PENDING_TERMINAL_REGRESSION"
            if all(checks.values())
            else "FAIL_CONTROLLED_ADVANTAGE"
        ),
        "unsupported_claims": [
            "population_level_error_rates",
            "live_coding_agent_improvement",
            "candidate01_improvement",
            "r1_replacement",
            "r3_r4_r5_r6_completion",
        ],
    }
    return rows, comparison


def _raw_jsonl(rows: list[dict[str, Any]]) -> str:
    return "".join(
        json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
        for row in rows
    )


def _comparison_json(comparison: dict[str, Any]) -> str:
    return json.dumps(comparison, indent=2, sort_keys=True) + "\n"


def _decision_markdown(comparison: dict[str, Any]) -> str:
    r1 = comparison["r1"]
    r2 = comparison["r2"]
    return f"""# ReplayAB R1-versus-R2 Controlled Discrimination Decision

Work ID: `{BENCHMARK_ID}`

Base: `main@{BASE_COMMIT}`

Decision: `PASS_CONTROLLED_ADVANTAGE`

Evidence grade: `C2 -- controlled independently labelled semantic discrimination`

## Result

On the frozen 16-pair / 32-arm controlled bank, R2 reduced correct-arm false rejection from `{r1['correct_arm_frr']:.3f}` to `{r2['correct_arm_frr']:.3f}` while both judges retained an incorrect-arm false-acceptance rate of `{r2['incorrect_arm_far']:.3f}`.

R1 arm accuracy was `{r1['arm_accuracy']:.3f}` and pair accuracy was `{r1['pair_accuracy']:.3f}`. R2 arm and pair accuracy were both `{r2['arm_accuracy']:.3f}`. R1 released efficiency for `{r1['efficiency_released_pairs']}/{r1['efficiency_eligible_pairs']}` truly eligible pairs; R2 released `{r2['efficiency_released_pairs']}/{r2['efficiency_eligible_pairs']}`. Both release precisions were `{r2['efficiency_release_precision']:.3f}`.

The frozen judge-level success gate passed. The final PR validation also requires the pre-existing R1 terminal non-regression suite, focused ReplayAB suite, full repository pytest, Ruff, compilation, and governance checks to remain green.

## Supported claim

Within this frozen controlled bank, R2 is a strict judge-level capability extension over R1 exact-artifact mode: it preserves rejection of the predeclared incorrect outcomes while reducing false rejection caused solely by implementation non-identity.

## Not supported

This result is not a live coding-agent A/B, does not estimate population-level error rates, does not prove Candidate 01 improves work, does not replace R1 for deterministic exact-output tasks, and does not complete ReplayAB R3, R4, R5, or R6.

No scientific experiment, handoff state, registry state, or R2 closure state changed.
"""


def test_frozen_inventory_and_ground_truth_are_complete() -> None:
    inventory = _load_yaml(INVENTORY_PATH)
    expected = _load_yaml(EXPECTED_PATH)
    assert inventory["selection_frozen_before_execution"] is True
    assert inventory["post_selection_rule"] == {
        "no_case_removal_after_results": True,
        "no_label_change_after_results": True,
        "no_orientation_change_after_results": True,
        "retain_failures_and_invalid_cases": True,
    }
    assert len(inventory["cases"]) == expected["case_count"] == 16
    assert {case["case_id"] for case in inventory["cases"]} == set(
        expected["verdicts"]
    )
    assert {
        family: [case["pair_class"] for case in inventory["cases"] if case["family"] == family]
        for family in inventory["families"]
    } == {
        family: ["identical_correct", "different_correct", "mixed", "both_wrong"]
        for family in inventory["families"]
    }


def test_r1_r2_controlled_discrimination_judge_gate() -> None:
    rows, comparison = execute_benchmark()
    assert len(rows) == 16
    assert comparison["r1"]["correct_arms"] == 20
    assert comparison["r1"]["incorrect_arms"] == 12
    assert comparison["r1"]["correct_arm_false_rejections"] == 4
    assert comparison["r2"]["correct_arm_false_rejections"] == 0
    assert comparison["r1"]["incorrect_arm_false_acceptances"] == 0
    assert comparison["r2"]["incorrect_arm_false_acceptances"] == 0
    assert comparison["r1"]["arm_accuracy"] == pytest.approx(0.875)
    assert comparison["r1"]["pair_accuracy"] == pytest.approx(0.75)
    assert comparison["r2"]["arm_accuracy"] == pytest.approx(1.0)
    assert comparison["r2"]["pair_accuracy"] == pytest.approx(1.0)
    assert comparison["r1"]["efficiency_release_coverage"] == pytest.approx(0.5)
    assert comparison["r2"]["efficiency_release_coverage"] == pytest.approx(1.0)
    assert comparison["r1"]["efficiency_release_precision"] == pytest.approx(1.0)
    assert comparison["r2"]["efficiency_release_precision"] == pytest.approx(1.0)
    assert comparison["judge_gate_pass"] is True


def test_checked_in_evidence_matches_frozen_execution() -> None:
    evidence_paths = (RAW_RESULTS_PATH, COMPARISON_PATH, DECISION_PATH)
    if not all(path.exists() for path in evidence_paths):
        pytest.skip("post-run evidence has not yet been checked in")
    rows, comparison = execute_benchmark()
    assert RAW_RESULTS_PATH.read_text(encoding="utf-8") == _raw_jsonl(rows)
    assert COMPARISON_PATH.read_text(encoding="utf-8") == _comparison_json(comparison)
    assert DECISION_PATH.read_text(encoding="utf-8") == _decision_markdown(comparison)
