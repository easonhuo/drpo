from __future__ import annotations

import os
import statistics
import sys
import time
from copy import deepcopy
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from drpo.workflow_replay.model import (  # noqa: E402
    ManifestError,
    load_case_manifest,
    validate_case_manifest,
)

FIXTURES = Path(__file__).parent / "fixtures" / "workflow_replay"
VALID = FIXTURES / "valid_code_only.yaml"


def payload() -> dict:
    return yaml.safe_load(VALID.read_text(encoding="utf-8"))


def test_valid_manifest_is_loaded_as_immutable_contract() -> None:
    manifest = load_case_manifest(VALID)
    assert manifest.case_id == "GOV-CODE-ONLY-01"
    assert manifest.benchmark["expected_terminal_state"] == "READY"
    with pytest.raises(TypeError):
        manifest.benchmark["expected_final_tree_or_semantic_hashes"]["extra"] = "x"


def test_valid_failure_outcome_has_explicit_boundary_and_no_repository_outcome() -> None:
    case = payload()
    case["case_id"] = "GOV-GATE-FAILURE-01"
    case["task_class"] = "gate_failure"
    case["benchmark"].update(
        expected_terminal_state="BLOCKED",
        expected_safety_boundary="focused_test_failure",
        expected_changed_paths=[],
        expected_final_tree_or_semantic_hashes={},
        replayability="partial",
        predeclared_exclusions=["external publication is outside local replay"],
    )
    assert validate_case_manifest(case).benchmark["expected_terminal_state"] == "BLOCKED"


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda case: case["benchmark"].update(auto_repair=True), "unknown keys"),
        (
            lambda case: case["benchmark"].update(expected_changed_paths=["../unsafe"]),
            "unsafe repository path",
        ),
        (lambda case: case["historical_task"].update(base_sha="bad"), "invalid hash syntax"),
        (lambda case: case.update(schema_version=True), "integer 1"),
        (
            lambda case: case["benchmark"].update(
                expected_changed_paths=[], expected_final_tree_or_semantic_hashes={}
            ),
            "READY outcome requires",
        ),
        (lambda case: case.update(task_class="integration"), "task_class must be one of"),
        (lambda case: case["benchmark"].update(post_hoc_exclusions=["slow case"]), "unknown keys"),
    ],
)
def test_invalid_contracts_fail_closed(mutate, message: str) -> None:
    case = deepcopy(payload())
    mutate(case)
    with pytest.raises(ManifestError, match=message):
        validate_case_manifest(case)


def test_negative_fixture_fails_closed() -> None:
    with pytest.raises(ManifestError, match="unknown keys"):
        load_case_manifest(FIXTURES / "invalid_unknown_key.yaml")


def test_manifest_symlink_is_rejected(tmp_path: Path) -> None:
    link = tmp_path / "case.yaml"
    try:
        os.symlink(VALID, link)
    except (OSError, NotImplementedError):
        pytest.skip("symlinks unavailable")
    with pytest.raises(ManifestError, match="symlink"):
        load_case_manifest(link)


def test_static_validation_runtime_guardrail() -> None:
    samples = []
    for _ in range(200):
        start = time.perf_counter_ns()
        load_case_manifest(VALID)
        samples.append((time.perf_counter_ns() - start) / 1_000_000_000)
    ordered = sorted(samples)
    assert statistics.median(ordered) <= 0.250
    assert ordered[int(0.95 * (len(ordered) - 1))] <= 1.000
