from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "validate_evidence_locator.py"
SPEC = importlib.util.spec_from_file_location("validate_evidence_locator", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(module)


def record(run_id: str = "E8_TEST_20260716_01") -> dict[str, str]:
    return {
        "run_id": run_id,
        "lane": "e8",
        "source_commit": "1" * 40,
        "results_repository": "easonhuo/drpo-results",
        "results_branch": "ingest/e8",
        "results_commit": "2" * 40,
        "result_path": f"runs/e8/{run_id}",
        "manifest_sha256": "3" * 64,
        "export_profile": "manifest_text_v1",
    }


def locator(*records: dict[str, str]) -> dict[str, object]:
    rows = list(records) or [record()]
    return {"schema_version": 1, "primary_run_id": rows[-1]["run_id"], "records": rows}


def delivered(locator_value: object | None = None) -> dict[str, object]:
    value: dict[str, object] = {
        "id": "EXT-C-E8-TEST-01",
        "status": "pilot",
        "execution": {"state": "delivered", "run_id": "E8_TEST_20260716_01"},
    }
    if locator_value is not None:
        value["evidence_locator"] = locator_value
    return value


def test_valid_results_repo_locator() -> None:
    normalized = module.validate_locator("EXT-C-E8-TEST-01", locator())
    assert normalized["primary_run_id"] == "E8_TEST_20260716_01"
    assert normalized["records"][0]["results_commit"] == "2" * 40


def test_changed_delivered_experiment_requires_locator() -> None:
    before = {"EXT-C-E8-TEST-01": {"id": "EXT-C-E8-TEST-01", "status": "not_run"}}
    after = {"EXT-C-E8-TEST-01": delivered()}
    with pytest.raises(module.EvidenceLocatorError, match="no evidence_locator") as error:
        module.validate_transition(before, after)
    assert error.value.code == "EVIDENCE_LOCATOR_MISSING"


def test_not_run_change_is_not_forced_to_claim_delivery() -> None:
    before = {"EXT-C-E8-TEST-01": {"id": "EXT-C-E8-TEST-01", "status": "not_run"}}
    after = {
        "EXT-C-E8-TEST-01": {
            "id": "EXT-C-E8-TEST-01",
            "status": "not_run",
            "note": "implementation updated",
        }
    }
    result = module.validate_transition(before, after)
    assert result["checked_locator_count"] == 0


@pytest.mark.parametrize(
    ("field", "bad_value"),
    [
        ("source_commit", "1" * 39),
        ("results_commit", "2" * 39),
        ("manifest_sha256", "3" * 63),
        ("results_repository", "easonhuo/drpo"),
        ("results_branch", "main"),
        ("result_path", "../escape"),
    ],
)
def test_malformed_record_fails_closed(field: str, bad_value: str) -> None:
    row = record()
    row[field] = bad_value
    with pytest.raises(module.EvidenceLocatorError):
        module.validate_locator("EXT-C-E8-TEST-01", locator(row))


def test_existing_record_cannot_be_removed_or_mutated() -> None:
    first = record()
    before = {"EXT-C-E8-TEST-01": delivered(locator(first))}
    mutated = dict(first)
    mutated["manifest_sha256"] = "4" * 64
    after = {"EXT-C-E8-TEST-01": delivered(locator(mutated))}
    with pytest.raises(module.EvidenceLocatorError) as error:
        module.validate_transition(before, after)
    assert error.value.code == "EVIDENCE_LOCATOR_MUTATED"


def test_new_record_must_append_and_can_become_primary() -> None:
    first = record()
    second = record("E8_TEST_20260716_02")
    second["results_commit"] = "4" * 40
    second["manifest_sha256"] = "5" * 64
    before = {"EXT-C-E8-TEST-01": delivered(locator(first))}
    after = {"EXT-C-E8-TEST-01": delivered(locator(first, second))}
    result = module.validate_transition(before, after)
    assert result["checked_locator_ids"] == ["EXT-C-E8-TEST-01"]


def test_current_mode_grandfathers_untouched_legacy_delivery() -> None:
    result = module.validate_current({"EXT-C-E8-TEST-01": delivered()})
    assert result["grandfathered_missing_ids"] == ["EXT-C-E8-TEST-01"]
