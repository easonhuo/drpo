from __future__ import annotations

import importlib.util
import subprocess
from datetime import date
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


def run_git(repo: Path, *args: str) -> str:
    proc = subprocess.run(
        ["git", "-C", str(repo), *args],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    return proc.stdout.strip()


def materialization_repo(tmp_path: Path) -> tuple[Path, str]:
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    run_git(repo, "config", "user.name", "DRPO Test")
    run_git(repo, "config", "user.email", "drpo-test@example.com")
    (repo / "docs").mkdir()
    (repo / "experiments").mkdir()
    (repo / "docs" / "handoff.md").write_text(
        "# Master v1\n\n## 0. Status\n\nBase state.\n",
        encoding="utf-8",
    )
    (repo / "experiments" / "registry.yaml").write_text(
        "schema_version: 2\nexperiments: []\n",
        encoding="utf-8",
    )
    run_git(repo, "add", ".")
    run_git(repo, "commit", "-q", "-m", "base")
    return repo, run_git(repo, "rev-parse", "HEAD")


def add_authoritative_delta(repo: Path, *, report: bool, materialized: bool) -> str:
    update_id = "TEST-RESULT-CLOSURE-01"
    block_id = "test-result-closure"
    delta_dir = repo / "docs" / "handoff_deltas" / update_id
    delta_dir.mkdir(parents=True)
    (delta_dir / "HANDOFF_DELTA.yaml").write_text(
        "schema_version: 3\n"
        f"update_id: {update_id}\n"
        "mode: authoritative\n"
        "operations:\n"
        "- operation_id: append-result\n"
        "  op: append_to_section\n"
        "  heading_path: [Master v1, 0. Status]\n"
        f"  block_id: {block_id}\n"
        "  content: result closure\n"
        "registry:\n"
        "  mode: unchanged\n"
        "  changes: []\n",
        encoding="utf-8",
    )
    if report:
        (delta_dir / "MATERIALIZATION_REPORT.json").write_text("{}\n", encoding="utf-8")
    if materialized:
        (repo / "docs" / "handoff.md").write_text(
            "# Master v1\n\n"
            "## 0. Status\n\n"
            "Base state.\n\n"
            f"<!-- HANDOFF-DELTA-BLOCK:section_end:{block_id}:START -->\n"
            "result closure\n"
            f"<!-- HANDOFF-DELTA-BLOCK:section_end:{block_id}:END -->\n",
            encoding="utf-8",
        )
    run_git(repo, "add", ".")
    run_git(repo, "commit", "-q", "-m", "closure")
    return run_git(repo, "rev-parse", "HEAD")


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


def test_yaml_date_values_compare_without_json_serialization() -> None:
    unchanged = {
        "id": "EXT-C-E8-TEST-01",
        "status": "not_run",
        "registered_on": date(2026, 7, 16),
    }
    result = module.validate_transition(
        {"EXT-C-E8-TEST-01": unchanged}, {"EXT-C-E8-TEST-01": dict(unchanged)}
    )
    assert result["changed_experiment_count"] == 0


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


def test_authoritative_delta_without_report_fails_closed(tmp_path: Path) -> None:
    repo, base = materialization_repo(tmp_path)
    head = add_authoritative_delta(repo, report=False, materialized=False)
    with pytest.raises(module.EvidenceLocatorError) as error:
        module.validate_handoff_materialization_transition(repo, base, head)
    assert error.value.code == "HANDOFF_MATERIALIZATION_MISSING"
    assert "MATERIALIZATION_REPORT.json" in error.value.message


def test_authoritative_delta_with_unmodified_handoff_fails_closed(tmp_path: Path) -> None:
    repo, base = materialization_repo(tmp_path)
    head = add_authoritative_delta(repo, report=True, materialized=False)
    with pytest.raises(module.EvidenceLocatorError) as error:
        module.validate_handoff_materialization_transition(repo, base, head)
    assert error.value.code == "HANDOFF_MATERIALIZATION_MISSING"
    assert "docs/handoff.md is unchanged" in error.value.message


def test_authoritative_delta_with_materialized_block_passes(tmp_path: Path) -> None:
    repo, base = materialization_repo(tmp_path)
    head = add_authoritative_delta(repo, report=True, materialized=True)
    result = module.validate_handoff_materialization_transition(repo, base, head)
    assert result["checked_handoff_materialization_count"] == 1
    assert result["checked_handoff_materialization_ids"] == ["TEST-RESULT-CLOSURE-01"]


def test_transition_without_authoritative_delta_is_unaffected(tmp_path: Path) -> None:
    repo, base = materialization_repo(tmp_path)
    (repo / "README.md").write_text("code-only\n", encoding="utf-8")
    run_git(repo, "add", ".")
    run_git(repo, "commit", "-q", "-m", "code only")
    head = run_git(repo, "rev-parse", "HEAD")
    result = module.validate_handoff_materialization_transition(repo, base, head)
    assert result["checked_handoff_materialization_count"] == 0
