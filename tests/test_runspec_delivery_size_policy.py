from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
AGENT_DIR = ROOT / "scripts" / "agent"
if str(AGENT_DIR) not in sys.path:
    sys.path.insert(0, str(AGENT_DIR))

import run_claimed_runspec as runner  # noqa: E402
import runspec_results_delivery as results_delivery  # noqa: E402
from runspec_delivery_policy import (  # noqa: E402
    RESULT_TOO_LARGE,
    is_result_too_large_error,
    validate_simple_size_policy,
)
from runspec_lib import RunSpecError  # noqa: E402


def delivery_spec() -> dict:
    return {
        "version": 1,
        "run_id": "E7-SIZE-POLICY-1",
        "lane": "e7",
        "experiment_id": "EXT-H-E7-SIZE-POLICY-01",
        "delivery": {
            "enabled": True,
            "auto": True,
            "mode": "results_repo",
            "repository": "easonhuo/drpo-results",
            "branch": "ingest/e7",
            "export_profile": "manifest_text_v1",
            "max_total_size_mb": 30,
            "max_file_size_mb": 10,
        },
        "publish": {"enabled": False, "auto": False},
    }


def test_v1_size_limits_are_hard_caps() -> None:
    spec = delivery_spec()
    assert validate_simple_size_policy(spec) == {
        "max_total_size_mb": 30,
        "max_file_size_mb": 10,
    }

    spec["delivery"]["max_total_size_mb"] = 31
    with pytest.raises(RunSpecError, match="between 1 and 30"):
        validate_simple_size_policy(spec)

    spec["delivery"]["max_total_size_mb"] = 30
    spec["delivery"]["max_file_size_mb"] = 11
    with pytest.raises(RunSpecError, match="between 1 and 10"):
        validate_simple_size_policy(spec)


def test_size_errors_are_classified_without_hiding_other_failures() -> None:
    assert is_result_too_large_error(
        RunSpecError("delivery review package is too large: 40 > 30 bytes")
    )
    assert is_result_too_large_error(
        RunSpecError("generated delivery file exceeds max_file_size_mb: results.jsonl")
    )
    assert not is_result_too_large_error(RunSpecError("git push failed"))


def test_auto_delivery_oversize_keeps_done_and_returns_success(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo = tmp_path / "repo"
    claimed = repo / ".runspec_state" / "claimed" / "E7-SIZE-POLICY-1.yaml"
    claimed.parent.mkdir(parents=True)
    claimed.write_text(yaml.safe_dump(delivery_spec(), sort_keys=False), encoding="utf-8")

    monkeypatch.setattr(runner, "validate_provenance", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        runner,
        "validate_recovery_policy",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        runner,
        "run_entrypoint_with_recovery",
        lambda *_args, **_kwargs: {
            "returncode": 0,
            "attempts": 1,
            "recovery_used": False,
            "recovery_report": None,
        },
    )
    monkeypatch.setattr(
        runner,
        "package_artifacts_safe",
        lambda *_args, **_kwargs: {
            "zip_path": "runspec_artifacts/E7-SIZE-POLICY-1_results.zip",
            "zip_sha256": "a" * 64,
        },
    )

    def too_large(*_args, **_kwargs):
        raise RunSpecError(
            "delivery review package is too large: 41943040 > 31457280 bytes"
        )

    monkeypatch.setattr(results_delivery, "deliver_completed_run", too_large)

    payload, code = runner.execute_claimed_runspec(repo, claimed)

    assert code == 0
    assert payload["status"] == "PASS"
    assert payload["delivery_status"] == RESULT_TOO_LARGE
    assert payload["delivery_upload_attempted"] is False
    assert payload["local_artifact_zip"].endswith("_results.zip")
    assert (repo / ".runspec_state" / "done" / "E7-SIZE-POLICY-1.yaml").is_file()
    report = json.loads(
        (
            repo
            / ".runspec_state"
            / "delivery"
            / "E7-SIZE-POLICY-1"
            / "DELIVERY_REPORT.json"
        ).read_text(encoding="utf-8")
    )
    assert report["status"] == RESULT_TOO_LARGE
    assert report["experiment_state"] == "done"
    assert report["upload_attempted"] is False
    assert report["max_total_size_mb"] == 30
    assert report["max_file_size_mb"] == 10
