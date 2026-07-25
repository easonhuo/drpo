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
import runspec_safety as safety  # noqa: E402
from runspec_delivery_policy import (  # noqa: E402
    RESULT_TOO_LARGE,
    formal_delivery_required,
    is_result_too_large_error,
    validate_formal_delivery_policy,
    validate_simple_size_policy,
)
from runspec_lib import RunSpecError  # noqa: E402


def delivery_spec() -> dict:
    return {
        "version": 1,
        "run_id": "E7-SIZE-POLICY-1",
        "lane": "e7",
        "experiment_id": "EXT-H-E7-SIZE-POLICY-01",
        "policy": {"formal_evidence_allowed": True},
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


def test_formal_delivery_is_fail_closed_and_local_only_is_explicit() -> None:
    missing_declaration = {
        "lane": "e7",
        "policy": {},
        "delivery": {"enabled": False, "auto": False},
    }
    assert formal_delivery_required(missing_declaration) is True
    with pytest.raises(RunSpecError, match="formal RunSpec requires"):
        validate_formal_delivery_policy(missing_declaration)

    explicit_formal = {
        "lane": "e7",
        "policy": {"formal_evidence_allowed": True},
        "delivery": {"enabled": True, "auto": False},
    }
    with pytest.raises(RunSpecError, match="delivery.enabled=true and delivery.auto=true"):
        validate_formal_delivery_policy(explicit_formal)

    local_only = {
        "lane": "e7",
        "policy": {"formal_evidence_allowed": False},
        "delivery": {"enabled": False, "auto": False},
    }
    assert formal_delivery_required(local_only) is False
    assert validate_formal_delivery_policy(local_only) is False
    assert validate_simple_size_policy(local_only) is None


def test_formal_delivery_requires_canonical_repository_and_lane_branch() -> None:
    spec = delivery_spec()
    assert validate_formal_delivery_policy(spec) is True

    spec["delivery"]["repository"] = "easonhuo/other-results"
    with pytest.raises(RunSpecError, match="easonhuo/drpo-results"):
        validate_formal_delivery_policy(spec)

    spec = delivery_spec()
    spec["delivery"]["branch"] = "ingest/e8"
    with pytest.raises(RunSpecError, match="ingest/e7"):
        validate_formal_delivery_policy(spec)


def test_formal_declaration_must_be_boolean() -> None:
    spec = delivery_spec()
    spec["policy"]["formal_evidence_allowed"] = "yes"
    with pytest.raises(RunSpecError, match="must be a boolean"):
        formal_delivery_required(spec)


def test_claim_rejects_missing_formal_delivery_before_state_creation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo = tmp_path / "repo"
    ready = repo / "runspecs" / "ready" / "E7-FORMAL-MISSING-DELIVERY.yaml"
    ready.parent.mkdir(parents=True)
    ready.write_text("version: 1\n", encoding="utf-8")
    spec = {
        "version": 1,
        "run_id": "E7-FORMAL-MISSING-DELIVERY",
        "lane": "e7",
        "experiment_id": "EXT-H-E7-FORMAL-MISSING-DELIVERY",
        "policy": {
            "existing_script_required": True,
            "forbid_new_launcher": True,
            "forbid_hparam_change": True,
            "forbid_cross_lane": True,
        },
        "delivery": {"enabled": False, "auto": False},
    }

    monkeypatch.setattr(safety, "read_yaml", lambda *_args, **_kwargs: dict(spec))
    monkeypatch.setattr(
        safety,
        "validate_registration_block",
        lambda *_args, **_kwargs: {"mode": "deferred", "closure_required": True},
    )
    monkeypatch.setattr(
        safety,
        "registration_requires_registry",
        lambda *_args, **_kwargs: False,
    )
    monkeypatch.setattr(
        safety,
        "validate_runspec",
        lambda *_args, **_kwargs: dict(spec),
    )
    monkeypatch.setattr(safety, "validate_recovery_policy", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(safety, "validate_delivery_block", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(safety, "validate_provenance", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(safety, "iter_ready_specs", lambda *_args, **_kwargs: [ready])

    with pytest.raises(RunSpecError, match="NO_READY_TASK") as rejected:
        safety.claim_next_runspec_safe(repo, lane_config={"lane": "e7"})

    assert "formal RunSpec requires" in str(rejected.value)
    assert not (
        repo
        / ".runspec_state"
        / "claimed"
        / "E7-FORMAL-MISSING-DELIVERY.yaml"
    ).exists()


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


@pytest.mark.parametrize("lane", ["e7", "e8"])
def test_missing_delivery_defaults_to_lane_results_repo(lane: str) -> None:
    spec = {"lane": lane}
    runner.apply_default_results_delivery(spec)
    assert spec["delivery"] == {
        "enabled": True,
        "auto": True,
        "mode": "results_repo",
        "repository": "easonhuo/drpo-results",
        "branch": f"ingest/{lane}",
        "export_profile": "manifest_text_v1",
        "max_total_size_mb": 30,
        "max_file_size_mb": 10,
    }


def test_explicit_disabled_delivery_is_preserved() -> None:
    spec = {"lane": "e8", "delivery": {"enabled": False, "auto": False}}
    runner.apply_default_results_delivery(spec)
    assert spec["delivery"]["enabled"] is False


def test_auto_delivery_oversize_keeps_done_and_returns_success(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo = tmp_path / "repo"
    claimed = repo / ".runspec_state" / "claimed" / "E7-SIZE-POLICY-1.yaml"
    claimed.parent.mkdir(parents=True)
    spec = delivery_spec()
    spec.pop("delivery")
    claimed.write_text(yaml.safe_dump(spec, sort_keys=False), encoding="utf-8")

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
    done = repo / ".runspec_state" / "done" / "E7-SIZE-POLICY-1.yaml"
    persisted = yaml.safe_load(done.read_text(encoding="utf-8"))
    assert persisted["delivery"]["branch"] == "ingest/e7"
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
