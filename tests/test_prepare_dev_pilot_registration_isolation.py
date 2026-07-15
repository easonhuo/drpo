from __future__ import annotations

import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from prepare_dev_pilot_registration import PreparationError, prepare  # noqa: E402


def approved_spec() -> dict:
    return {
        "schema_version": 1,
        "preparation_id": "GOV-OUTPUT-ISOLATION-PREP-01",
        "source": {
            "repository": "easonhuo/drpo",
            "remote": "origin",
            "main_ref": "refs/heads/main",
            "expected_main_sha": "1" * 40,
            "dev_branch": "dev/output-isolation",
            "expected_dev_sha": "2" * 40,
            "result_commit_sha": "2" * 40,
            "result_git_dirty": False,
        },
        "subject": {
            "experiment_id": "OUTPUT-ISOLATION-EXPERIMENT-01",
            "governance_claims": [],
        },
        "implementation": {
            "operations": [
                {
                    "op": "add",
                    "source_path": "src/drpo/output_isolation.py",
                    "destination_path": "src/drpo/output_isolation.py",
                    "expected_blob_sha": "3" * 40,
                    "expected_old_blob_sha": None,
                    "expected_mode": "100644",
                }
            ]
        },
        "review": {
            "reviewer_id": "chatgpt-reviewer",
            "decision_token": "output-isolation-review",
            "decision": {
                "approved": True,
                "code_integration_eligible": True,
                "evidence_level": "pilot",
                "result_status": "pilot",
                "claim_support_level": "diagnostic",
                "terminal_audit": "partial",
                "task_performance_collapse": "inconclusive",
                "support_boundary": "inconclusive",
                "numerical_failure": "none",
            },
            "limitations": [],
            "unresolved": [],
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


def write_spec(path: Path) -> None:
    path.write_text(
        yaml.safe_dump(approved_spec(), sort_keys=False, width=1000),
        encoding="utf-8",
    )


def test_output_root_inside_repository_is_rejected_before_writing(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    spec = tmp_path / "spec.yaml"
    write_spec(spec)
    output_root = repo / "generated"
    with pytest.raises(PreparationError, match="UNSAFE_OUTPUT_ROOT"):
        prepare(repo, spec, output_root)
    assert not output_root.exists()


def test_existing_publish_lock_blocks_concurrent_preparation(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    spec = tmp_path / "spec.yaml"
    write_spec(spec)
    output_root = tmp_path / "prepared"
    output_root.mkdir()
    lock = output_root / ".GOV-OUTPUT-ISOLATION-PREP-01.lock"
    lock.write_text("held\n", encoding="utf-8")
    with pytest.raises(PreparationError, match="OUTPUT_LOCKED"):
        prepare(repo, spec, output_root)
    assert lock.read_text(encoding="utf-8") == "held\n"
    assert not (output_root / "GOV-OUTPUT-ISOLATION-PREP-01").exists()
