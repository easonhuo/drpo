from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / "src", ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from dev_integration_write_path import load_json, sha256, write_json  # noqa: E402
from drpo.workflow_replay.orchestrate import OrchestrationError  # noqa: E402
import run_workflow_replay as replay  # noqa: E402

TOOLCHAIN = "9dc9920afdbf8c4c8b16eb99f562c1a7d85358ca"


def packet(name: str):
    path = ROOT / "tests/fixtures/workflow_replay/candidate01_c1" / name
    return path, yaml.safe_load(path.read_text(encoding="utf-8"))


def contract(base: str, implementation: str | None):
    return SimpleNamespace(
        base=SimpleNamespace(
            historical_task={"base_sha": base, "frozen_implementation_sha": implementation},
            benchmark={"toolchain_sha": TOOLCHAIN},
        ),
        sha256="arm-a-calibration",
    )


def identity(case_id: str, digit: str):
    return SimpleNamespace(
        run_id=digit * 64,
        arm="A",
        case_id=case_id,
        pair_id="calibration",
        order_position=0,
    )


def test_arm_a_truth_calibration(tmp_path: Path) -> None:
    summaries = {}
    spec, payload = packet("c01_code_only_packet.yaml")
    frozen = contract(
        "2f677f4b00954ea71d0efa7def552a1ea3daa565",
        "4bdc1fa80bafd997b6358c51a83f5e57dd77ed16",
    )
    root = tmp_path / "c01"
    root.mkdir()
    workspace = replay._clone(ROOT, root, payload["source"], frozen)
    before = replay._commit_workspace(root / "source.git", frozen.base.historical_task["base_sha"])
    journal = replay.Journal(
        root / "events.jsonl", identity("C01-CODE-ONLY", "1"), frozen.sha256, before, workspace
    )
    completed = replay._explicit(
        workspace, spec, root / "preparations", root / "transactions", sys.executable, journal
    )
    transaction = Path(completed.transaction_dir)
    ready = load_json(transaction / "READY_COMMIT.json", "ready")
    gate = load_json(transaction / "GATE_REPORT.json", "gate")
    paths = tuple(sorted(ready["changed_paths"]))
    modes = replay._modes(transaction / "integration-repo", completed.ready_commit_sha, paths)
    result_path = root / "result.json"
    write_json(
        result_path,
        {
            "case_id": "C01-CODE-ONLY",
            "base_sha": frozen.base.historical_task["base_sha"],
            "tree_sha": ready["tree_sha"],
            "changed_paths": paths,
            "file_modes": modes,
        },
    )
    gate_results = {
        item["label"]: "PASS" if item["passed"] else "FAIL" for item in gate["outcomes"]
    }
    assert all(value == "PASS" for value in gate_results.values())
    assert ready["authority_verify"]["status"] == "PASS"
    summaries["C01"] = {
        "packet_sha256": sha256(spec),
        "terminal_state": "READY",
        "tree_sha": ready["tree_sha"],
        "changed_paths": paths,
        "file_modes": modes,
        "result_sha256": sha256(result_path),
        "authority_result": "PASS",
        "gate_results": gate_results,
    }

    spec, payload = packet("c06_stale_main_packet.yaml")
    frozen = contract(
        "d0b028bf438d32e550a281a79246cf828b36450e",
        "6795aa6f086c44e8073c5a995a1612f334a3a067",
    )
    root = tmp_path / "c06"
    root.mkdir()
    workspace = replay._clone(ROOT, root, payload["source"], frozen)
    before = replay._commit_workspace(root / "source.git", frozen.base.historical_task["base_sha"])
    journal = replay.Journal(
        root / "events.jsonl", identity("C06-STALE-MAIN", "2"), frozen.sha256, before, workspace
    )
    with pytest.raises(OrchestrationError):
        replay._explicit(
            workspace, spec, root / "preparations", root / "transactions", sys.executable, journal
        )
    failure = json.loads(journal.last_result.stdout)
    attempt = Path(failure["attempt_dir"])
    diagnostic = load_json(attempt / "DIAGNOSTIC.json", "diagnostic")
    assert failure["error_code"] == "SOURCE_DRIFT"
    assert diagnostic["phase"] == "source_lock"
    assert not (attempt / "integration-repo").exists()
    summaries["C06"] = {
        "packet_sha256": sha256(spec),
        "terminal_state": "BLOCKED",
        "safety_boundary": "source_lock",
        "diagnostic_codes": ["SOURCE_DRIFT"],
        "recovery_class": "refresh_main_and_regenerate_packet",
        "authority_result": "NOT_RUN",
        "workspace_unchanged": True,
        "result_artifact": None,
    }
    pytest.fail("REPLAYAB_ARM_A_CALIBRATION=" + json.dumps(summaries, sort_keys=True))
