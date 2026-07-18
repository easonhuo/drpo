from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path
from types import SimpleNamespace

import yaml

ROOT = Path(__file__).resolve().parents[1]
TOOLCHAIN = "9dc9920afdbf8c4c8b16eb99f562c1a7d85358ca"
HISTORICAL_BASE = "cd770f47b89f8971923945c19caec49720c0e139"
HISTORICAL_HEAD = "df6c7a2a506bffebe64f41a92d5611a94116c828"
HISTORICAL_RESULT = "b65882993eaf674390989bb9082be2b79f1f1e44"
for path in (ROOT / "src", ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from dev_integration_write_path import load_json, sha256, write_json  # noqa: E402
import run_workflow_replay as replay  # noqa: E402


def command(argv: list[str], cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        argv,
        cwd=cwd,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        timeout=1800,
    )


def test_replacement_c01_arm_a_calibration(tmp_path: Path) -> None:
    toolchain = tmp_path / "toolchain"
    added = command(["git", "worktree", "add", "--detach", str(toolchain), TOOLCHAIN], ROOT)
    assert added.returncode == 0, added.stderr
    packet = ROOT / "tests/fixtures/workflow_replay/candidate01_c1/c01_code_only_packet.yaml"
    source = yaml.safe_load(packet.read_text(encoding="utf-8"))["source"]
    contract = SimpleNamespace(
        base=SimpleNamespace(
            historical_task={
                "base_sha": HISTORICAL_BASE,
                "frozen_implementation_sha": HISTORICAL_HEAD,
            },
            benchmark={"toolchain_sha": TOOLCHAIN},
        ),
        sha256="c01-pr114-merge-arm-a-calibration",
    )
    root = tmp_path / "calibration"
    root.mkdir()
    workspace = replay._clone(ROOT, root, source, contract)
    before = replay._commit_workspace(root / "source.git", HISTORICAL_BASE)
    identity = SimpleNamespace(
        run_id="1" * 64,
        arm="A",
        case_id="C01-CODE-ONLY",
        pair_id="calibration",
        order_position=0,
    )
    journal = replay.Journal(root / "events.jsonl", identity, contract.sha256, before, workspace)
    completed = replay._explicit(
        workspace,
        packet,
        root / "preparations",
        root / "transactions",
        sys.executable,
        journal,
    )
    transaction = Path(completed.transaction_dir)
    ready = load_json(transaction / "READY_COMMIT.json", "ready")
    gate = load_json(transaction / "GATE_REPORT.json", "gate")
    normalization = load_json(transaction / "NORMALIZATION_REPORT.json", "normalization")
    paths = tuple(sorted(ready["changed_paths"]))
    modes = replay._modes(transaction / "integration-repo", completed.ready_commit_sha, paths)
    result_path = root / "result.json"
    write_json(
        result_path,
        {
            "case_id": "C01-CODE-ONLY",
            "base_sha": HISTORICAL_BASE,
            "tree_sha": ready["tree_sha"],
            "changed_paths": paths,
            "file_modes": modes,
        },
    )
    after = replay._workspace(transaction / "integration-repo")
    timing = journal.finish(
        "READY",
        after,
        len(journal.commands) + len({item.split(":", 1)[0] for item in journal.placements}),
    )
    gate_results = {
        item["label"]: "PASS" if item["passed"] else "FAIL" for item in gate["outcomes"]
    }
    summary = {
        "schema_version": 1,
        "claim_id": "GOV-DEV-WORKFLOW-OPTIMIZATION-BENCHMARK-01",
        "historical_claim_id": (
            "EXT-C-E8-ORACLE-OFFLINE-V2-PAPER-ALIGNED-LINEAR-C-EXTENSION-0.5B-01"
        ),
        "case_id": "C01-CODE-ONLY",
        "candidate_invoked": False,
        "historical_pr": 114,
        "historical_result_commit": HISTORICAL_RESULT,
        "toolchain_sha": TOOLCHAIN,
        "execution_sha": os.environ.get("GITHUB_SHA"),
        "workflow_run_id": os.environ.get("GITHUB_RUN_ID"),
        "packet_sha256": sha256(packet),
        "terminal_state": "READY",
        "ready_commit_sha": completed.ready_commit_sha,
        "tree_sha": ready["tree_sha"],
        "changed_paths": paths,
        "file_modes": modes,
        "result_sha256": sha256(result_path),
        "authority_result": (
            "PASS" if ready.get("authority_verify", {}).get("status") == "PASS" else "FAIL"
        ),
        "gate_results": gate_results,
        "workspace_before_sha256": before,
        "workspace_after_sha256": after,
        "timing": timing,
        "normalization_registration_mode": normalization.get("registration_mode"),
    }
    assert summary["authority_result"] == "PASS"
    assert all(value == "PASS" for value in gate_results.values())
    summary_path = root / "CALIBRATION_SUMMARY.json"
    write_json(summary_path, summary)
    evidence = ROOT / "c01-pr114-merge-calibration.tar.gz"
    with tarfile.open(evidence, "w:gz") as archive:
        archive.add(packet, arcname="c01_code_only_packet.yaml")
        archive.add(summary_path, arcname="CALIBRATION_SUMMARY.json")
        archive.add(result_path, arcname="result.json")
        archive.add(root / "events.jsonl", arcname="events.jsonl")
        for name in (
            "READY_COMMIT.json",
            "GATE_REPORT.json",
            "NORMALIZATION_REPORT.json",
            "TRANSACTION.json",
        ):
            archive.add(transaction / name, arcname=f"transaction/{name}")
        logs = transaction / "gate-logs"
        if logs.is_dir():
            archive.add(logs, arcname="transaction/gate-logs")
    assert evidence.is_file() and evidence.stat().st_size > 0
    shutil.rmtree(root / "source.git", ignore_errors=True)
