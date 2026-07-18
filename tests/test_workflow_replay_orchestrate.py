from __future__ import annotations

import json
import os
import subprocess
import sys
import tarfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
TOOLCHAIN = "9dc9920afdbf8c4c8b16eb99f562c1a7d85358ca"
CASES = (
    ("C01", "c01_code_only_contract.yaml", "c01_code_only_packet.yaml"),
    ("C06", "c06_stale_main_contract.yaml", "c06_stale_main_packet.yaml"),
)


def run(command: list[str], cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        cwd=cwd,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        timeout=1800,
    )


@pytest.mark.timeout(1800)
def test_candidate01_c01_c06_real_pair_liveness(tmp_path: Path) -> None:
    toolchain = tmp_path / "toolchain"
    added = run(["git", "worktree", "add", "--detach", str(toolchain), TOOLCHAIN], ROOT)
    assert added.returncode == 0, added.stderr
    fixture_root = ROOT / "tests/fixtures/workflow_replay/candidate01_c1"
    summaries = {}
    success = True
    for case_id, contract_name, packet_name in CASES:
        output = tmp_path / case_id.lower()
        command = [
            sys.executable,
            str(toolchain / "scripts/run_workflow_replay.py"),
            "real-pair",
            "--contract",
            str(fixture_root / contract_name),
            "--case-packet",
            str(fixture_root / packet_name),
            "--source-repo",
            str(ROOT),
            "--output-root",
            str(output),
            "--backend-id",
            "github-actions-local-git-v1",
            "--json",
        ]
        completed = run(command, toolchain)
        try:
            payload = json.loads(completed.stdout)
        except json.JSONDecodeError:
            payload = {
                "status": "FAIL",
                "state": "INVALID_OUTPUT",
                "stdout": completed.stdout[-4000:],
                "stderr": completed.stderr[-4000:],
            }
        summaries[case_id] = {
            "command": command,
            "returncode": completed.returncode,
            "payload": payload,
            "stderr": completed.stderr[-4000:],
        }
        valid = (
            completed.returncode == 0
            and payload.get("status") == "PASS"
            and payload.get("state") == "C1_LIVENESS_READY"
            and payload.get("run_count") == 4
            and payload.get("pair_count") == 2
            and all(item.get("equivalent") for item in payload.get("pairs", []))
        )
        success &= valid
    summary_path = tmp_path / "LIVENESS_SUMMARY.json"
    summary_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "claim_id": "GOV-DEV-WORKFLOW-OPTIMIZATION-BENCHMARK-01",
                "implementation_sha": TOOLCHAIN,
                "workflow_run_id": os.environ.get("GITHUB_RUN_ID"),
                "workflow_sha": os.environ.get("GITHUB_SHA"),
                "success": success,
                "cases": summaries,
            },
            sort_keys=True,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    archive = ROOT / ".replayab-liveness.tar.gz"
    with tarfile.open(archive, "w:gz") as handle:
        handle.add(summary_path, arcname="LIVENESS_SUMMARY.json")
        for case_id, _, _ in CASES:
            output = tmp_path / case_id.lower()
            if output.exists():
                handle.add(output, arcname=case_id.lower())
    assert archive.is_file() and archive.stat().st_size > 0
    assert success, json.dumps(summaries, sort_keys=True)
