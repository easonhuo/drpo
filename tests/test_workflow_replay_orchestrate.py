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


def upload_evidence(archive: Path, workspace: Path) -> dict:
    runtime = os.environ.get("ACTIONS_RUNTIME_TOKEN")
    results_url = os.environ.get("ACTIONS_RESULTS_URL")
    if not runtime or not results_url:
        raise AssertionError("GitHub Actions artifact runtime is unavailable")
    uploader = workspace / "uploader"
    installed = run(
        [
            "npm",
            "install",
            "--silent",
            "--no-audit",
            "--no-fund",
            "--prefix",
            str(uploader),
            "@actions/artifact@6.2.1",
        ],
        workspace,
    )
    if installed.returncode:
        raise AssertionError(installed.stderr[-2000:])
    script = uploader / "upload.mjs"
    script.write_text(
        "import {DefaultArtifactClient} from '@actions/artifact';\n"
        "const client = new DefaultArtifactClient();\n"
        "const result = await client.uploadArtifact(\n"
        "  process.env.REPLAY_ARTIFACT_NAME,\n"
        "  [process.env.REPLAY_ARTIFACT_FILE],\n"
        "  process.env.REPLAY_ARTIFACT_ROOT,\n"
        "  {retentionDays: 7}\n"
        ");\n"
        "console.log(JSON.stringify(result));\n",
        encoding="utf-8",
    )
    environment = os.environ.copy()
    environment["REPLAY_ARTIFACT_NAME"] = (
        "candidate01-c1-liveness-" + os.environ.get("GITHUB_RUN_ID", "local")
    )
    environment["REPLAY_ARTIFACT_FILE"] = str(archive)
    environment["REPLAY_ARTIFACT_ROOT"] = str(workspace)
    uploaded = subprocess.run(
        ["node", str(script)],
        cwd=uploader,
        env=environment,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        timeout=300,
    )
    if uploaded.returncode:
        raise AssertionError(uploaded.stderr[-2000:])
    return json.loads(uploaded.stdout.strip().splitlines()[-1])


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
    archive = tmp_path / "candidate01-c1-liveness.tar.gz"
    with tarfile.open(archive, "w:gz") as handle:
        handle.add(summary_path, arcname="LIVENESS_SUMMARY.json")
        for case_id, _, _ in CASES:
            output = tmp_path / case_id.lower()
            if output.exists():
                handle.add(output, arcname=case_id.lower())
    upload = upload_evidence(archive, tmp_path)
    assert upload.get("id"), upload
    assert success, json.dumps(summaries, sort_keys=True)
