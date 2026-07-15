from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
RUN_ID = "E8_REPRO_RNG_AUDIT_20260716_01"
SPEC = ROOT / "runspecs" / "ready" / f"{RUN_ID}.yaml"
WRAPPER = ROOT / "scripts" / "run_countdown_e8_oracle_offline_v2_repro_rng_audit_one_click.sh"


def test_e8_repro_rng_audit_runspec_is_frozen_and_valid() -> None:
    spec = yaml.safe_load(SPEC.read_text(encoding="utf-8"))

    assert spec["run_id"] == RUN_ID
    assert spec["lane"] == "e8"
    assert spec["experiment_id"] == (
        "EXT-C-E8-ORACLE-OFFLINE-V2-REPRO-RNG-AUDIT-0.5B-01"
    )
    assert spec["execution_class"] == "pilot"
    assert spec["registration"] == {
        "mode": "deferred",
        "closure_required": True,
    }
    assert spec["repo_commit"] == "8982928f415e0fd79515e1e77b0515ae5792856c"

    command = spec["entrypoint"]["command"]
    assert "/root/models/Qwen2.5-0.5B-Instruct" in command
    assert "/root/experiment_output/e8_oracle_bank_v2/data/offline_bank_v2.jsonl" in command
    assert "/root/experiment_output/e8_oracle_bank_v2/data/val.jsonl" in command
    assert "--gpus 0,1,2,3,4,5,6,7" in command
    assert "test" not in command.lower()

    assert subprocess.run(
        ["bash", "-n", str(WRAPPER)],
        cwd=ROOT,
        check=False,
    ).returncode == 0

    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "agent" / "validate_runspec.py"),
            "--repo-root",
            str(ROOT),
            "--lane",
            "e8",
            str(SPEC),
            "--json",
        ],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr or completed.stdout
    payload = json.loads(completed.stdout)
    assert payload["status"] == "PASS"
    assert payload["registration_mode"] == "deferred"
    assert payload["registration_closure_required"] is True
