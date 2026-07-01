from __future__ import annotations
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
E = ROOT / "docs/governance_stage4b_acceptance"
BASE = "cf775893b9885ba893278437556abb4d1d5dd1a8"


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def test_stage4b_acceptance_evidence_is_pass():
    r = json.loads((E / "ACCEPTANCE_REPORT.json").read_text())
    f = json.loads((E / "FAULT_INJECTION_REPORT.json").read_text())
    assert (
        r["status"] == "PASS"
        and r["evaluated_base_commit"] == BASE
        and r["stage_4b_state"] == "accepted"
        and r["stage_4c_state"] == "ready_for_authorization"
        and r["authority_cutover_allowed"] is False
    )
    assert f["status"] == "PASS" and f["passed"] == f["total"] and f["total"] >= 15


def test_stage4b_after_image_and_checksums_match():
    a = json.loads((E / "AFTER_IMAGE.json").read_text())
    assert a["base_commit"] == BASE and a["file_count"] == len(a["files"])
    for x in a["files"]:
        assert sha(ROOT / x["path"]) == x["sha256"]
    tree = hashlib.sha256(
        json.dumps(a["files"], sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    assert tree == a["tree_hash"]
    for line in (E / "CHECKSUMS.sha256").read_text().splitlines():
        d, n = line.split("  ", 1)
        assert sha(E / n) == d
