from __future__ import annotations

import json
from pathlib import Path

from drpo_reference.categorical.du1_protocol import FORMAL_METHODS
from drpo_reference.categorical.du1_public import run_du1


def test_du1_smoke_writes_complete_nonformal_matrix(
    tmp_path: Path,
) -> None:
    manifest = run_du1(
        output_root=tmp_path,
        smoke=True,
        device="cpu",
        workers=1,
    )
    assert manifest["protocol_revision"] == 4
    assert manifest["formal_evidence_allowed"] is False
    assert manifest["terminology"] == ("same-distribution held-out-context generalization")
    terminal = json.loads((tmp_path / "terminal_audit.json").read_text(encoding="utf-8"))
    assert terminal["actual_runs"] == len(FORMAL_METHODS)
    assert terminal["all_registered_runs_present"] is True
    assert terminal["formal_evidence_allowed"] is False
    assert (tmp_path / "aggregate_summary.json").exists()
    assert (tmp_path / "mechanism_summary.json").exists()
    assert (tmp_path / "taper_summary.json").exists()
    checkpoint = tmp_path / "checkpoints" / "seed_0" / "CHECKPOINT_COMPLETE.json"
    assert checkpoint.exists()
    complete = json.loads(checkpoint.read_text(encoding="utf-8"))
    assert complete["methods_completed"] == list(FORMAL_METHODS)
