from __future__ import annotations

import json
from pathlib import Path

from drpo_reference import cli
from drpo_reference.continuous import cu1_suite


def test_source_smoke_is_audited_but_never_formal(tmp_path: Path) -> None:
    manifest = cu1_suite.run_cu1_stage(
        stage="source",
        output_root=tmp_path,
        smoke=True,
        device="cpu",
    )
    assert manifest["paper_stage"] == "source"
    assert manifest["terminology"] == ("same-distribution held-out-context generalization")
    assert manifest["formal_evidence_allowed"] is False
    audit = json.loads((tmp_path / "terminal_audit" / "source.json").read_text(encoding="utf-8"))
    assert audit["matrix"]["passed"] is True
    assert audit["formal_evidence_allowed"] is False
    assert (tmp_path / "aggregate" / "source.csv").exists()
    assert (tmp_path / "source" / "seed_10.json").exists()
    assert (
        tmp_path / "preparation" / "positive_checkpoints" / "seed_10_adam3_initialization.pt"
    ).exists()


def test_cu1_source_cpu_cli_liveness(tmp_path: Path) -> None:
    output = tmp_path / "cu1_source"
    assert (
        cli.main(
            [
                "cu1",
                "--stage",
                "source",
                "--output",
                str(output),
                "--device",
                "cpu",
                "--smoke",
            ]
        )
        == 0
    )

    audit_path = output / "terminal_audit" / "source.json"
    assert audit_path.is_file()
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    assert audit["matrix"]["passed"] is True
    assert audit["formal_evidence_allowed"] is False
    assert (output / "aggregate" / "source.csv").is_file()
    assert (output / "source" / "seed_10.json").is_file()
    assert (
        output / "preparation" / "positive_checkpoints" / "seed_10_adam3_initialization.pt"
    ).is_file()
