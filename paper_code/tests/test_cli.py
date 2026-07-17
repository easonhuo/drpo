from __future__ import annotations

from pathlib import Path

import pytest

from drpo_reference import cli


def test_cli_dispatches_cu1_stage(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    observed: dict[str, object] = {}

    def fake_run(**kwargs: object) -> dict[str, object]:
        observed.update(kwargs)
        return {}

    monkeypatch.setattr(cli, "run_cu1_stage", fake_run)
    assert (
        cli.main(
            [
                "cu1",
                "--stage",
                "source",
                "--output",
                str(tmp_path),
                "--seeds",
                "10,11",
                "--device",
                "cpu",
            ]
        )
        == 0
    )
    assert observed == {
        "stage": "source",
        "output_root": tmp_path,
        "seeds": (10, 11),
        "smoke": False,
        "device": "cpu",
    }


def test_cli_rejects_seed_override_for_all(tmp_path: Path) -> None:
    with pytest.raises(SystemExit, match="not valid"):
        cli.main(
            [
                "cu1",
                "--stage",
                "all",
                "--output",
                str(tmp_path),
                "--seeds",
                "10",
            ]
        )
