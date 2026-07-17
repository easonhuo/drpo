from __future__ import annotations

from pathlib import Path

import pytest

from drpo_reference import cli


def test_cli_dispatches_du1(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    observed: dict[str, object] = {}

    def fake_run(**kwargs: object) -> dict[str, object]:
        observed.update(kwargs)
        return {}

    monkeypatch.setattr(cli, "run_du1", fake_run)
    assert (
        cli.main(
            [
                "du1",
                "--output",
                str(tmp_path),
                "--seeds",
                "200,201",
                "--device",
                "cpu",
                "--workers",
                "2",
            ]
        )
        == 0
    )
    assert observed == {
        "output_root": tmp_path,
        "seeds": (200, 201),
        "smoke": False,
        "device": "cpu",
        "workers": 2,
    }
