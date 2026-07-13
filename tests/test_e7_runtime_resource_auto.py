from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from drpo.runtime_resource_autotune import RuntimeResourceError


SCRIPT = Path("scripts/run_e7_canonical_exp_horizon_joint_auto.py")
SPEC = importlib.util.spec_from_file_location("e7_runtime_resource_auto", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
auto = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(auto)


def test_effective_probe_steps_cover_two_evaluation_windows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        auto.joint,
        "load_exp_horizon_run_spec",
        lambda _path: (
            {
                "trainer_argv_template": [
                    "--steps",
                    "1000000",
                    "--eval_interval",
                    "50000",
                ]
            },
            "sha",
        ),
    )
    assert auto._resolve_effective_probe_steps("run-spec.json", 20_000) == 100_000
    assert auto._resolve_effective_probe_steps("run-spec.json", 150_000) == 150_000


def test_positive_int_cli_option_supports_equals_and_rejects_bad_values() -> None:
    assert (
        auto._positive_int_cli_option(
            ["--eval-interval=400", "--eval_interval", "500"],
            ("--eval_interval", "--eval-interval"),
        )
        == 500
    )
    with pytest.raises(RuntimeResourceError, match="literal positive integer"):
        auto._positive_int_cli_option(
            ["--eval_interval", "{eval_interval}"],
            ("--eval_interval", "--eval-interval"),
        )
    with pytest.raises(RuntimeResourceError, match="must be positive"):
        auto._positive_int_cli_option(
            ["--eval_interval", "0"],
            ("--eval_interval", "--eval-interval"),
        )


def test_main_passes_effective_probe_horizon_to_selector(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(
        auto.joint,
        "load_exp_horizon_run_spec",
        lambda _path: (
            {
                "trainer_argv_template": [
                    "--eval_interval",
                    "50000",
                ]
            },
            "sha",
        ),
    )
    monkeypatch.setattr(auto, "discover_machine", lambda **_kwargs: object())
    observed: dict[str, object] = {}

    def fake_select_e7_runtime(**kwargs: object) -> dict[str, object]:
        observed.update(kwargs)
        return {
            "mode": "auto",
            "selection": {"selected_workers": 12},
        }

    monkeypatch.setattr(auto, "select_e7_runtime", fake_select_e7_runtime)
    monkeypatch.setattr(auto, "_run_with_joint_hooks", lambda _argv: 0)
    work_dir = tmp_path / "acceptance"

    result = auto.main(
        [
            "plan",
            "--contract",
            str(tmp_path / "contract.json"),
            "--run-spec",
            str(tmp_path / "run-spec.json"),
            "--grid",
            str(tmp_path / "grid.json"),
            "--work-dir",
            str(work_dir),
        ]
    )

    assert result == 0
    assert observed["probe_steps"] == 100_000
    payload = json.loads(capsys.readouterr().out)
    assert payload["requested_probe_steps"] == 20_000
    assert payload["effective_probe_steps"] == 100_000
    assert payload["probe_steps_adjusted"] is True
