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


def patch_probe_horizon(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        auto.joint,
        "load_exp_horizon_run_spec",
        lambda _path: (
            {"trainer_argv_template": ["--eval_interval", "50000"]},
            "sha",
        ),
    )


def test_plan_passes_effective_probe_horizon_and_binds_selection(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    patch_probe_horizon(monkeypatch)
    monkeypatch.setattr(auto, "discover_machine", lambda **_kwargs: object())
    observed: dict[str, object] = {}

    def fake_select_e7_runtime(**kwargs: object) -> dict[str, object]:
        observed.update(kwargs)
        return {
            "mode": "auto",
            "selection": {"selected_workers": 12},
            "selection_digest": "digest-12",
        }

    def fake_plan(argv: list[str]) -> int:
        work = Path(argv[argv.index("--work-dir") + 1])
        workers = int(argv[argv.index("--max-workers") + 1])
        work.mkdir(parents=True, exist_ok=True)
        (work / "RUN_IDENTITY.json").write_text(
            json.dumps({"plan": {"max_workers": workers}}), encoding="utf-8"
        )
        return 0

    monkeypatch.setattr(auto, "select_e7_runtime", fake_select_e7_runtime)
    monkeypatch.setattr(auto, "_run_with_joint_hooks", fake_plan)
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
    assert observed["per_worker_cpu_safety_factor"] == 1.25
    payload = json.loads(capsys.readouterr().out)
    assert payload["requested_probe_steps"] == 20_000
    assert payload["effective_probe_steps"] == 100_000
    assert payload["probe_steps_adjusted"] is True
    identity = json.loads((work_dir / "RUN_IDENTITY.json").read_text())
    assert identity["runtime_resource_selection"]["selection_digest"] == "digest-12"


def test_run_consumes_selection_and_calls_revalidation_not_selection(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    patch_probe_horizon(monkeypatch)
    monkeypatch.setattr(auto, "discover_machine", lambda **_kwargs: object())
    work = tmp_path / "run"
    work.mkdir()
    (work / "RUNTIME_SELECTION.json").write_text(
        json.dumps(
            {
                "mode": "auto",
                "selection": {"selected_workers": 12},
                "selection_digest": "digest-12",
            }
        ),
        encoding="utf-8",
    )
    (work / "RUN_IDENTITY.json").write_text(
        json.dumps(
            {
                "plan": {"max_workers": 12},
                "runtime_resource_selection": {
                    "selected_workers": 12,
                    "selection_digest": "digest-12",
                },
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        auto,
        "select_e7_runtime",
        lambda **_kwargs: pytest.fail("run must not call automatic selection"),
    )
    observed: dict[str, object] = {}

    def fake_revalidate(**kwargs: object) -> dict[str, object]:
        observed.update(kwargs)
        return {
            "mode": "auto",
            "selection": {"selected_workers": 12},
            "selection_digest": "digest-12",
            "revalidation": {"decision": "ALLOW"},
        }

    monkeypatch.setattr(auto, "revalidate_e7_runtime", fake_revalidate)
    monkeypatch.setattr(auto, "_run_with_joint_hooks", lambda _argv: 0)

    assert (
        auto.main(
            [
                "run",
                "--contract",
                str(tmp_path / "contract.json"),
                "--run-spec",
                str(tmp_path / "run-spec.json"),
                "--grid",
                str(tmp_path / "grid.json"),
                "--work-dir",
                str(work),
            ]
        )
        == 0
    )
    assert observed["probe_steps"] == 100_000
    assert observed["revalidation_samples"] == 3


def test_run_rejects_missing_run_identity_before_revalidation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    patch_probe_horizon(monkeypatch)
    monkeypatch.setattr(auto, "discover_machine", lambda **_kwargs: object())
    work = tmp_path / "run"
    work.mkdir()
    (work / "RUNTIME_SELECTION.json").write_text(
        json.dumps(
            {
                "selection": {"selected_workers": 12},
                "selection_digest": "digest-12",
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        auto,
        "revalidate_e7_runtime",
        lambda **_kwargs: pytest.fail("identity gate must run before revalidation"),
    )
    with pytest.raises(RuntimeResourceError, match="RUN_IDENTITY"):
        auto.main(
            [
                "run",
                "--contract",
                str(tmp_path / "contract.json"),
                "--run-spec",
                str(tmp_path / "run-spec.json"),
                "--grid",
                str(tmp_path / "grid.json"),
                "--work-dir",
                str(work),
            ]
        )
