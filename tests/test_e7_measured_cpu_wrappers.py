from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from drpo import e7_ppo_w0_runtime_autotune as shared
from drpo import e7_squared_exp_night_runtime_autotune as night_adapter
from drpo import e7_w0_highc_runtime_autotune as highc_adapter
from drpo.runtime_resource_autotune import RuntimeResourceError


def load_script(name: str, path: str):
    spec = importlib.util.spec_from_file_location(name, Path(path))
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ppo_script = load_script(
    "e7_ppo_w0_auto_v2", "scripts/run_e7_ppo_w0_grid_pilot_auto.py"
)
highc_script = load_script(
    "e7_highc_auto_v2", "scripts/run_e7_w0_highc_actor_auto.py"
)
night_script = load_script(
    "e7_night_auto_v2", "scripts/run_e7_squared_exp_night_auto.py"
)


def required_args() -> list[str]:
    return [
        "plan",
        "--contract",
        "contract.json",
        "--run-spec",
        "run-spec.json",
        "--grid",
        "grid.json",
        "--work-dir",
        "work",
    ]


@pytest.mark.parametrize("module", [ppo_script, highc_script, night_script])
def test_auto_script_exposes_measured_cpu_defaults(module) -> None:
    args = module.build_parser().parse_args(required_args())
    assert args.cpu_fraction == 0.85
    assert args.per_worker_cpu_safety_factor == 1.25
    assert args.minimum_cpu_cores_per_worker == 1.0
    assert args.revalidation_samples == 3
    assert args.revalidation_sample_seconds == 1.0


@pytest.mark.parametrize("module", [ppo_script, highc_script, night_script])
def test_run_identity_is_required_before_revalidation(
    module, tmp_path: Path
) -> None:
    with pytest.raises(RuntimeResourceError, match="RUN_IDENTITY"):
        module._validate_existing_run_identity(tmp_path, 4, "digest")  # noqa: SLF001


def test_highc_adapter_binds_and_restores_implementation_identity() -> None:
    original = shared._selector_implementation_identity  # noqa: SLF001
    with highc_adapter._installed_adapter():  # noqa: SLF001
        assert shared._selector_implementation_identity is (  # noqa: SLF001
            highc_adapter._selector_implementation_identity  # noqa: SLF001
        )
        values = shared._selector_implementation_identity(Path.cwd())  # noqa: SLF001
        assert "e7_w0_highc_runtime_autotune.py" in values
    assert shared._selector_implementation_identity is original  # noqa: SLF001


def test_night_adapter_binds_and_restores_implementation_identity() -> None:
    original = shared._selector_implementation_identity  # noqa: SLF001
    with night_adapter._installed_adapter():  # noqa: SLF001
        assert shared._selector_implementation_identity is (  # noqa: SLF001
            night_adapter._selector_implementation_identity  # noqa: SLF001
        )
        values = shared._selector_implementation_identity(Path.cwd())  # noqa: SLF001
        assert "e7_squared_exp_night_runtime_autotune.py" in values
    assert shared._selector_implementation_identity is original  # noqa: SLF001
