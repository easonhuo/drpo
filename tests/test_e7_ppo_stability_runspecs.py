from __future__ import annotations

from pathlib import Path

import yaml


def _root() -> Path:
    return Path(__file__).resolve().parents[1]


def _load(path: Path) -> dict:
    value = yaml.safe_load(path.read_text())
    assert isinstance(value, dict)
    return value


def test_registry_registers_smoke_ready_and_full_pilot_blocked() -> None:
    root = _root()
    registry = _load(root / "experiments" / "registry.yaml")
    experiments = {
        item["id"]: item
        for item in registry["experiments"]
        if isinstance(item, dict) and "id" in item
    }
    experiment = experiments["EXT-H-E7-PPO-STABILITY-01"]
    assert experiment["status"] == "not_run"
    assert experiment["scientific_status"] == "pilot"
    assert experiment["smoke_gate"]["status"] == "pending"
    assert experiment["full_pilot"]["state"] == "blocked"
    assert experiment["full_pilot"]["held_out_seeds_reserved_untouched"] == [
        204,
        205,
        206,
        207,
    ]
    assert experiment["execution"]["smoke_run_id"] == (
        "E7_PPO_STABILITY_SMOKE_20260712_01"
    )
    assert experiment["execution"]["pilot_promotion_requires_new_reviewed_commit"]


def test_smoke_runspec_is_ready_but_full_pilot_remains_template() -> None:
    root = _root()
    smoke_path = (
        root / "runspecs" / "ready" / "E7_PPO_STABILITY_SMOKE_20260712_01.yaml"
    )
    pilot_path = (
        root
        / "runspecs"
        / "templates"
        / "E7_PPO_STABILITY_PILOT_20260712_01.yaml"
    )
    assert smoke_path.is_file()
    assert pilot_path.is_file()
    assert not (
        root / "runspecs" / "ready" / "E7_PPO_STABILITY_PILOT_20260712_01.yaml"
    ).exists()

    smoke = _load(smoke_path)
    pilot = _load(pilot_path)
    assert smoke["repo_commit"] == "76874d6cc40cca83dcf9917fbf779761c222a1be"
    assert pilot["repo_commit"] == "76874d6cc40cca83dcf9917fbf779761c222a1be"
    assert smoke["entrypoint"]["command"] == (
        "bash scripts/run_e7_ppo_stability_smoke_one_click.sh"
    )
    assert pilot["entrypoint"]["command"] == (
        "bash scripts/run_e7_ppo_stability_pilot_auto_one_click.sh"
    )
    assert smoke["resources"]["max_parallel_processes"] == 1
    assert pilot["resources"]["max_parallel_processes"] == (
        "auto_from_RUNTIME_SELECTION"
    )
