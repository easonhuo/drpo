from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from drpo import e7_ppo_w0_runtime_autotune as shared
from drpo import e7_squared_exp_night as night
from drpo import e7_squared_exp_night_bootstrap as bootstrap
from drpo import e7_squared_exp_night_runtime_autotune as night_adapter
from drpo import e7_w0_highc_runtime_autotune as highc_adapter
from drpo.runtime_resource_autotune import RuntimeResourceError, canonical_json_sha256


P3_GRID = Path("configs/e7_bench_joint_gae_p3_left_saturation.json")
HISTORICAL_GRID = Path("configs/e7_squared_exp_night_v1.json")
P3_RUNSPEC = Path(
    "runspecs/ready/E7_BENCH_JOINT_GAE_P3_LEFT_SATURATION_FULL_20260722_03.yaml"
)


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
    "e7_night_auto_v3", "scripts/run_e7_squared_exp_night_auto.py"
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


@pytest.mark.parametrize("module", [ppo_script, night_script])
def test_plan_materializes_identity_from_execution_plan(
    module, tmp_path: Path
) -> None:
    work = tmp_path / "work"
    work.mkdir()
    plan = {
        "created_utc": "2026-07-15T00:00:00+00:00",
        "max_workers": 4,
        "branch_count": 186,
    }
    (work / "EXECUTION_PLAN.json").write_text(
        json.dumps(plan), encoding="utf-8"
    )
    module._bind_selection_to_run_identity(  # noqa: SLF001
        work,
        selected_workers=4,
        selection_digest="selection-digest",
    )
    identity = json.loads((work / "RUN_IDENTITY.json").read_text(encoding="utf-8"))
    stable_plan = {key: value for key, value in plan.items() if key != "created_utc"}
    assert identity["run_identity_sha256"] == canonical_json_sha256(stable_plan)
    assert identity["plan"] == plan
    assert identity["runtime_resource_selection"] == {
        "selection_digest": "selection-digest",
        "selected_workers": 4,
        "path": str(work / "RUNTIME_SELECTION.json"),
        "scientific_matrix_changed": False,
    }
    module._validate_existing_run_identity(  # noqa: SLF001
        work, 4, "selection-digest"
    )


@pytest.mark.parametrize("module", [ppo_script, night_script])
def test_plan_identity_requires_execution_plan(module, tmp_path: Path) -> None:
    with pytest.raises(RuntimeResourceError, match="EXECUTION_PLAN"):
        module._bind_selection_to_run_identity(  # noqa: SLF001
            tmp_path,
            selected_workers=4,
            selection_digest="selection-digest",
        )


def test_night_parser_reuses_existing_command_with_optional_gae_pair() -> None:
    args = night_script.build_parser().parse_args(
        [*required_args(), "--matched-gae-pair"]
    )
    assert args.command == "plan"
    assert args.matched_gae_pair is True


def test_highc_adapter_binds_and_restores_implementation_identity() -> None:
    original = shared._selector_implementation_identity  # noqa: SLF001
    with highc_adapter._installed_adapter():  # noqa: SLF001
        assert shared._selector_implementation_identity is (  # noqa: SLF001
            highc_adapter._selector_implementation_identity  # noqa: SLF001
        )
        values = shared._selector_implementation_identity(Path.cwd())  # noqa: SLF001
        assert "e7_w0_highc_runtime_autotune.py" in values
    assert shared._selector_implementation_identity is original  # noqa: SLF001


def test_night_adapter_binds_and_restores_v3_policy() -> None:
    original_identity = shared._selector_implementation_identity  # noqa: SLF001
    original_adapter = shared.ADAPTER_ID
    original_candidates = shared.candidate_workers
    original_benchmark = shared.benchmark_concurrency
    original_policy = shared.SELECTOR_POLICY_VERSION
    with night_adapter._installed_adapter():  # noqa: SLF001
        assert shared._selector_implementation_identity is (  # noqa: SLF001
            night_adapter._selector_implementation_identity  # noqa: SLF001
        )
        assert shared.candidate_workers is (  # noqa: SLF001
            night_adapter._low_first_candidate_workers  # noqa: SLF001
        )
        assert shared.benchmark_concurrency is (  # noqa: SLF001
            night_adapter._bounded_benchmark_concurrency  # noqa: SLF001
        )
        assert shared.ADAPTER_ID == "e7_squared_exp_night_cpu_v3"
        assert shared.SELECTOR_POLICY_VERSION == 3
    assert shared._selector_implementation_identity is original_identity  # noqa: SLF001
    assert shared.ADAPTER_ID == original_adapter
    assert shared.candidate_workers is original_candidates
    assert shared.benchmark_concurrency is original_benchmark
    assert shared.SELECTOR_POLICY_VERSION == original_policy


def test_night_candidate_grid_starts_low_without_configured_cap() -> None:
    assert night_adapter._low_first_candidate_workers(130, 60) == [  # noqa: SLF001
        1,
        2,
        4,
        7,
        13,
        23,
        41,
        60,
        72,
        126,
        130,
    ]


def test_night_probe_policy_caps_steps_and_timeout() -> None:
    assert night_adapter._bounded_probe_policy(100_000, 2_500.0) == {  # noqa: SLF001
        "requested_probe_steps": 100_000,
        "effective_probe_steps": 5_000,
        "requested_probe_seconds": 2_500.0,
        "effective_probe_seconds": 120.0,
    }


def test_requested_probe_policy_is_non_identity_evidence(tmp_path: Path) -> None:
    document = {"selection_digest": "stable", "selection": {"selected_workers": 41}}
    policy = night_adapter._bounded_probe_policy(100_000, 2_500.0)  # noqa: SLF001
    result = night_adapter._attach_requested_probe_policy(  # noqa: SLF001
        document, work_dir=tmp_path, policy=policy
    )
    assert result["selection_digest"] == "stable"
    assert result["requested_probe_policy"]["identity_affecting"] is False
    assert result["requested_probe_policy"]["effective_probe_steps"] == 5_000
    assert result["requested_probe_policy"]["effective_probe_seconds"] == 120.0


class _FakeBinding:
    def as_dict(self) -> dict[str, object]:
        return {"affinity_cpu_ids": [0, 1], "quota_domains": []}


def _candidate_kwargs(tmp_path: Path) -> dict[str, object]:
    return {
        "probe_root": tmp_path,
        "concurrency": 17,
        "probe_steps": 100_000,
        "probe_seed": 99,
        "timeout_seconds": 2_500.0,
        "cpu_fraction": 0.85,
        "cpu_safety_factor": 1.25,
        "usable_memory_bytes": 1_000_000,
        "binding": _FakeBinding(),
    }


def test_candidate_probe_is_bounded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    observed: dict[str, object] = {}

    def fake_benchmark(**kwargs: object) -> dict[str, object]:
        observed.update(kwargs)
        return {"valid": True, "aggregate_peak_rss_bytes": 1_024}

    monkeypatch.setattr(
        night_adapter, "_ORIGINAL_BENCHMARK_CONCURRENCY", fake_benchmark
    )
    token = night_adapter._ACTIVE_PLAN_IDENTITY.set("plan")  # noqa: SLF001
    try:
        result = night_adapter._bounded_benchmark_concurrency(  # noqa: SLF001
            **_candidate_kwargs(tmp_path)
        )
    finally:
        night_adapter._ACTIVE_PLAN_IDENTITY.reset(token)  # noqa: SLF001
    assert observed["probe_steps"] == 5_000
    assert observed["timeout_seconds"] == 120.0
    assert result["checkpoint_reused"] is False


def test_exact_valid_candidate_checkpoint_is_reused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    kwargs = _candidate_kwargs(tmp_path)
    token = night_adapter._ACTIVE_PLAN_IDENTITY.set("plan")  # noqa: SLF001
    try:
        identity = night_adapter._candidate_checkpoint_identity(  # noqa: SLF001
            kwargs, effective_steps=5_000, effective_seconds=120.0
        )
        path = night_adapter._candidate_summary_path(kwargs)  # noqa: SLF001
        path.parent.mkdir(parents=True)
        path.write_text(
            json.dumps(
                {
                    "valid": True,
                    "aggregate_peak_rss_bytes": 1_024,
                    "aggregate_updates_per_second": 1_700.0,
                    "v3_checkpoint_identity": identity,
                }
            ),
            encoding="utf-8",
        )

        def should_not_run(**_kwargs: object) -> dict[str, object]:
            raise AssertionError("valid checkpoint should be reused")

        monkeypatch.setattr(
            night_adapter, "_ORIGINAL_BENCHMARK_CONCURRENCY", should_not_run
        )
        result = night_adapter._bounded_benchmark_concurrency(**kwargs)  # noqa: SLF001
    finally:
        night_adapter._ACTIVE_PLAN_IDENTITY.reset(token)  # noqa: SLF001
    assert result["checkpoint_reused"] is True
    assert result["aggregate_updates_per_second"] == 1_700.0


def test_valid_lower_candidates_survive_higher_failure() -> None:
    selected, _ = shared.select_from_throughput(
        [
            {"concurrency": 1, "valid": True, "aggregate_updates_per_second": 100.0},
            {"concurrency": 23, "valid": True, "aggregate_updates_per_second": 900.0},
            {"concurrency": 41, "valid": True, "aggregate_updates_per_second": 1_700.0},
            {"concurrency": 60, "valid": False, "aggregate_updates_per_second": 0.0},
        ],
        retention_fraction=0.97,
    )
    assert selected == 41


def _p3_run_spec() -> dict[str, object]:
    digest = "0" * 64
    return {
        "datasets": [
            {
                "id": dataset,
                "path": f"/tmp/{dataset}.hdf5",
                "sha256": digest,
            }
            for dataset in night.TUNING_DATASETS
        ],
        "seeds": list(night.TUNING_SEEDS),
    }


def test_p3_stable_main_builds_exact_180_branch_matrix() -> None:
    try:
        night.configure_execution(P3_GRID)
        grid, digest = night.load_grid(P3_GRID)
        branches = night.build_branches(
            SimpleNamespace(expected_canonical_alpha=0.11),
            _p3_run_spec(),
            grid,
        )
        assert len(digest) == 64
        assert len(branches) == night.P3_EXPECTED_BRANCHES == 180
        assert len({branch.branch_id for branch in branches}) == 180
        assert {branch.seed for branch in branches} == {200, 201}
        assert not ({branch.seed for branch in branches} & set(night.HELD_OUT_SEEDS))
        assert sum(
            branch.template_values["weight_method"] == "positive_only"
            for branch in branches
        ) == 18
        observed = {
            float(branch.template_values["remoteness_scale"])
            for branch in branches
            if branch.template_values["weight_method"] == "thresholded_exponential"
        }
        assert sorted(observed) == pytest.approx(
            sorted(night.P3_REMOTENESS_SCALES)
        )
    finally:
        night.configure_execution(HISTORICAL_GRID)


def test_p3_profile_enters_v3_runtime_adapter() -> None:
    previous_adapter = shared.ADAPTER_ID
    try:
        night.configure_execution(P3_GRID)
        profile = night.active_runtime_profile()
        assert profile["adapter_id"] == (
            "e7_joint_gae_thresholded_p3_left_saturation_cpu_v2"
        )
        assert night_adapter._v3_adapter_id(profile) == (  # noqa: SLF001
            "e7_joint_gae_thresholded_p3_left_saturation_cpu_v3"
        )
        assert math.isclose(float(profile["exp_coefficient"]), 1000.0)
        with night_adapter._installed_adapter():  # noqa: SLF001
            assert shared.ADAPTER_ID == (
                "e7_joint_gae_thresholded_p3_left_saturation_cpu_v3"
            )
        assert shared.ADAPTER_ID == previous_adapter
    finally:
        night.configure_execution(HISTORICAL_GRID)


def test_p3_profile_passes_bootstrap_profile_gate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        bootstrap,
        "CanonicalContract",
        SimpleNamespace(
            load=lambda _path: SimpleNamespace(expected_canonical_alpha=0.11)
        ),
    )
    branch_config = tmp_path / "branch_config.json"
    branch_config.write_text(
        json.dumps(
            {
                "experiment_id": night.GAE_EXPERIMENT_ID,
                "profile_id": night.P3_PROFILE_ID,
                "branch_kind": "deliberately_invalid_after_profile_gate",
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="bootstrap requires a public injected branch"):
        bootstrap.main(
            [
                "--contract",
                str(tmp_path / "contract.json"),
                "--branch-config",
                str(branch_config),
                "--branch-manifest",
                str(tmp_path / "manifest.json"),
            ]
        )


def test_p3_ready_runspec_and_launcher_are_stable_main_only() -> None:
    spec = yaml.safe_load(P3_RUNSPEC.read_text(encoding="utf-8"))
    assert spec["run_id"] == (
        "E7_BENCH_JOINT_GAE_P3_LEFT_SATURATION_FULL_20260722_03"
    )
    assert spec["repo_commit"] == "e05aaeeeaa86563df906226624e93b8116e432ca"
    assert spec["registration"] == {
        "mode": "deferred",
        "closure_required": True,
    }
    criteria = " ".join(spec["success_criteria"])
    assert "180 branches" in criteria
    assert "not a launch gate" in criteria
    launcher = Path("scripts/run_e7_squared_exp_night_one_click.sh").read_text(
        encoding="utf-8"
    )
    assert "p3_left_saturation" in launcher
    assert "DRPO_E7_P3_LEFT_SATURATION_FULL_RUN" in launcher
    assert "p2_left" not in launcher
