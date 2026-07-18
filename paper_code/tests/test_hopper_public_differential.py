from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from drpo_reference.experiments import hopper as public
from drpo_reference.external.hopper_data import Normalizer, OfflineData
from drpo_reference.external.hopper_protocol import METHODS, smoke_protocol


def _data() -> OfflineData:
    count = 6
    return OfflineData(
        observations=np.arange(count * 3, dtype=np.float32).reshape(count, 3),
        actions=np.zeros((count, 2), dtype=np.float32),
        rewards=np.ones(count, dtype=np.float32),
        next_observations=np.ones((count, 3), dtype=np.float32),
        terminals=np.array([0, 1, 0, 1, 0, 1], dtype=np.bool_),
        timeouts=np.zeros(count, dtype=np.bool_),
        episode_ids=np.array([0, 0, 1, 1, 2, 2], dtype=np.int64),
    )


def _audit(value: float = 10.0, state: str = "finite_terminal") -> dict[str, Any]:
    return {
        "state": state,
        "fixed_budget_completed": True,
        "terminal_audit_complete": True,
        "task_performance_status": "available",
        "task_performance_collapse": False,
        "normalized_return_available": True,
        "support_boundary_event": False,
        "numerical_nonfinite": False,
        "final_normalized_return": value,
        "final_metrics": {
            "positive_nll": 1.0,
            "mean_boundary_fraction": 0.0,
            "sigma_mean": 1.0,
            "phantom_joint_output_score_mean": 2.0,
            "phantom_log_scale_to_mean_ratio": 1.0,
        },
    }


def _summary(seed: int, offset: float = 0.0) -> dict[str, Any]:
    methods = {
        method: _audit(value=20.0 + offset + index)
        for index, method in enumerate(METHODS)
    }
    mechanism_subchecks = {
        "natural_far_field_present": True,
        "corrected_quadratic_branch_empirically_active": True,
        "measurable_full_parameter_contribution": True,
        "log_scale_relative_dominance_observed": True,
        "targeted_control_outcomes_reported": True,
        "terminal_audit_records_complete": True,
        "terminal_state_classification_complete": True,
        "rollout_available_for_all_methods": True,
        "all_mechanism_subchecks_passed": True,
    }
    return {
        "seed": seed,
        "canonical_critic_seed": 100,
        "canonical_critic_artifact": {
            "identity": {"canonical_critic_seed": 100}
        },
        "critic": {
            "selected_checkpoint_metrics": {"test_r2": 0.5, "test_pearson": 0.9},
            "fixed_budget_completed": True,
            "optimization_terminal": True,
            "critic_accepted_for_frozen_advantage": True,
        },
        "advantage": {"positive_fraction": 0.5},
        "suite_status": "complete",
        "all_methods_completed": True,
        "all_branch_initial_states_identical": True,
        "prepared_checkpoint": {"reload_identity": True},
        "positive_only_initialization": _audit(value=15.0 + offset),
        "matching": {"pairs": 4},
        "gradient_probe": {
            "abs_advantage_far_near_ratio": 1.0,
            "standardized_distance_far_near_ratio": 3.0,
            "mean_output_score_far_near_ratio": 2.0,
            "raw_log_scale_output_score_far_near_ratio": 4.0,
            "corrected_q_xi_far_near_ratio": 9.0,
            "joint_output_score_far_near_ratio": 4.0,
            "log_scale_to_mean_far_near_ratio": 2.0,
            "full_parameter_gradient_far_near_ratio": 4.0 + offset,
            "mean_score_loglog_slope_vs_radius": 1.0,
            "corrected_q_xi_loglog_slope_vs_radius": 2.0,
            "analytic_autograd_relative_error_max": 1.0e-8,
            "natural_far_field_present": True,
        },
        "global_budget": {"global_scale": 0.5},
        "methods": methods,
        "branch_failures": {},
        "mechanism_subchecks": mechanism_subchecks,
    }


def _context(tmp_path: Path, data: OfflineData) -> public.CanonicalCriticContext:
    return public.CanonicalCriticContext(
        root=tmp_path / "canonical",
        split={
            "train": np.array([0, 1], dtype=np.int64),
            "validation": np.array([2, 3], dtype=np.int64),
            "test": np.array([4, 5], dtype=np.int64),
        },
        observation_normalizer=Normalizer(
            mean=np.zeros(3, dtype=np.float32),
            std=np.ones(3, dtype=np.float32),
        ),
        advantage_arrays={
            "advantage": np.array([1, -1, 1, -1, 1, -1], dtype=np.float32)
        },
        critic_audit={
            "fixed_budget_completed": True,
            "critic_accepted_for_frozen_advantage": True,
        },
        artifact_manifest={
            "complete": True,
            "identity": {"test": True},
            "critic_training_count": 1,
            "files": {},
        },
        reused=False,
    )


def test_execution_plan_separates_formal_subset_and_smoke() -> None:
    formal = public.resolve_hopper_execution()
    assert formal.execution_kind == "formal"
    assert formal.formal_evidence_eligible is True
    assert formal.seeds == tuple(range(100, 110))
    assert formal.method_ranking_claim_allowed is False

    subset = public.resolve_hopper_execution(seeds=(100, 101))
    assert subset.execution_kind == "formal_subset_non_evidence"
    assert subset.formal_evidence_eligible is False
    assert subset.protocol.critic_steps == 100_000

    smoke = public.resolve_hopper_execution(smoke=True)
    assert smoke.execution_kind == "smoke_non_evidence"
    assert smoke.seeds == (42,)
    assert smoke.protocol.critic_steps == 4

    with pytest.raises(ValueError, match="subset of the registered"):
        public.resolve_hopper_execution(seeds=(999,))
    with pytest.raises(ValueError, match="preserve registered seed order"):
        public.resolve_hopper_execution(seeds=(101, 100))
    with pytest.raises(ValueError, match="duplicates"):
        public.resolve_hopper_execution(seeds=(100, 100))


def test_dataset_identity_checks_basename_and_sha256(tmp_path: Path) -> None:
    path = tmp_path / "tiny.hdf5"
    path.write_bytes(b"registered-test-dataset")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    protocol = replace(
        smoke_protocol(),
        dataset_basename=path.name,
        dataset_sha256=digest,
    )
    manifest = public.validate_dataset_identity(path, protocol)
    assert manifest["identity_verified"] is True
    assert manifest["sha256"] == digest

    wrong_name = replace(protocol, dataset_basename="other.hdf5")
    with pytest.raises(ValueError, match="basename mismatch"):
        public.validate_dataset_identity(path, wrong_name)
    wrong_hash = replace(protocol, dataset_sha256="0" * 64)
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        public.validate_dataset_identity(path, wrong_hash)


def test_canonical_reuse_is_identity_and_hash_strict(tmp_path: Path) -> None:
    data = _data()
    protocol = smoke_protocol()
    dataset_manifest = {
        "basename": protocol.dataset_basename,
        "sha256": protocol.dataset_sha256,
    }
    root = tmp_path / "canonical"
    root.mkdir()
    (root / "canonical_critic.pt").write_bytes(b"selected")
    (root / "final_training_critic.pt").write_bytes(b"terminal")
    (root / "critic_metrics.csv").write_text("step,loss\n4,1\n", encoding="utf-8")
    (root / "critic_terminal_audit.json").write_text(
        json.dumps(
            {
                "fixed_budget_completed": True,
                "critic_accepted_for_frozen_advantage": True,
            }
        ),
        encoding="utf-8",
    )
    np.savez_compressed(
        root / "frozen_advantages.npz",
        advantage=np.array([1, -1, 1, -1, 1, -1], dtype=np.float32),
    )
    np.savez_compressed(
        root / "split_indices.npz",
        train=np.array([0, 1], dtype=np.int64),
        validation=np.array([2, 3], dtype=np.int64),
        test=np.array([4, 5], dtype=np.int64),
    )
    np.savez_compressed(
        root / "observation_normalizer.npz",
        mean=np.zeros(3, dtype=np.float32),
        std=np.ones(3, dtype=np.float32),
    )
    identity = public._canonical_identity(
        protocol=protocol,
        dataset_manifest=dataset_manifest,
        data=data,
    )
    manifest = {
        "schema_version": 1,
        "identity": identity,
        "complete": True,
        "critic_training_count": 1,
        "shared_across_all_actor_seeds": True,
        "files": public._hash_canonical_files(root),
    }
    (root / "canonical_critic_manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )

    context = public.prepare_canonical_critic_context(
        data=data,
        protocol=protocol,
        dataset_manifest=dataset_manifest,
        device="cpu",
        artifact_root=tmp_path / "unused",
        reuse_root=root,
    )
    assert context.reused is True
    assert context.artifact_manifest["identity"] == identity

    (root / "canonical_critic.pt").write_bytes(b"tampered")
    with pytest.raises(RuntimeError, match="file hashes"):
        public.prepare_canonical_critic_context(
            data=data,
            protocol=protocol,
            dataset_manifest=dataset_manifest,
            device="cpu",
            artifact_root=tmp_path / "unused-2",
            reuse_root=root,
        )


def test_aggregation_matches_legacy_event_separation() -> None:
    summaries = [_summary(100, 0.0), _summary(101, 2.0)]
    ours = public.aggregate_seed_summaries(summaries)

    root = Path(__file__).parents[2]
    module_path = root / "src" / "drpo" / "e7_hopper_q2.py"
    spec = importlib.util.spec_from_file_location("legacy_e7_hopper_public_test", module_path)
    assert spec and spec.loader
    legacy = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = legacy
    spec.loader.exec_module(legacy)
    theirs = legacy.aggregate_seed_summaries(summaries)

    assert ours["seeds_completed"] == theirs["seeds_completed"] == 2
    assert ours["terminal_state_counts"] == theirs["terminal_state_counts"]
    for method in METHODS:
        for key in (
            "task_performance_available_count",
            "task_performance_unavailable_count",
            "task_performance_collapse_count",
            "support_or_variance_boundary_count",
            "nan_inf_numerical_count",
        ):
            assert ours["reporting_separation"][method][key] == (
                theirs["reporting_separation"][method][key]
            )
    assert ours["method_ranking_claim_allowed"] is False
    assert theirs["method_ranking_claim_allowed"] is False
    assert ours["full_parameter_gradient_far_near_ratio"]["mean"] == pytest.approx(5.0)


def test_root_audit_never_authorizes_ranking(tmp_path: Path) -> None:
    data = _data()
    context = _context(tmp_path, data)
    plan = public.resolve_hopper_execution(seeds=(100, 101))
    audit = public.build_root_terminal_audit(
        summaries=[_summary(100), _summary(101)],
        plan=plan,
        canonical=context,
        rollout_preflight={"status": "passed"},
        dataset_manifest={"identity_verified": True},
    )
    assert audit["engineering_pipeline_complete"] is True
    assert audit["formal_evidence_prerequisites_complete"] is False
    assert audit["formal_scientific_gate_passed"] is False
    assert audit["method_ranking_claim_allowed"] is False
    assert audit["root_completion_marker_allowed"] is True


def test_public_runner_wires_rollouts_and_writes_completion(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data = _data()
    context = _context(tmp_path, data)
    dataset = tmp_path / "dataset.hdf5"
    dataset.write_bytes(b"placeholder")
    stages: list[str] = []

    monkeypatch.setattr(
        public,
        "validate_dataset_identity",
        lambda path, protocol: {
            "path": str(path),
            "basename": protocol.dataset_basename,
            "sha256": protocol.dataset_sha256,
            "size_bytes": 11,
            "identity_verified": True,
        },
    )
    monkeypatch.setattr(public, "load_hopper_hdf5", lambda path: data)
    monkeypatch.setattr(
        public,
        "prepare_canonical_critic_context",
        lambda **kwargs: context,
    )

    def fake_preflight(**kwargs: Any) -> dict[str, Any]:
        assert kwargs["required"] is True
        return {"status": "passed"}

    def fake_rollout(**kwargs: Any) -> dict[str, Any]:
        stages.append(str(kwargs["diagnostics_path"]))
        return {
            "rollout_status": "available",
            "rollout_return_mean": 1.0,
            "rollout_return_std": 0.0,
            "normalized_return": 10.0,
            "normalized_return_available": True,
            "rollout_episodes": kwargs["episodes"],
        }

    def fake_train_actor_stage(**kwargs: Any) -> tuple[object, dict[str, Any]]:
        rollout = kwargs["rollout_evaluator"](
            kwargs["policy"], kwargs["max_steps"], kwargs["method"]
        )
        audit = _audit(value=float(rollout["normalized_return"]))
        return kwargs["policy"], audit

    monkeypatch.setattr(public, "train_actor_stage", fake_train_actor_stage)

    def fake_suite(**kwargs: Any) -> dict[str, Any]:
        runner = kwargs["stage_runner"]
        positive_policy = object()
        _, positive = runner(
            policy=positive_policy,
            method="positive_only",
            output_dir=kwargs["output_dir"] / "positive_only_initialization",
            max_steps=kwargs["protocol"].positive_steps,
        )
        methods: dict[str, Any] = {}
        for method in METHODS:
            _, methods[method] = runner(
                policy=object(),
                method=method,
                output_dir=kwargs["output_dir"] / "methods" / method,
                max_steps=kwargs["protocol"].branch_steps,
            )
        return {
            **_summary(kwargs["seed"]),
            "positive_only_initialization": positive,
            "methods": methods,
        }

    output = tmp_path / "run"
    result = public.run_hopper(
        dataset_path=dataset,
        output_root=output,
        seeds=(42,),
        smoke=True,
        device="cpu",
        suite_runner=fake_suite,
        preflight_runner=fake_preflight,
        rollout_runner=fake_rollout,
    )
    assert result["completion"]["formal_result_claim"] is False
    assert result["terminal_audit"]["method_ranking_claim_allowed"] is False
    assert (output / "RUN_COMPLETE.json").is_file()
    assert (output / "terminal_audit.json").is_file()
    assert any("positive_only_initialization" in path for path in stages)
    assert any("methods/signed" in path for path in stages)


def test_public_runner_writes_incomplete_and_failure_records(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data = _data()
    context = _context(tmp_path, data)
    dataset = tmp_path / "dataset.hdf5"
    dataset.write_bytes(b"placeholder")
    monkeypatch.setattr(
        public,
        "validate_dataset_identity",
        lambda path, protocol: {
            "path": str(path),
            "basename": protocol.dataset_basename,
            "sha256": protocol.dataset_sha256,
            "size_bytes": 11,
            "identity_verified": True,
        },
    )
    monkeypatch.setattr(public, "load_hopper_hdf5", lambda path: data)
    monkeypatch.setattr(
        public,
        "prepare_canonical_critic_context",
        lambda **kwargs: context,
    )

    def partial_suite(**kwargs: Any) -> dict[str, Any]:
        summary = _summary(kwargs["seed"])
        summary["methods"].pop("signed")
        summary["all_methods_completed"] = False
        summary["suite_status"] = "partial_failure"
        summary["branch_failures"] = {
            "signed": {"exception_type": "RuntimeError", "message": "boom"}
        }
        return summary

    output = tmp_path / "partial"
    with pytest.raises(RuntimeError, match="incomplete"):
        public.run_hopper(
            dataset_path=dataset,
            output_root=output,
            seeds=(42,),
            smoke=True,
            device="cpu",
            suite_runner=partial_suite,
            preflight_runner=lambda **kwargs: {"status": "passed"},
            rollout_runner=lambda **kwargs: {},
        )
    assert (output / "RUN_INCOMPLETE.json").is_file()
    assert (output / "SCIENTIFIC_RUN_FAILED.json").is_file()
    assert not (output / "RUN_COMPLETE.json").exists()


def test_cli_exposes_hopper_and_dispatches(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from drpo_reference import cli

    captured: dict[str, Any] = {}

    def fake_run_hopper(**kwargs: Any) -> dict[str, Any]:
        captured.update(kwargs)
        return {}

    monkeypatch.setattr(cli, "run_hopper", fake_run_hopper)
    dataset = tmp_path / "hopper_medium_replay-v2.hdf5"
    output = tmp_path / "output"
    exit_code = cli.main(
        [
            "hopper",
            "--dataset",
            str(dataset),
            "--output",
            str(output),
            "--seeds",
            "100,101",
            "--device",
            "cpu",
        ]
    )
    assert exit_code == 0
    assert captured == {
        "dataset_path": dataset,
        "output_root": output,
        "seeds": (100, 101),
        "smoke": False,
        "device": "cpu",
        "critic_artifact": None,
    }
