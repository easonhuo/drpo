#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path


def replace_once(path: Path, old: str, new: str) -> None:
    text = path.read_text(encoding="utf-8")
    count = text.count(old)
    if count != 1:
        raise SystemExit(f"{path}: expected one target, found {count}: {old[:120]!r}")
    path.write_text(text.replace(old, new, 1), encoding="utf-8")


py = Path("src/drpo/e8_multitask_exp_tuning.py")
tests = Path("tests/test_e8_multitask_p0.py")

marker = "\n\n@dataclass(frozen=True)\nclass Cell:\n"
constants = '''

FROZEN_COLDSTART_PROVENANCE_IDENTITIES: dict[str, dict[str, Any]] = {
    COLDSTART_EXPERIMENT_ID: {
        "reporting": {
            "separate_events": [
                "task_performance",
                "valid_or_structure_diagnostic",
                "nan_inf_numerical_failure",
            ],
            "countdown_role": "diagnostic_regression_sentinel_not_result_gate",
            "other_tasks_task_interface_adaptation_only": True,
            "transfer_exp_scope": "single_seed_response_shape_localization",
            "positive_only_seed_count_per_transfer_task": 4,
            "convergence_claim_allowed": False,
            "significance_claim_allowed": False,
            "method_ranking_allowed": False,
            "causal_identification_environment": "D-U1",
        },
        "expected_waves": 13,
        "historical_curve_anchor": None,
    },
    LAMBDA_COMPLETION_EXPERIMENT_ID: {
        "reporting": {
            "separate_events": [
                "task_performance",
                "valid_or_structure_diagnostic",
                "nan_inf_numerical_failure",
            ],
            "countdown_role": "predecessor_only_no_new_scientific_cells",
            "other_tasks_task_interface_adaptation_only": True,
            "transfer_exp_scope": "single_seed_response_shape_localization",
            "positive_only_seed_count_per_transfer_task": 2,
            "convergence_claim_allowed": False,
            "significance_claim_allowed": False,
            "method_ranking_allowed": False,
            "scientific_role": "external_validity_response_shape",
        },
        "expected_waves": 13,
        "historical_curve_anchor": {
            "path": "experiments/results/e8_multitask_exp_coldstart_20260820_02/CURVE_ANCHOR.csv",
            "role": "immutable_predecessor_curve_for_concatenation",
        },
    },
    LAMBDA_CURVE_COMPLETION_EXPERIMENT_ID: {
        "reporting": {
            "separate_events": [
                "task_performance",
                "valid_or_structure_diagnostic",
                "nan_inf_numerical_failure",
            ],
            "countdown_role": "predecessor_only_no_new_scientific_cells",
            "other_tasks_task_interface_adaptation_only": True,
            "transfer_exp_scope": "single_seed_response_shape_localization",
            "positive_only_seed_count_per_transfer_task": 0,
            "convergence_claim_allowed": False,
            "significance_claim_allowed": False,
            "method_ranking_allowed": False,
            "scientific_role": "external_validity_curve_boundary_completion",
        },
        "expected_waves": 9,
        "historical_curve_anchor": {
            "path": "experiments/results/e8_multitask_exp_coldstart_20260820_02/CURVE_ANCHOR.csv",
            "role": "immutable_predecessor_curve_for_concatenation",
        },
    },
}
'''
replace_once(py, marker, constants + marker)

replace_once(
    py,
    '''def _configured_seed(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{label} must be a non-negative integer")
    return value


def experiment_id(config: Mapping[str, Any]) -> str:
''',
    '''def _configured_seed(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{label} must be a non-negative integer")
    return value


def _configured_bool(value: Any, label: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{label} must be boolean")
    return value


def experiment_id(config: Mapping[str, Any]) -> str:
''',
)

replace_once(
    py,
    '''    if observed != expected:
        raise ValueError(
            f"Frozen cold-start experiment identity drifted for {experiment_id(config)}"
        )


def sweep_profile(config: Mapping[str, Any]) -> str:
''',
    '''    if observed != expected:
        raise ValueError(
            f"Frozen cold-start experiment identity drifted for {experiment_id(config)}"
        )
    expected_provenance = FROZEN_COLDSTART_PROVENANCE_IDENTITIES[
        experiment_id(config)
    ]
    observed_provenance = {
        "reporting": copy.deepcopy(dict(config.get("reporting", {}))),
        "expected_waves": _configured_seed(
            config["execution"].get("expected_waves"),
            "Frozen cold-start expected_waves",
        ),
        "historical_curve_anchor": copy.deepcopy(
            config.get("historical_curve_anchor")
        ),
    }
    if observed_provenance != expected_provenance:
        raise ValueError(
            "Frozen cold-start reporting/execution provenance drifted for "
            f"{experiment_id(config)}"
        )


def _validate_coldstart_reporting(
    config: Mapping[str, Any],
    *,
    countdown_seeds: Sequence[int],
    transfer_positive_seeds: Sequence[int],
) -> None:
    reporting = config.get("reporting")
    if not isinstance(reporting, Mapping):
        raise ValueError("Cold-start reporting must be a mapping")
    if tuple(reporting.get("separate_events", ())) != (
        "task_performance",
        "valid_or_structure_diagnostic",
        "nan_inf_numerical_failure",
    ):
        raise ValueError("Cold-start reporting must preserve the three terminal event classes")
    if reporting.get("other_tasks_task_interface_adaptation_only") is not True:
        raise ValueError("Cold-start reporting must preserve task-interface-only adaptation")
    if reporting.get("transfer_exp_scope") != "single_seed_response_shape_localization":
        raise ValueError("Cold-start transfer reporting scope drifted")
    positive_count = _configured_seed(
        reporting.get("positive_only_seed_count_per_transfer_task"),
        "Cold-start reporting Positive-only seed count",
    )
    if positive_count != len(transfer_positive_seeds):
        raise ValueError(
            "Cold-start reporting Positive-only seed count must match configured seeds"
        )
    for field in (
        "convergence_claim_allowed",
        "significance_claim_allowed",
        "method_ranking_allowed",
    ):
        if reporting.get(field) is not False:
            raise ValueError(f"Cold-start reporting must keep {field}=false")
    expected_countdown_role = (
        "diagnostic_regression_sentinel_not_result_gate"
        if countdown_seeds
        else "predecessor_only_no_new_scientific_cells"
    )
    if reporting.get("countdown_role") != expected_countdown_role:
        raise ValueError("Cold-start Countdown reporting role does not match scheduled cells")
    if "scientific_role" in reporting:
        role = reporting.get("scientific_role")
        if not isinstance(role, str) or not role.startswith("external_validity"):
            raise ValueError("Cold-start scientific_role must remain external validity")
    if (
        "causal_identification_environment" in reporting
        and reporting.get("causal_identification_environment") != "D-U1"
    ):
        raise ValueError(
            "Cold-start causal-identification authority may only point to D-U1"
        )


def sweep_profile(config: Mapping[str, Any]) -> str:
''',
)

replace_once(
    py,
    '''        if expected_cells != expanded_cells:
            raise ValueError("Cold-start expected_cells must match the configured matrix")
        _validate_frozen_coldstart_sweep_identity(config)

        initialization = config.get("initialization", {})
''',
    '''        if expected_cells != expanded_cells:
            raise ValueError("Cold-start expected_cells must match the configured matrix")
        _validate_coldstart_reporting(
            config,
            countdown_seeds=countdown_seeds,
            transfer_positive_seeds=transfer_positive_seeds,
        )
        _validate_frozen_coldstart_sweep_identity(config)

        initialization = config.get("initialization", {})
''',
)

replace_once(
    py,
    '''    execution = config["execution"]
    expected_capacity = 16
    if int(execution["max_concurrent_cells"]) != expected_capacity:
        raise ValueError(f"The scheduler must expose exactly {expected_capacity} slots")
    if tuple(int(value) for value in execution["gpu_ids"]) != tuple(range(8)):
        raise ValueError("The default GPU pool must remain 0--7")
    if int(execution["slots_per_gpu"]) != 2:
        raise ValueError("The frozen topology is two slots per GPU")
    if not _is_coldstart(config):
        expected_waves = 7 if profile == SWEEP_PROFILE_DENSE else 5
        if int(execution["expected_waves"]) != expected_waves:
            raise ValueError(
                f"The frozen topology requires {expected_waves} waves for this profile"
            )
    if _is_coldstart(config) and execution.get("scheduler") != "dynamic_slot_queue":
        raise ValueError("Cold-start execution must use the recovery-aware slot scheduler")
    if _is_coldstart(config) and (
        bool(execution.get("wave_barriers", True))
        or not bool(execution.get("identity_checked_resume", False))
        or not bool(execution.get("retry_incomplete_requires_explicit_flag", False))
        or not bool(execution.get("fail_closed", False))
        or not bool(execution.get("test_partition_forbidden", False))
        or execution.get("oom_policy")
        != "fail_cell_no_automatic_scientific_parameter_mutation"
    ):
        raise ValueError("Cold-start recovery/OOM safety contract drifted")
''',
    '''    execution = config["execution"]
    expected_capacity = 16
    max_concurrent_cells = _configured_seed(
        execution.get("max_concurrent_cells"), "execution.max_concurrent_cells"
    )
    if max_concurrent_cells != expected_capacity:
        raise ValueError(f"The scheduler must expose exactly {expected_capacity} slots")
    raw_gpu_ids = execution.get("gpu_ids")
    if isinstance(raw_gpu_ids, (str, bytes)) or not isinstance(raw_gpu_ids, Sequence):
        raise ValueError("execution.gpu_ids must be a sequence of non-negative integers")
    gpu_ids = tuple(
        _configured_seed(value, "execution.gpu_ids entry") for value in raw_gpu_ids
    )
    if gpu_ids != tuple(range(8)):
        raise ValueError("The default GPU pool must remain 0--7")
    slots_per_gpu = _configured_seed(
        execution.get("slots_per_gpu"), "execution.slots_per_gpu"
    )
    if slots_per_gpu != 2:
        raise ValueError("The frozen topology is two slots per GPU")
    if not _is_coldstart(config):
        expected_waves = 7 if profile == SWEEP_PROFILE_DENSE else 5
        configured_waves = _configured_seed(
            execution.get("expected_waves"), "execution.expected_waves"
        )
        if configured_waves != expected_waves:
            raise ValueError(
                f"The frozen topology requires {expected_waves} waves for this profile"
            )
    if _is_coldstart(config) and execution.get("scheduler") != "dynamic_slot_queue":
        raise ValueError("Cold-start execution must use the recovery-aware slot scheduler")
    if _is_coldstart(config):
        wave_barriers = _configured_bool(
            execution.get("wave_barriers"), "execution.wave_barriers"
        )
        identity_checked_resume = _configured_bool(
            execution.get("identity_checked_resume"), "execution.identity_checked_resume"
        )
        retry_explicit = _configured_bool(
            execution.get("retry_incomplete_requires_explicit_flag"),
            "execution.retry_incomplete_requires_explicit_flag",
        )
        fail_closed = _configured_bool(
            execution.get("fail_closed"), "execution.fail_closed"
        )
        test_partition_forbidden = _configured_bool(
            execution.get("test_partition_forbidden"),
            "execution.test_partition_forbidden",
        )
        if (
            wave_barriers
            or not identity_checked_resume
            or not retry_explicit
            or not fail_closed
            or not test_partition_forbidden
            or execution.get("oom_policy")
            != "fail_cell_no_automatic_scientific_parameter_mutation"
        ):
            raise ValueError("Cold-start recovery/OOM safety contract drifted")
''',
)

text = tests.read_text(encoding="utf-8")
text = text.replace(
    'sweep["transfer_positive_only_seed_offsets"] = [8000, 9000]\n    sweep["countdown_seed_offsets"] = []\n',
    'sweep["transfer_positive_only_seed_offsets"] = [8000, 9000]\n    config["reporting"]["positive_only_seed_count_per_transfer_task"] = 2\n    sweep["countdown_seed_offsets"] = []\n',
    1,
)
text = text.replace(
    'sweep["countdown_include_positive_only"] = False\n    sweep["transfer_positive_only_seed_offsets"] = []\n',
    'sweep["countdown_include_positive_only"] = False\n    sweep["transfer_positive_only_seed_offsets"] = []\n    countdown_only["reporting"]["positive_only_seed_count_per_transfer_task"] = 0\n',
    1,
)

marker = '''def test_exp_coldstart_rejects_adapter_runtime_or_grid_drift() -> None:\n'''
extra = '''def test_generic_coldstart_execution_requires_exact_yaml_types() -> None:\n    from drpo import e8_multitask_exp_tuning as exp_tuning\n\n    config = exp_tuning.load_config(\n        Path("configs/e8_multitask_exp_lambda_curve_completion.yaml")\n    )\n    config["experiment_id"] = "EXT-C-E8-MULTITASK-EXP-EXEC-TYPE-UNSEEN-TEST"\n\n    for field, invalid in (("max_concurrent_cells", 16.9), ("slots_per_gpu", 2.9)):\n        bad = copy.deepcopy(config)\n        bad["execution"][field] = invalid\n        with pytest.raises(ValueError, match="non-negative integer"):\n            exp_tuning.validate_config(bad)\n\n    bad = copy.deepcopy(config)\n    bad["execution"]["gpu_ids"][1] = True\n    with pytest.raises(ValueError, match="non-negative integer"):\n        exp_tuning.validate_config(bad)\n\n    for field in (\n        "wave_barriers",\n        "identity_checked_resume",\n        "retry_incomplete_requires_explicit_flag",\n        "fail_closed",\n        "test_partition_forbidden",\n    ):\n        bad = copy.deepcopy(config)\n        bad["execution"][field] = "true"\n        with pytest.raises(ValueError, match="must be boolean"):\n            exp_tuning.validate_config(bad)\n\n\ndef test_generic_coldstart_reporting_preserves_external_validity_boundary() -> None:\n    from drpo import e8_multitask_exp_tuning as exp_tuning\n\n    config = exp_tuning.load_config(\n        Path("configs/e8_multitask_exp_lambda_curve_completion.yaml")\n    )\n    config["experiment_id"] = "EXT-C-E8-MULTITASK-EXP-REPORTING-UNSEEN-TEST"\n\n    bad = copy.deepcopy(config)\n    bad["reporting"]["separate_events"] = ["task_performance", "nan_inf_numerical_failure"]\n    with pytest.raises(ValueError, match="three terminal event classes"):\n        exp_tuning.validate_config(bad)\n\n    for field in (\n        "convergence_claim_allowed",\n        "significance_claim_allowed",\n        "method_ranking_allowed",\n    ):\n        bad = copy.deepcopy(config)\n        bad["reporting"][field] = True\n        with pytest.raises(ValueError, match=field):\n            exp_tuning.validate_config(bad)\n\n    bad = copy.deepcopy(config)\n    bad["reporting"]["positive_only_seed_count_per_transfer_task"] = 1\n    with pytest.raises(ValueError, match="Positive-only seed count"):\n        exp_tuning.validate_config(bad)\n\n    bad = copy.deepcopy(config)\n    bad["reporting"]["scientific_role"] = "causal_identification"\n    with pytest.raises(ValueError, match="scientific_role must remain external validity"):\n        exp_tuning.validate_config(bad)\n\n    bad = copy.deepcopy(config)\n    bad["reporting"]["causal_identification_environment"] = "countdown"\n    with pytest.raises(ValueError, match="only point to D-U1"):\n        exp_tuning.validate_config(bad)\n\n\ndef test_historical_coldstart_reporting_wave_and_anchor_provenance_are_frozen() -> None:\n    from drpo import e8_multitask_exp_tuning as exp_tuning\n\n    paths = (\n        Path("configs/e8_multitask_exp_coldstart.yaml"),\n        Path("configs/e8_multitask_exp_lambda_completion.yaml"),\n        Path("configs/e8_multitask_exp_lambda_curve_completion.yaml"),\n    )\n    for path in paths:\n        config = exp_tuning.load_config(path)\n\n        changed = copy.deepcopy(config)\n        changed["reporting"]["scientific_role"] = "external_validity_alternate_label"\n        with pytest.raises(ValueError, match="reporting/execution provenance drifted"):\n            exp_tuning.validate_config(changed)\n\n        changed = copy.deepcopy(config)\n        changed["execution"]["expected_waves"] += 1\n        with pytest.raises(ValueError, match="reporting/execution provenance drifted"):\n            exp_tuning.validate_config(changed)\n\n        changed = copy.deepcopy(config)\n        changed["historical_curve_anchor"] = {\n            "path": "other.csv",\n            "role": "immutable_predecessor_curve_for_concatenation",\n        }\n        with pytest.raises(ValueError, match="reporting/execution provenance drifted"):\n            exp_tuning.validate_config(changed)\n\n\n'''
if marker not in text or "test_generic_coldstart_execution_requires_exact_yaml_types" in text:
    raise SystemExit("cannot insert ninth-pass tests")
text = text.replace(marker, extra + marker, 1)
tests.write_text(text, encoding="utf-8")
