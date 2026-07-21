from pathlib import Path


def replace(path: str, old: str, new: str, expected: int = 1) -> None:
    target = Path(path)
    text = target.read_text()
    found = text.count(old)
    if found != expected:
        raise RuntimeError(
            f"{path}: expected {expected} occurrence(s), found {found}: {old[:120]!r}"
        )
    target.write_text(text.replace(old, new))


runner = Path("src/drpo/e7_squared_exp_night.py").read_text()
for required in (
    'TUNING_PROFILE_ID = "d4rl9_common_c_p2_left"',
    'TUNING_EXPECTED_BRANCHES = 180',
    'TUNING_LIVENESS_SCALE = 0.1',
    'TUNING_FULL_RUN_ENV = "DRPO_E7_P2_LEFT_FULL_RUN"',
    '"uncontrolled_anchor": False',
    '"stage": "p2_left_common_c_screen"',
):
    if required not in runner:
        raise RuntimeError(f"P2 runner contract missing: {required}")
if 'controls.append(("uncontrolled", None, 0.0))' in runner:
    raise RuntimeError("P2 runner still creates an Uncontrolled branch")

script = Path("scripts/run_e7_squared_exp_night_one_click.sh").read_text()
for required in (
    'p2_left)',
    'configs/e7_bench_joint_gae_tuning_p2_left_c.json',
    '[[ "${DRPO_E7_P2_LEFT_FULL_RUN:-0}" != "1" ]]',
):
    if required not in script:
        raise RuntimeError(f"P2 one-click contract missing: {required}")

aggregate = "src/drpo/e7_squared_exp_night_aggregate.py"
replace(aggregate, 'TUNING_PROFILE_ID = "d4rl9_common_c_p1"', 'TUNING_PROFILE_ID = "d4rl9_common_c_p2_left"')
replace(aggregate, "TUNING_EXPECTED_BRANCHES = 198", "TUNING_EXPECTED_BRANCHES = 180")
replace(
    aggregate,
    "TUNING_SCALES = (0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0)",
    "TUNING_SCALES = (0.2, 0.16, 0.125, 0.1, 0.08, 0.0625, 0.04, 0.025, 0.015625)",
)
replace(aggregate, "P1 requires canonical A2C", "P2 left requires canonical A2C")
replace(aggregate, "P1 public taper contract changed", "P2 left public taper contract changed")
replace(
    aggregate,
    'method not in {"positive_only", "thresholded_exponential", "uncontrolled"}',
    'method not in {"positive_only", "thresholded_exponential"}',
)
replace(aggregate, "unknown P1 control", "unknown P2 left control")
replace(aggregate, "def _p1_label", "def _tuning_label")
replace(aggregate, "def _p1_aggregate", "def _tuning_aggregate")
replace(aggregate, "_p1_label(row)", "_tuning_label(row)")
replace(aggregate, "_p1_aggregate(work, rows, mode)", "_tuning_aggregate(work, rows, mode)")
replace(
    aggregate,
    'expected = 3 if mode == "liveness" else TUNING_EXPECTED_BRANCHES',
    'expected = 2 if mode == "liveness" else TUNING_EXPECTED_BRANCHES',
)
replace(aggregate, "expected {expected} P1 branches", "expected {expected} P2-left branches")
replace(aggregate, "P1 full branch did not reach one million steps", "P2-left full branch did not reach one million steps")
replace(
    aggregate,
    'required = {"positive_only", "uncontrolled", "drpo_c4"}',
    'required = {"positive_only", "drpo_c0.1"}',
)
replace(aggregate, "P1 liveness controls changed", "P2-left liveness controls changed")
replace(aggregate, '"expected_branch_count": 3', '"expected_branch_count": 2')
replace(
    aggregate,
    'expected_controls = {"positive_only", "uncontrolled", *(f"drpo_c{x:g}" for x in TUNING_SCALES)}',
    'expected_controls = {"positive_only", *(f"drpo_c{x:g}" for x in TUNING_SCALES)}',
)
for old, new in (
    ("P1 task-seed matrix changed", "P2-left task-seed matrix changed"),
    ("P1 control matrix changed", "P2-left control matrix changed"),
    ("P1 seed set changed", "P2-left seed set changed"),
    ("P1 task coverage changed", "P2-left task coverage changed"),
    ("p1_paired_deltas.csv", "p2_paired_deltas.csv"),
    ("p1_task_summary.csv", "p2_task_summary.csv"),
    ("p1_control_summary.csv", "p2_control_summary.csv"),
    ("p1_stratum_summary.csv", "p2_stratum_summary.csv"),
):
    replace(aggregate, old, new)

kernel_test = "tests/test_e7_squared_exp_kernel.py"
replace(kernel_test, "configs/e7_bench_joint_gae_tuning_p1_c.json", "configs/e7_bench_joint_gae_tuning_p2_left_c.json")
replace(kernel_test, "test_p1_grid_builds_exact_198_branch_common_c_matrix", "test_p2_left_grid_builds_exact_180_branch_common_c_matrix")
replace(kernel_test, "assert len(branches) == 198", "assert len(branches) == 180")
replace(kernel_test, "assert len(group) == 11", "assert len(group) == 10")
replace(
    kernel_test,
    '''                assert sum(\n                    branch.template_values["weight_method"] == "uncontrolled"\n                    for branch in group\n                ) == 1\n''',
    '''                assert not any(\n                    branch.template_values["weight_method"] == "uncontrolled"\n                    for branch in group\n                )\n                assert {\n                    float(branch.template_values["remoteness_scale"])\n                    for branch in group\n                    if branch.template_values["weight_method"]\n                    == "thresholded_exponential"\n                } == set(night.TUNING_REMOTENESS_SCALES)\n''',
)
replace(kernel_test, "test_p1_public_c_maps_to_existing_exponential_slope", "test_p2_left_public_c_maps_to_existing_exponential_slope")
replace(kernel_test, '== 4.0\n        )', '== 0.1\n        )')
replace(kernel_test, 'assert public["remoteness_scale"] == 4.0', 'assert public["remoteness_scale"] == 0.1')
replace(kernel_test, 'assert public["derived_exp_coefficient"] == 0.25', 'assert public["derived_exp_coefficient"] == 10.0')
replace(kernel_test, 'assert internal.exponential_coefficient == 0.25', 'assert internal.exponential_coefficient == 10.0')
replace(kernel_test, "test_p1_full_run_requires_explicit_authorization", "test_p2_left_full_run_requires_explicit_authorization")
replace(kernel_test, "test_p1_authorized_run_uses_existing_runner_and_aggregator", "test_p2_left_authorized_run_uses_existing_runner_and_aggregator")

night_test = "tests/test_e7_squared_exp_night.py"
replace(night_test, "def _write_p1_branch", "def _write_p2_branch")
replace(night_test, "def test_p1_full_aggregate_is_task_balanced_and_claim_bounded", "def test_p2_left_full_aggregate_is_task_balanced_and_claim_bounded")
replace(night_test, "_write_p1_branch(", "_write_p2_branch(")
replace(
    night_test,
    '''    controls = [\n        ("positive_only", None, 0.0),\n        *(("thresholded_exponential", scale, 2.0 - abs(math.log2(scale / 4.0))) for scale in night.TUNING_REMOTENESS_SCALES),\n        ("uncontrolled", None, -2.0),\n    ]''',
    '''    controls = [\n        ("positive_only", None, 0.0),\n        *(\n            (\n                "thresholded_exponential",\n                scale,\n                2.0 - abs(math.log2(scale / 0.1)),\n            )\n            for scale in night.TUNING_REMOTENESS_SCALES\n        ),\n    ]''',
)
replace(night_test, 'assert summary["branch_count"] == 198', 'assert summary["branch_count"] == 180')
replace(night_test, 'assert summary["control_count"] == 11', 'assert summary["control_count"] == 10')
replace(night_test, "p1_paired_deltas.csv", "p2_paired_deltas.csv")
replace(night_test, "assert len(paired) == 181", "assert len(paired) == 163")

runspec_test = "tests/test_e7_squared_exp_night_runspecs.py"
replace(
    runspec_test,
    "def test_p1_one_click_does_not_self_authorize_full_run() -> None:",
    "def test_p2_left_one_click_requires_standard_runspec_authorization() -> None:",
)
replace(runspec_test, 'assert "export DRPO_E7_P1_FULL_RUN=1" not in script', 'assert "export DRPO_E7_P2_LEFT_FULL_RUN=1" not in script')
replace(runspec_test, 'assert \'[[ "${DRPO_E7_P1_FULL_RUN:-0}" != "1" ]]\' in script', 'assert \'[[ "${DRPO_E7_P2_LEFT_FULL_RUN:-0}" != "1" ]]\' in script')
replace(runspec_test, 'assert "authorized only by the standard RunSpec entrypoint" in script', 'assert "P2-left mode is authorized only by the standard RunSpec entrypoint" in script')
