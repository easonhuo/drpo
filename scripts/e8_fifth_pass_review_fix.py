#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path


def replace_once(path: Path, old: str, new: str) -> None:
    text = path.read_text(encoding="utf-8")
    count = text.count(old)
    if count != 1:
        raise SystemExit(
            f"{path}: expected one target, found {count}: {old[:120]!r}"
        )
    path.write_text(text.replace(old, new, 1), encoding="utf-8")


py = Path("src/drpo/e8_multitask_exp_tuning.py")
tests = Path("tests/test_e8_multitask_p0.py")

anchor = '''LOCKED_COUNTDOWN_COEFFICIENTS = frozenset(
    PAPER_ROUND1_COEFFICIENTS + PAPER_EXTENSION_COEFFICIENTS
)
FROZEN_COLDSTART_SWEEP_IDENTITIES: dict[str, dict[str, Any]] = {
'''
replacement = '''LOCKED_COUNTDOWN_COEFFICIENTS = frozenset(
    PAPER_ROUND1_COEFFICIENTS + PAPER_EXTENSION_COEFFICIENTS
)
FROZEN_COLDSTART_GRID_PROVENANCE = {
    "word_sorting": "historical_19_plus_dense_refinement_1.05",
    "spiral_matrix": "historical_7_anchors_plus_13_canonical_paper_coefficients",
    "mini_sudoku": "historical_19_plus_dense_refinement_2.00",
    "maze": "historical_19_plus_dense_refinement_0.10",
    "word_ladder": "historical_19_plus_dense_refinement_1.05",
    "knights_knaves": "historical_19_plus_dense_refinement_1.28",
    "graph_color": "historical_19_plus_dense_refinement_0.10",
    "wikisql": "historical_19_plus_dense_refinement_2.35",
}
FROZEN_LAMBDA_COMPLETION_GRID_PROVENANCE = {
    task: "approved_5x_geometric_tail_completion_20260825"
    for task in FROZEN_COLDSTART_P0_TASK_ORDER
}
FROZEN_LAMBDA_CURVE_GRID_PROVENANCE = {
    "word_sorting": "approved_right_tail_curve_completion_20260829",
    "spiral_matrix": "closed_no_new_cells_20260829",
    "mini_sudoku": "approved_right_tail_curve_completion_20260829",
    "maze": "approved_right_tail_curve_completion_20260829",
    "word_ladder": "approved_right_tail_curve_completion_20260829",
    "knights_knaves": "approved_right_tail_curve_completion_20260829",
    "graph_color": "approved_left_right_boundary_completion_20260829",
    "wikisql": "approved_aggressive_right_tail_curve_completion_20260829",
}
FROZEN_COLDSTART_SWEEP_IDENTITIES: dict[str, dict[str, Any]] = {
'''
replace_once(py, anchor, replacement)

for experiment, provenance_name in (
    ("COLDSTART_EXPERIMENT_ID", "FROZEN_COLDSTART_GRID_PROVENANCE"),
    ("LAMBDA_COMPLETION_EXPERIMENT_ID", "FROZEN_LAMBDA_COMPLETION_GRID_PROVENANCE"),
    ("LAMBDA_CURVE_COMPLETION_EXPERIMENT_ID", "FROZEN_LAMBDA_CURVE_GRID_PROVENANCE"),
):
    old = f'''    {experiment}: {{\n        "parameterization": '''
    text = py.read_text(encoding="utf-8")
    if text.count(old) != 1:
        raise SystemExit(f"cannot locate frozen identity entry for {experiment}")
    # Add suite/reporting metadata immediately after the entry opener.
    new = f'''    {experiment}: {{\n        "excluded_tasks": {{}},\n        "task_grid_provenance": {provenance_name},\n        "parameterization": '''
    py.write_text(text.replace(old, new, 1), encoding="utf-8")

replace_once(
    py,
    '''    sweep = config["sweep"]
    observed = {
        "parameterization": str(sweep.get("parameterization", "")),
''',
    '''    sweep = config["sweep"]
    observed = {
        "excluded_tasks": dict(config["suite"].get("excluded_tasks", {})),
        "task_grid_provenance": dict(sweep.get("task_grid_provenance", {})),
        "parameterization": str(sweep.get("parameterization", "")),
''',
)

replace_once(
    py,
    '''        "expected_cells": int(sweep.get("expected_cells", -1)),
''',
    '''        "expected_cells": _configured_seed(
            sweep.get("expected_cells"), "Cold-start expected_cells"
        ),
''',
)

replace_once(
    py,
    '''        if expanded_cells <= 0:
            raise ValueError("Cold-start sweep must schedule at least one scientific cell")
        if int(sweep.get("expected_cells", -1)) != expanded_cells:
            raise ValueError("Cold-start expected_cells must match the configured matrix")
''',
    '''        if expanded_cells <= 0:
            raise ValueError("Cold-start sweep must schedule at least one scientific cell")
        expected_cells = _configured_seed(
            sweep.get("expected_cells"), "Cold-start expected_cells"
        )
        if expected_cells != expanded_cells:
            raise ValueError("Cold-start expected_cells must match the configured matrix")
''',
)

text = tests.read_text(encoding="utf-8")
marker = '''def test_generic_coldstart_global_endpoint_and_countdown_controls_are_config_driven() -> None:
'''
extra = '''def test_frozen_coldstart_ids_reject_provenance_and_reporting_drift() -> None:
    from drpo import e8_multitask_exp_tuning as exp_tuning

    for path in (
        Path("configs/e8_multitask_exp_coldstart.yaml"),
        Path("configs/e8_multitask_exp_lambda_completion.yaml"),
        Path("configs/e8_multitask_exp_lambda_curve_completion.yaml"),
    ):
        config = exp_tuning.load_config(path)

        bad = copy.deepcopy(config)
        bad["sweep"]["task_grid_provenance"]["word_sorting"] = "tampered_provenance"
        with pytest.raises(ValueError, match="Frozen cold-start experiment identity drifted"):
            exp_tuning.validate_config(bad)

        bad = copy.deepcopy(config)
        bad["suite"]["excluded_tasks"] = {"word_sorting": "false_historical_exclusion"}
        with pytest.raises(ValueError, match="Frozen cold-start experiment identity drifted"):
            exp_tuning.validate_config(bad)


def test_generic_coldstart_expected_cells_requires_integer_type() -> None:
    from drpo import e8_multitask_exp_tuning as exp_tuning

    config = exp_tuning.load_config(
        Path("configs/e8_multitask_exp_lambda_curve_completion.yaml")
    )
    config["experiment_id"] = "EXT-C-E8-MULTITASK-EXP-CELL-TYPE-UNSEEN-TEST"
    for invalid in (140.0, 140.9, "140", True):
        bad = copy.deepcopy(config)
        bad["sweep"]["expected_cells"] = invalid
        with pytest.raises(ValueError, match="Cold-start expected_cells must be a non-negative integer"):
            exp_tuning.validate_config(bad)


'''
if text.count(marker) != 1 or "test_frozen_coldstart_ids_reject_provenance_and_reporting_drift" in text:
    raise SystemExit("cannot insert fifth-pass provenance tests")
tests.write_text(text.replace(marker, extra + marker, 1), encoding="utf-8")
