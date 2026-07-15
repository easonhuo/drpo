"""Run one branch of the E7 squared-EXP high-c boundary pilot."""

from __future__ import annotations

import math
from typing import Any, Mapping

from drpo import e7_sqexp_actor_decision_bootstrap as predecessor
from drpo.e7_sqexp_highc_boundary import (
    EXPERIMENT_ID,
    REFERENCE_DISTANCE,
    SQUARED_FORMULA,
)


EXPECTED_CONTROLS = {
    "squared_c256": 256.0,
    "squared_c512": 512.0,
}


def _validate_weight_control(raw: Mapping[str, Any]) -> dict[str, Any]:
    forbidden = {"negative_scale", "canonical_alpha", "effective_alpha"}
    present = sorted(forbidden & set(raw))
    if present:
        raise ValueError("public weight control forbids legacy fields: " + ", ".join(present))
    control_id = str(raw.get("id"))
    family = str(raw.get("family"))
    weight_at_zero = float(raw.get("weight_at_zero"))
    coefficient = float(raw.get("exp_coefficient"))
    reference_distance = float(raw.get("reference_distance"))
    formula = str(raw.get("formula"))
    if control_id not in EXPECTED_CONTROLS:
        raise ValueError(f"unsupported high-c boundary control: {control_id}")
    if family != "squared_exponential":
        raise ValueError(f"{control_id} family must remain squared_exponential")
    if not math.isclose(weight_at_zero, 1.0, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError(f"{control_id} w(0) must remain 1")
    if not math.isclose(
        coefficient,
        EXPECTED_CONTROLS[control_id],
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError(f"{control_id} coefficient changed")
    if not math.isclose(
        reference_distance,
        REFERENCE_DISTANCE,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError(f"{control_id} reference distance changed")
    if formula != SQUARED_FORMULA:
        raise ValueError(f"{control_id} formula changed")
    return {
        "id": control_id,
        "family": family,
        "weight_at_zero": weight_at_zero,
        "exp_coefficient": coefficient,
        "reference_distance": reference_distance,
        "formula": formula,
    }


def main(argv: list[str] | None = None) -> int:
    previous_experiment_id = predecessor.EXPERIMENT_ID
    previous_validator = predecessor._validate_weight_control  # noqa: SLF001
    predecessor.EXPERIMENT_ID = EXPERIMENT_ID
    predecessor._validate_weight_control = _validate_weight_control  # noqa: SLF001
    try:
        return predecessor.main(argv)
    finally:
        predecessor.EXPERIMENT_ID = previous_experiment_id
        predecessor._validate_weight_control = previous_validator  # noqa: SLF001


if __name__ == "__main__":
    raise SystemExit(main())
