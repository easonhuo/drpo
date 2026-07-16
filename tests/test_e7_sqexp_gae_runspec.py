from __future__ import annotations

import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
AGENT_DIR = ROOT / "scripts" / "agent"
if str(AGENT_DIR) not in sys.path:
    sys.path.insert(0, str(AGENT_DIR))

from runspec_delivery_policy import validate_simple_size_policy  # noqa: E402
from runspec_lib import validate_runspec  # noqa: E402
from runspec_recovery import validate_recovery_policy  # noqa: E402
from runspec_registration import validate_registration_block  # noqa: E402
from runspec_results_delivery import validate_delivery_block  # noqa: E402


RUNSPEC = ROOT / "runspecs" / "ready" / "E7_SQEXP_GAE_PILOT_20260716_03.yaml"
RETIRED_RUNSPECS = (
    ROOT / "runspecs" / "retired" / "E7_SQEXP_GAE_PILOT_20260716_01.yaml",
    ROOT / "runspecs" / "retired" / "E7_SQEXP_GAE_PILOT_20260716_02.yaml",
)
FROZEN_IMPLEMENTATION = "5a286a83dff96853fc83c8ca361717fac07e7ee4"


def test_e7_sqexp_gae_ready_runspec_is_structurally_valid() -> None:
    spec = validate_runspec(ROOT, RUNSPEC, require_registry=False)
    registration = validate_registration_block(spec)
    delivery = validate_delivery_block(spec, "e7")
    validate_simple_size_policy(spec)
    validate_recovery_policy(ROOT, spec)

    assert spec["run_id"] == "E7_SQEXP_GAE_PILOT_20260716_03"
    assert spec["lane"] == "e7"
    assert spec["experiment_id"] == "EXT-H-E7-SQEXP-GAE-01"
    assert spec["execution_class"] == "pilot"
    assert spec["repo_commit"] == FROZEN_IMPLEMENTATION
    assert registration == {"mode": "deferred", "closure_required": True}
    assert delivery["enabled"] is True
    assert delivery["auto"] is True
    assert delivery["repository"] == "easonhuo/drpo-results"
    assert delivery["branch"] == "ingest/e7"
    assert spec["entrypoint"]["command"] == (
        "bash scripts/run_e7_sqexp_gae_auto_one_click.sh"
    )
    assert spec["outputs"]["run_dir"] == "outputs/e7/sqexp_gae_002"
    assert spec["recovery"]["enabled"] is True
    assert spec["recovery"]["resume_command"] == (
        "bash scripts/run_e7_sqexp_gae_auto_one_click.sh"
    )


def test_e7_sqexp_gae_ready_runspec_freezes_scientific_boundaries() -> None:
    raw = yaml.safe_load(RUNSPEC.read_text())
    purpose = str(raw["purpose"])
    success = "\n".join(str(item) for item in raw["success_criteria"])
    protected = set(raw["provenance"]["protected_paths"])
    includes = set(raw["artifacts"]["include"])

    assert raw["provenance"]["frozen_implementation_commit"] == FROZEN_IMPLEMENTATION
    assert (
        raw["provenance"]["supersedes_run_id"]
        == "E7_SQEXP_GAE_PILOT_20260716_02"
    )
    assert "192-branch" in purpose
    assert "development seeds 200-203" in purpose
    assert "c={64,128,256}" in purpose
    assert "--variant {variant}" in purpose
    assert "--eval_max_steps 1000" in purpose
    assert "float64 parity" in purpose
    assert "exactly 192 runnable branches" in success
    assert "held-out seeds untouched" in success
    assert "without failed-cell imputation" in success
    assert "not described as convergence" in success
    assert "stored_gae_dtype=float32" in success
    assert "storage quantization is reported separately" in success
    assert "absent from every materialized GAE trainer argument vector" in success
    assert "accepted by the checked-in GAE trainer parser" in success
    assert "monotonically indexed failed_attempts archive" in success
    assert (
        "outputs/e7/sqexp_gae_002/branches/*/failed_attempts/*/FAILED.json"
        in includes
    )
    assert (
        "outputs/e7/sqexp_gae_002/branches/*/failed_attempts/*/stdout_stderr.log"
        in includes
    )
    assert raw["policy"]["formal_evidence_allowed"] is False
    assert raw["policy"]["forbid_hparam_change"] is True
    assert "configs/e7_sqexp_gae_v1.json" in protected
    assert "src/drpo/e7_offline_gae.py" in protected
    assert "src/drpo/e7_sqexp_gae_prepare.py" in protected
    assert "src/drpo/e7_sqexp_gae_trainer.py" in protected
    assert "src/drpo/e7_sqexp_gae_contract.py" in protected
    assert "scripts/run_e7_sqexp_gae_auto_one_click.sh" in protected


def test_failed_single_use_runspecs_are_not_ready() -> None:
    for suffix in ("01", "02"):
        old_ready = (
            ROOT
            / "runspecs"
            / "ready"
            / f"E7_SQEXP_GAE_PILOT_20260716_{suffix}.yaml"
        )
        assert not old_ready.exists()

    assert all(path.is_file() for path in RETIRED_RUNSPECS)
    first_text = RETIRED_RUNSPECS[0].read_text()
    second_text = RETIRED_RUNSPECS[1].read_text()
    assert "RETIRED" in first_text
    assert "RETIRED" in second_text
    assert "No actor branch was started" in first_text
    assert "before actor update 1" in second_text
    assert "corrected _03" in second_text
