from __future__ import annotations

import json
from pathlib import Path

import pytest

from drpo import countdown_e8_oracle_offline_v2_matrix as runner


def test_plan_invariants_match_spec() -> None:
    runner.assert_plan_invariants()
    training = [c.name for c in runner.TRAINING_CELLS]
    assert len(training) == 8
    assert [c for c in training if c.startswith("base_")] == [
        "base_positive_only",
        "base_bank_global_matched_x0p25",
        "base_bank_global_matched_x0p5",
        "base_bank_global_matched_x1p0",
        "base_bank_global_matched_x2p0",
    ]
    assert [c for c in training if c.startswith("low_sft_")] == [
        "low_sft_positive_only",
        "low_sft_bank_global_matched_x1p0",
    ]
    assert [c for c in training if c.startswith("full_sft_")] == ["full_sft_positive_only"]


def test_no_onpolicy_or_replay_or_c2_in_plan() -> None:
    names = {c.name for c in runner.CELLS}
    for forbidden in (
        "base_onpolicy_positive_only",
        "base_online_replay_positive_only",
        "base_online_replay_pos_neg",
    ):
        assert forbidden not in names, f"{forbidden} must not be in the v2 init-matrix"
    assert not any(c.name.startswith("C2") for c in runner.CELLS), "C2 is not registered"
    assert "A0" not in names, "A0 does not enter the execution queue"


def test_eval_only_cells_are_b0_c0() -> None:
    assert {c.name for c in runner.EVAL_ONLY_CELLS} == {"B0", "C0"}


def test_alias_cells_reference_base_training_products() -> None:
    a1 = next(c for c in runner.ALIAS_CELLS if c.name == "A1")
    a2 = next(c for c in runner.ALIAS_CELLS if c.name == "A2")
    assert a1.alias_of == "base_positive_only"
    assert a2.alias_of == "base_bank_global_matched_x1p0"
    assert all(c.kind == "alias" for c in runner.ALIAS_CELLS)


def test_calibration_routing_isolated_by_init() -> None:
    for c in runner.TRAINING_CELLS:
        if c.method == "positive_only":
            assert c.calibration == "none"
        if c.init == runner.BASE_INIT and c.method == "bank_global_matched":
            assert c.calibration == "base"
        if c.init == runner.LOW_SFT_INIT and c.method == "bank_global_matched":
            assert c.calibration == "low_sft"
        if c.calibration == "base":
            assert c.init == runner.BASE_INIT
        if c.calibration == "low_sft":
            assert c.init == runner.LOW_SFT_INIT
        if c.init == runner.FULL_SFT_INIT:
            assert c.method == "positive_only"
            assert c.calibration == "none"


def test_identity_match_accepts_exact_and_rejects_drift() -> None:
    ident = {
        "experiment_id": runner.EXPERIMENT_ID,
        "runner_version": runner.VERSION,
        "config_sha256": "c",
        "bank_sha256": "b",
        "val_sha256": "v",
        "test_sha256": "t",
        "model_sha256": "m",
        "git_commit": "g",
        "git_dirty": False,
        "offline_training": {"steps": 1200},
        "evaluation": {"seed": 1},
    }
    assert runner._identity_matches(ident, ident)
    for key in ("config_sha256", "bank_sha256", "val_sha256", "test_sha256",
                "model_sha256", "git_commit", "git_dirty"):
        drift = dict(ident)
        drift[key] = "CHANGED" if isinstance(ident[key], str) else (not ident[key])
        assert not runner._identity_matches(ident, drift), f"drift in {key} must invalidate"
    drift = dict(ident)
    drift["offline_training"] = {"steps": 9999}
    assert not runner._identity_matches(ident, drift)


def test_selftest_entrypoint(capsys: pytest.CaptureFixture[str]) -> None:
    runner.cmd_selftest(object())
    assert "V2_INIT_MATRIX_SELFTEST_OK" in capsys.readouterr().out


def test_plan_subcommand_dumps_machine_readable(capsys: pytest.CaptureFixture[str]) -> None:
    args = runner.build_parser().parse_args(["plan"])
    args.func(args)
    out = capsys.readouterr().out
    plan = json.loads(out)
    assert isinstance(plan, list)
    assert {p["name"] for p in plan} >= {c.name for c in runner.CELLS}


def test_converter_fail_closed_on_duplicates(tmp_path: Path) -> None:
    import sys
    repo = Path(runner.__file__).resolve().parents[2]
    sys.path.insert(0, str(repo / "scripts"))
    import v2_bank_convert  # type: ignore[import-not-found]
    # build a mini corpus with a duplicate expression in one row
    row = {
        "row_id": "r0",
        "source_prompt_id": "p0",
        "prompt": "Numbers: 1,2,3,4\nTarget: 10",
        "oracle_positive": "1+2+3+4",
        "oracle_structure": "A+B+C+D",
        "negatives": [
            {"expression": "1+2+3-4", "structure": "A+B+C-D", "tree_depth": 1, "value_error": 8.0, "negative_bin": "near_value_wrong"},
            {"expression": "1+2+3-4", "structure": "A+B+C-D", "tree_depth": 1, "value_error": 8.0, "negative_bin": "near_value_wrong"},
        ],
    }
    src = tmp_path / "in.jsonl"
    src.write_text(json.dumps(row) + "\n")
    v2_bank_convert.V2_TRAIN = src
    v2_bank_convert.OUT = tmp_path / "out.jsonl"
    v2_bank_convert.MANIFEST = tmp_path / "manifest.json"
    with pytest.raises(RuntimeError, match="duplicate expressions"):
        v2_bank_convert.main()


def test_converter_fail_closed_on_more_than_16(tmp_path: Path) -> None:
    import sys
    repo = Path(runner.__file__).resolve().parents[2]
    sys.path.insert(0, str(repo / "scripts"))
    import v2_bank_convert  # type: ignore[import-not-found]
    negs = [
        {"expression": f"{i}+0", "structure": "A+B", "tree_depth": 1,
         "value_error": float(i), "negative_bin": "far_value_wrong"}
        for i in range(17)
    ]
    row = {
        "row_id": "r0", "source_prompt_id": "p0",
        "prompt": "p", "oracle_positive": "1", "oracle_structure": "A",
        "negatives": negs,
    }
    src = tmp_path / "in.jsonl"
    src.write_text(json.dumps(row) + "\n")
    v2_bank_convert.V2_TRAIN = src
    v2_bank_convert.OUT = tmp_path / "out.jsonl"
    v2_bank_convert.MANIFEST = tmp_path / "manifest.json"
    with pytest.raises(RuntimeError, match="> 16"):
        v2_bank_convert.main()
