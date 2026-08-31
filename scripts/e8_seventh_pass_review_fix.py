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

anchor = '''CANONICAL_COLD_MODULES = {
    "arena": "drpo.countdown_qwen_arena_onefile",
    # Import paper_runtime before the base runtime/trainer so its activation
    # patches the base symbols before the trainer binds them.
    "paper_common": "drpo.countdown_e8_alpha1_highc_scan_common",
    "paper_runtime": "drpo.countdown_e8_alpha1_highc_scan_runtime",
    "scan_common": "drpo.countdown_e8_alpha1_c_scan_common",
    "scan_runtime": "drpo.countdown_e8_alpha1_c_scan_runtime",
    "scan_trainer": "drpo.countdown_e8_alpha1_c_scan_trainer",
}
'''
replacement = anchor + '''
FROZEN_CANONICAL_COLDSTART_BLOB_SHAS = {
    "arena": "d8a04f3ae3edd08042aa1004b4cbf927fc5cea72",
    "scan_common": "572f6ad98bf063c88e52a4594fde892842c4fe15",
    "scan_runtime": "b4ad8581f0afd6e4d24069524f909eaa1b0c9563",
    "scan_trainer": "e026afbefc09205bb1632b5dd1bd6db33b5df358",
    "paper_common": "720415583e9e372fafa5aa3520e07de04e6494d8",
    "paper_runtime": "a57cd88287daf95864f7e30caf658473a32d3602",
    "base_config": "10f27f32719298376bdc7be7e01023626c6ad3f8",
    "round1_grid": "e6d70895ad9e4caceb029425fbed523b8530c2d3",
    "extension_grid": "e7657ef7c8fbb1ae81e0a7b9dd6f4b9cea32262d",
    "bank_generator": "545119fdf8e560b5e81a862e1e48134dd52ac869",
    "bank_config": "d1873efae15c778d2472a927206d8620aa43be71",
    "bank_converter": "a935e2d721b06437568556040475736bbf45ceee",
    "p0_bank_pipeline": "3482967fe656156500f4598f16f5e7031e198d48",
    "p0_task_adapters": "454f3076171ee25636109c33f5a177ee2201b5f8",
    "p0_config": "14605ae3a79f18e435feafd3927bc21485edbbc9",
    "p0_launcher": "ffcad2a64cb2f42906cae67dabdcc98c3eb46ff0",
    "result_reference": "972a67867aafb5ddea6e1625bacd337b6939f097",
}
'''
replace_once(py, anchor, replacement)

replace_once(
    py,
    '''        blob_shas = canonical.get("expected_git_blob_shas", {})
        if set(blob_shas) != set(expected_paths) or any(
            len(str(value)) != 40 for value in blob_shas.values()
        ):
            raise ValueError("Cold-start must pin every old source/config Git blob SHA")
        if canonical.get("scientific_kernel") != "import_only_no_loss_reimplementation":
''',
    '''        blob_shas = canonical.get("expected_git_blob_shas", {})
        if blob_shas != FROZEN_CANONICAL_COLDSTART_BLOB_SHAS:
            raise ValueError(
                "Cold-start canonical source/config Git blob identities drifted"
            )
        if (
            canonical.get("initialization")
            != "qwen_pretrained_base_plus_fresh_lora"
            or canonical.get("formula")
            != "alpha_times_exp_minus_c_times_current_sequence_surprisal_div_2"
        ):
            raise ValueError("Cold-start canonical initialization/formula provenance drifted")
        if canonical.get("scientific_kernel") != "import_only_no_loss_reimplementation":
''',
)

text = tests.read_text(encoding="utf-8")
marker = '''def test_generic_coldstart_rejects_scientific_input_drift() -> None:
'''
extra = '''def test_generic_coldstart_rejects_canonical_byte_identity_drift() -> None:
    from drpo import e8_multitask_exp_tuning as exp_tuning

    config = exp_tuning.load_config(
        Path("configs/e8_multitask_exp_lambda_curve_completion.yaml")
    )
    config["experiment_id"] = "EXT-C-E8-MULTITASK-EXP-CANONICAL-LOCK-UNSEEN-TEST"
    assert (
        config["canonical_coldstart"]["expected_git_blob_shas"]
        == exp_tuning.FROZEN_CANONICAL_COLDSTART_BLOB_SHAS
    )

    bad = copy.deepcopy(config)
    bad["canonical_coldstart"]["expected_git_blob_shas"]["scan_trainer"] = "0" * 40
    with pytest.raises(ValueError, match="canonical source/config Git blob identities drifted"):
        exp_tuning.validate_config(bad)

    for key, value in (
        ("initialization", "other_initialization"),
        ("formula", "other_formula"),
    ):
        bad = copy.deepcopy(config)
        bad["canonical_coldstart"][key] = value
        with pytest.raises(ValueError, match="canonical initialization/formula provenance drifted"):
            exp_tuning.validate_config(bad)


def test_all_frozen_coldstart_configs_share_exact_canonical_blob_lock() -> None:
    from drpo import e8_multitask_exp_tuning as exp_tuning

    for path in (
        Path("configs/e8_multitask_exp_coldstart.yaml"),
        Path("configs/e8_multitask_exp_lambda_completion.yaml"),
        Path("configs/e8_multitask_exp_lambda_curve_completion.yaml"),
    ):
        config = exp_tuning.load_config(path)
        assert (
            config["canonical_coldstart"]["expected_git_blob_shas"]
            == exp_tuning.FROZEN_CANONICAL_COLDSTART_BLOB_SHAS
        )


'''
if text.count(marker) != 1 or "test_generic_coldstart_rejects_canonical_byte_identity_drift" in text:
    raise SystemExit("cannot insert seventh-pass canonical lock tests")
tests.write_text(text.replace(marker, extra + marker, 1), encoding="utf-8")
