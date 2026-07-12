from pathlib import Path


SCRIPT = Path("scripts/run_e7_canonical_exp_horizon_joint_one_click.sh")


def test_e7_exp_horizon_one_click_is_resumable() -> None:
    text = SCRIPT.read_text(encoding="utf-8")
    run_marker = "python scripts/run_e7_canonical_exp_horizon_joint.py run"
    assert run_marker in text
    run_section = text.split(run_marker, 1)[1]
    assert "--resume" in run_section
    assert "--max-workers 60" in run_section
