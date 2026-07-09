from __future__ import annotations

from pathlib import Path

import pytest

from drpo import countdown_e8_base_rl_replay as runner


def test_load_config_scope_guards() -> None:
    config = runner.load_config()
    assert config["experiment_id"] == runner.EXPERIMENT_ID
    assert tuple(config["methods"]) == runner.METHODS
    assert config["scope_guards"]["no_countdown_sft_warmstart"] is True
    assert config["scope_guards"]["starts_from_qwen_pretrained_base"] is True


def test_prompt_plan_and_rollout_selection() -> None:
    plan = runner.prompt_plan(5, seed=11, steps=7, prompts_per_step=2)
    assert len(plan) == 7
    assert all(len(chunk) == 2 for chunk in plan)
    row = {
        "id": "p0",
        "prompt": "Numbers: 1, 2, 3, 4\nTarget: 10",
        "numbers": [1, 2, 3, 4],
        "target": 10,
    }
    positives, negatives, stats = runner.verify_rollouts(
        [row],
        [["1+2+3+4", "1+2+3-4", "(1+2)*(3+4)", "bad text"]],
        max_correct_per_prompt=1,
        max_negative_per_prompt=2,
    )
    assert len(positives) == 1
    assert positives[0]["completion"].replace(" ", "") == "1+2+3+4"
    assert len(negatives) >= 1
    assert stats["sampled_completions"] == 4.0
    assert stats["usable_positive_prompt_fraction"] == 1.0


def test_load_config_rejects_method_drift(tmp_path: Path) -> None:
    config_text = Path(runner.DEFAULT_CONFIG).read_text().replace("  - base_eval\n", "")
    path = tmp_path / "bad.yaml"
    path.write_text(config_text)
    with pytest.raises(ValueError, match="method set/order"):
        runner.load_config(path)


def test_selftest_entrypoint(capsys: pytest.CaptureFixture[str]) -> None:
    runner.cmd_selftest(object())
    assert "BASE_RL_REPLAY_SELFTEST_OK" in capsys.readouterr().out
