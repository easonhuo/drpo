from __future__ import annotations

import json
import math
import subprocess
from pathlib import Path

import pytest
import torch

from drpo import countdown_e8_paper_aligned_lambda_minimal_common as paper

REPO = Path(__file__).resolve().parents[1]
CONFIG = REPO / paper.DEFAULT_GRID_CONFIG
PREDECESSOR_BLOBS = {
    "src/drpo/countdown_e8_alpha1_c_scan_common.py": "572f6ad98bf063c88e52a4594fde892842c4fe15",
    "src/drpo/countdown_e8_alpha1_c_scan_trainer.py": "27d8e926222d75d79ae600c93ca633d1cf4c4753",
    "src/drpo/countdown_e8_alpha1_c_scan_runtime.py": "b4ad8581f0afd6e4d24069524f909eaa1b0c9563",
    "scripts/run_countdown_e8_oracle_offline_v2_alpha1_c_scan_auto.py": "a56dcf097f32eb0b6f4b70316b982c15abc8ecaa",
}


def test_predecessor_files_are_blob_identical() -> None:
    for relative, expected in PREDECESSOR_BLOBS.items():
        actual = subprocess.check_output(
            ["git", "hash-object", str(REPO / relative)], text=True
        ).strip()
        assert actual == expected, relative


def test_grid_is_32_cells_on_8_by_2_runtime() -> None:
    config = paper.load_yaml(CONFIG)
    paper.validate_grid_config(config)
    cells = paper.build_cells(config)
    assert len(cells) == 32
    assert len(paper.PARAMETER_POINTS) == 16
    assert paper.SEED_OFFSETS == (4000, 5000)
    assert sum(cell.method == "positive_only" for cell in cells) == 2
    assert sum(cell.method == "uncontrolled_negative" for cell in cells) == 2
    assert config["execution"]["default_gpus"] == list(range(8))
    assert config["execution"]["parallel_cells_per_gpu"] == 2
    assert 8 * 2 == 16


def test_paper_formula_has_no_extra_square(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calibration = tmp_path / "calibration.json"
    calibration.write_text(
        json.dumps(
            {
                "experiment_id": paper.EXPERIMENT_ID,
                "tau": 2.0,
                "scale_c": 3.0,
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv(paper.CALIBRATION_ENV, str(calibration))
    seq_lp = torch.tensor([-1.0, -2.0, -5.0, -8.0], requires_grad=True)
    weights = paper.continuous_exp_weights(
        seq_lp, alpha=1.0, c=math.log(2.0)
    )
    assert torch.allclose(weights, torch.tensor([1.0, 1.0, 0.5, 0.25]), atol=1e-6)
    assert not math.isclose(float(weights[-1]), math.exp(-4.0 * math.log(2.0)))
    assert weights.requires_grad is False


def test_calibration_is_deterministic() -> None:
    result = paper.calibration_from_surprisals(
        [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        minimum_scale=1e-6,
        minimum_active_fraction=0.25,
    )
    assert result["tau"] == pytest.approx(3.5)
    assert result["scale_c"] == pytest.approx(3.0)
    assert result["active_fraction"] == pytest.approx(0.5)


def test_adapters_do_not_reimplement_scheduler_or_trainer() -> None:
    runtime = (
        REPO / "src/drpo/countdown_e8_paper_aligned_lambda_minimal_runtime.py"
    ).read_text(encoding="utf-8")
    launcher = (
        REPO / "scripts/run_countdown_e8_paper_aligned_lambda_minimal_auto.py"
    ).read_text(encoding="utf-8")
    assert "ThreadPoolExecutor" not in runtime
    assert "DataLoader" not in runtime
    assert "torch.optim" not in runtime
    assert "_base_runtime" in runtime
    assert "_BASE_LAUNCHER" in launcher


def test_no_wrong_rewritten_stack_is_present_on_successor_branch() -> None:
    forbidden = [
        "src/drpo/countdown_e8_paper_aligned_lambda_common.py",
        "src/drpo/countdown_e8_paper_aligned_lambda_trainer.py",
        "src/drpo/countdown_e8_paper_aligned_lambda_runtime.py",
        "scripts/run_countdown_e8_paper_aligned_lambda_auto.py",
    ]
    for relative in forbidden:
        assert not (REPO / relative).exists()
