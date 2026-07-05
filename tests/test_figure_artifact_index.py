from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_figure_scripts_indexed_and_present() -> None:
    readme = (ROOT / "scripts/figures/README.md").read_text()
    index = (ROOT / "paper/figures/FIGURE_INDEX.md").read_text()
    scripts = [
        "plot_figure1_external_gradient.py",
        "plot_figure2_controlled_source_heatmap.py",
        "plot_figure2_controlled_rescue_plot.py",
        "plot_figure3_phase_transition.py",
        "plot_figure4_taper_left_panel.py",
    ]
    for script in scripts:
        assert (ROOT / "scripts/figures" / script).exists()
        assert script in readme
        assert script in index


def test_figure_result_statuses_are_explicit() -> None:
    figure1 = json.loads(
        (ROOT / "results/FIGURE1_EXTERNAL_GRADIENT/figure1_external_gradient_manifest.json").read_text()
    )
    figure2 = json.loads(
        (ROOT / "results/FIGURE2_CONTROLLED_SOURCE_TRANSMISSION/manifest.json").read_text()
    )
    figure3 = json.loads((ROOT / "results/FIGURE3_PHASE_TRANSITION/manifest.json").read_text())
    figure4 = json.loads((ROOT / "results/FIGURE4_TAPER_CONTROL_TRANSFER/manifest.json").read_text())

    assert "real-data-backed" in figure1["data_status"]
    assert "real-data-backed" in figure2["data_status"]
    assert "template" in figure3["data_status"]
    assert "template" in figure4["data_status"]


def test_countdown_gradient_summary_matches_plot_deciles() -> None:
    summary = json.loads(
        (ROOT / "results/EXT-C-E8-V75/gradient_probe/countdown_gradient_summary_seed100.json").read_text()
    )
    deciles = (ROOT / "results/EXT-C-E8-V75/gradient_probe/countdown_gradient_deciles_seed100.csv").read_text().splitlines()
    assert summary["raw_shape"] == [12000, 14]
    assert summary["filtered_rows"] == 12000
    assert summary["negative_coefficient_abs_unique"] == [1.0]
    assert len(deciles) == 11  # header + 10 equal-count decile rows
    assert "external_user_archive_not_in_git" in (
        ROOT / "results/EXT-C-E8-V75/gradient_probe/DATA_STATUS.md"
    ).read_text()
