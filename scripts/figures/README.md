# DRPO figure-generation scripts

This directory contains the reproducible plotting entry points for the current paper Figure 1--4 staging assets and registered paper-facing diagnostic figures.

## Figure index

| Figure asset | Script | Input directory | Output stem | Data status |
|---|---|---|---|---|
| Figure 1 external gradient | `scripts/figures/plot_figure1_external_gradient.py` | `results/FIGURE1_EXTERNAL_GRADIENT/` | `paper/figures/figure1_external_gradient` | real-data-backed pilot/external diagnostics |
| Figure 2a source heatmap | `scripts/figures/plot_figure2_controlled_source_heatmap.py` | `results/FIGURE2_CONTROLLED_SOURCE_TRANSMISSION/` | `paper/figures/fig_6_3_1_source_heatmap` | controlled real-data-backed mechanism summary |
| Figure 2b causal rescue | `scripts/figures/plot_figure2_controlled_rescue_plot.py` | `results/FIGURE2_CONTROLLED_SOURCE_TRANSMISSION/` | `paper/figures/fig_6_3_2_rescue_plot` | controlled real-data-backed mechanism summary |
| Figure 3 phase transition | `scripts/figures/plot_figure3_phase_transition.py` | `results/FIGURE3_PHASE_TRANSITION/` | `paper/figures/fig_6_4_1_phase_transition` | template/layout; formal phase-scan aggregates pending |
| Figure 4 taper/control transfer | `scripts/figures/plot_figure4_taper_left_panel.py` | `results/FIGURE4_TAPER_CONTROL_TRANSFER/` | `paper/figures/fig_6_4_2_leftfig_bigtext_legend_protocol` | template/layout; formal 6.4.2/6.4.3 aggregates pending |
| Structured-9 DRPO coefficient response | `scripts/figures/plot_e8_multitask_exp_response_curves.py` | `paper/iclr2027/figures/data/e8_multitask_exp_response/` | `paper/overleaf/figures/fig_app_structured9_drpo_coefficient_response` (PDF); ICLR working PNG/SVG previews use `paper/iclr2027/figures/fig_e8_multitask_exp_response_nine_panel` | historical real-data-backed `pilot_response_shape_only`; uniform visualization smoothing |

## Optional plotting dependencies

Pandas, NumPy, and Matplotlib are needed only when regenerating figure assets. Install them in the active environment before running the rebuild commands:

```bash
python3 -m pip install -r scripts/figures/requirements.txt
```

The ICLR build installs these optional dependencies because it materializes the Structured-9 DRPO coefficient-response PDF at the exact manuscript asset path before copying manuscript assets into the release project.

## Rebuild commands

```bash
python3 scripts/figures/plot_figure1_external_gradient.py \
  --hopper results/FIGURE1_EXTERNAL_GRADIENT/hopper_e7_q2_plot_data.csv \
  --countdown results/FIGURE1_EXTERNAL_GRADIENT/countdown_gradient_deciles_seed100.csv \
  --out paper/figures/figure1_external_gradient

python3 scripts/figures/plot_figure2_controlled_source_heatmap.py \
  --data results/FIGURE2_CONTROLLED_SOURCE_TRANSMISSION/fig_6_3_1_source_heatmap_data.csv \
  --out paper/figures/fig_6_3_1_source_heatmap

python3 scripts/figures/plot_figure2_controlled_rescue_plot.py \
  --summary results/FIGURE2_CONTROLLED_SOURCE_TRANSMISSION/fig_6_3_2_causal_summary.csv \
  --out paper/figures/fig_6_3_2_rescue_plot

python3 scripts/figures/plot_figure3_phase_transition.py \
  --data results/FIGURE3_PHASE_TRANSITION/fig_6_4_1_phase_transition_template.csv \
  --out paper/figures/fig_6_4_1_phase_transition

python3 scripts/figures/plot_figure4_taper_left_panel.py \
  --input results/FIGURE4_TAPER_CONTROL_TRANSFER/fig_6_4_2_leftfig_template.csv \
  --out paper/figures/fig_6_4_2_leftfig_bigtext_legend_protocol

python3 scripts/figures/plot_e8_multitask_exp_response_curves.py
```

The E8 response renderer uses one fixed presentation rule for every task:
`320` uniform points in `log10(lambda)` space, Gaussian mean sigma `12`,
Gaussian band sigma `12`, and band scale `1.45`. These are visualization
parameters only; they do not alter stored observations or scientific status.
The default PDF output is the exact asset already referenced by the current
ICLR manuscript: `paper/overleaf/figures/fig_app_structured9_drpo_coefficient_response.pdf`.

## Storage policy

Plot-ready CSV/JSON summaries, manifests, TeX snippets, and ordinary final figure assets are kept in Git. Large raw CSVs, replay buffers, checkpoints, model weights, and mega logs are not kept in ordinary Git; their filename, size, and SHA-256 must be recorded in the relevant `DATA_STATUS.md` or manifest. The Structured-9 response PDF is deterministically materialized from its checked-in plot-ready data by the plotting script and the ICLR build, so the release project contains the generated paper-facing PDF at the manuscript path.

Figure 3 and Figure 4 files currently preserve layout and writing scaffolds. Do not cite their template numbers as formal experimental results until their manifests are updated to point to formal aggregates.
