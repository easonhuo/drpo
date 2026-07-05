# Paper figure artifact index

This index maps paper figure outputs back to their scripts and structured result assets.

| Figure asset | Claim / role | Script | Result inputs | Output files | Status |
|---|---|---|---|---|---|
| `figure1_external_gradient` | External far-field gradient evidence in Hopper and Countdown | `scripts/figures/plot_figure1_external_gradient.py` | `results/FIGURE1_EXTERNAL_GRADIENT/` | `.pdf`, `.svg` | real-data-backed pilot/external diagnostics; Countdown is single-seed full-bank pilot |
| `fig_6_3_1_source_heatmap` | Controlled source isolation: gradient scale comes from policy-relative remoteness rather than worse advantages | `scripts/figures/plot_figure2_controlled_source_heatmap.py` | `results/FIGURE2_CONTROLLED_SOURCE_TRANSMISSION/fig_6_3_1_source_heatmap_data.csv` | `.pdf`, `.svg` | controlled real-data-backed mechanism summary |
| `fig_6_3_2_rescue_plot` | Controlled causal transmission and rescue: far-field interventions prevent task-performance collapse | `scripts/figures/plot_figure2_controlled_rescue_plot.py` | `results/FIGURE2_CONTROLLED_SOURCE_TRANSMISSION/fig_6_3_2_causal_summary.csv` | `.pdf`, `.svg` | controlled real-data-backed mechanism summary |
| `fig_6_4_1_phase_transition` | Stable extrapolation / phase transition layout for Section 6.4.1 | `scripts/figures/plot_figure3_phase_transition.py` | `results/FIGURE3_PHASE_TRANSITION/fig_6_4_1_phase_transition_template.csv` | `.pdf`, `.svg` | template/layout; formal phase-scan aggregates pending |
| `fig_6_4_2_leftfig_bigtext_legend_protocol` | Taper/control transfer layout for Section 6.4.2/6.4.3 | `scripts/figures/plot_figure4_taper_left_panel.py` | `results/FIGURE4_TAPER_CONTROL_TRANSFER/` | `.pdf`, `.svg` | template/layout; formal aggregates pending |

## Result storage notes

- Countdown full raw CSV is excluded from Git. The plot-ready deciles and summary are stored under `results/EXT-C-E8-V75/gradient_probe/` and duplicated for Figure 1 under `results/FIGURE1_EXTERNAL_GRADIENT/`.
- Figure 1 and Figure 2 assets are suitable for paper drafting with the status caveats recorded in their manifests.
- Figure 3 and Figure 4 assets are scaffolds. Their manifests and captions must be replaced or amended before any formal claim is made from their numbers.
