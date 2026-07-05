# Figure 1 data status

## Hopper
The Hopper panel uses the previously verified E7-Q2 plot data:
`data/hopper_e7_q2_plot_data.csv`.

## Countdown
The Countdown panel uses the uploaded raw CSV:
`data/countdown_uploaded_raw.csv`.

Raw CSV shape: 12000 rows x 14 columns.
Filtered rows used in the panel: 12000.
Filter: arithmetic_wrong responses with valid format, correct use of numbers, incorrect answer, finite surprisal, finite trainable-gradient norm, and finite negative coefficient magnitude.

The final plotted Countdown curve is an exact equal-count decile curve over current-policy mean-token surprisal.
The confidence band uses 400 bootstrap replicates with mode `cluster_by_puzzle_id_6000_clusters_x_2_responses`.

Key checks:
- negative coefficient magnitude unique values: [1.0]
- sample Pearson/Spearman surprisal-gradient correlations: 0.363 / 0.445
- lowest decile mean gradient norm: 40.274
- highest decile mean gradient norm: 98.563
- highest/lowest relative gradient: 2.447x

## Raw data storage policy

The full Countdown raw CSV with response text is intentionally not stored in ordinary Git. It remains an `external_user_archive_not_in_git` asset indexed by filename, size, and SHA-256 in the staging audit metadata; this repository stores only plot-ready deciles, summary JSON, manifests, and figure-specific data slices.
