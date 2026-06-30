# C-U1 E4 Taper Budget-Match compact deposition

This directory deposits the compact, repository-tracked result for `C-U1-E4-TAPER-BUDGET-MATCH-01` run `run_003` at commit `1faea3a92f74af5d11409779d96b9ed21fe846ad`.

The formal matrix completed all 140 method-seed runs and passed the terminal audit. The primary fairness coordinate was the per-step raw negative-gradient L2 norm before Adam. Adam parameter-update norms were logged but were not matched. The result status is **finite-step validated**.

The original guard marked the lifecycle as failed only after computation because the runner omitted `scientific_run_manifest.json` and the default 25 MiB package limit was exceeded. The original failed run tree is preserved in the separately delivered full-raw sidecar whose SHA-256 is recorded in `RESULT_SUMMARY.json`. This compact directory does not replace raw trajectories, checkpoints, logs, or the original failure evidence.

Current C-U1 terminology is held-out-context or unseen-state generalization, not OOD generalization. Task-performance collapse, support/variance-boundary events, and NaN/Inf failures remain separately reported.
