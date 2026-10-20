# REGEN-GAE-CANONICAL-EXTENSION-01 — Leakage Exclusions

This file is evaluator-only. It is not included in either generator task packet.

The following later historical facts are withheld from both Arm A and Arm B until
their terminal outputs are frozen:

- PR #92 implementation diff and module layout;
- PR #102 and PR #107 compact implementation diffs;
- the historical counts `5093`, `3255`, `1363`, and `896`;
- the later recommendation to use a specific 35-line recurrence module;
- the exact compact adapter implementation;
- the historical unresolved `{variant}` placeholder failure;
- the historical mixed float32-storage versus float64-reference validation error;
- the historical `--eval_max_steps 1000` parser mismatch;
- the fact that the old actor launch reached `0/192` actor updates;
- the names and contents of later regression tests written specifically for those
  failures;
- the later review conclusion that a standalone trainer/runner stack was
  duplicated;
- candidate gate diagnostics from the E8 historical replay;
- the other regenerated arm's prompts, patches, tests, gate verdicts, and timing.

The task contract may require inspection and reuse of existing repository
responsibilities, but it must not reveal which exact later architecture was judged
best.

After both arms terminate, these facts may be used for descriptive comparison and
failure analysis. They may not be used to grant an additional repair attempt.
