# ReplayAB R1-versus-R2 Controlled Discrimination Decision

Work ID: `REPLAYAB-R1-R2-DISCRIMINATION-01`

Base: `main@b18aea9186d7e3ccc5d43b456719cafc23761e03`

Decision: `PASS_CONTROLLED_ADVANTAGE`

Evidence grade: `C2 -- controlled independently labelled semantic discrimination`

## Result

On the frozen 16-pair / 32-arm controlled bank, R2 reduced correct-arm false rejection from `0.200` to `0.000` while both judges retained an incorrect-arm false-acceptance rate of `0.000`.

R1 arm accuracy was `0.875` and pair accuracy was `0.750`. R2 arm and pair accuracy were both `1.000`. R1 released efficiency for `4/8` truly eligible pairs; R2 released `8/8`. Both release precisions were `1.000`.

The frozen judge-level success gate passed. The final PR validation also requires the pre-existing R1 terminal non-regression suite, focused ReplayAB suite, full repository pytest, Ruff, compilation, and governance checks to remain green.

## Supported claim

Within this frozen controlled bank, R2 is a strict judge-level capability extension over R1 exact-artifact mode: it preserves rejection of the predeclared incorrect outcomes while reducing false rejection caused solely by implementation non-identity.

## Not supported

This result is not a live coding-agent A/B, does not estimate population-level error rates, does not prove Candidate 01 improves work, does not replace R1 for deterministic exact-output tasks, and does not complete ReplayAB R3, R4, R5, or R6.

No scientific experiment, handoff state, registry state, or R2 closure state changed.
