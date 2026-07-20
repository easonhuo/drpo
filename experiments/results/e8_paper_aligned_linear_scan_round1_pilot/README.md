# E8 Paper-Aligned Linear Scan — Round-1 Pilot Result

Experiment: `EXT-C-E8-ORACLE-OFFLINE-V2-PAPER-ALIGNED-LINEAR-SCAN-0.5B-01`

This directory preserves the compact, non-weight result evidence received from the completed 32-cell development sweep. The raw upload package is not committed; its SHA-256 is `00d53286a6642998b5563045bbb278876ba38713f98819c679ed4825e49bdd48`.

## Status

- scientific status: `pilot`;
- 16 parameter points x 2 paired development seeds = 32 cells;
- all cells complete; terminal audit `PASS`;
- validation split only; test data not used;
- 0 NaN/Inf failures;
- fixed 1200 steps are not convergence or steady state;
- no formal method-ranking or statistical-significance claim.

## Main observation

The primary late-window Pass@8 rises from `0.0236` at the uncontrolled `c=0` endpoint to `0.1398` for Positive-only, then reaches `0.1572` at `c=1.897119985` and `0.1568` at the tested right boundary `c=2.995732274`. The right boundary remains near-tied with the best point, so Round 1 has not yet located the descending right branch.

Task performance, valid-structure proxy degradation, and NaN/Inf numerical failure are reported separately.
