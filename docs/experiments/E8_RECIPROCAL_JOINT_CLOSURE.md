# E8 reciprocal joint pilot closure

## Decision

The reciprocal lambda exploration line is closed. No fourth scan or further right-boundary expansion is authorized by this closure.

## Locked pilot claim

Under the frozen Countdown current-surprisal coordinate, fixed 1200-step training budget, paired development seeds, and validation-only protocol, Reciprocal-Linear and Reciprocal-Quadratic move from under-suppression into a broad plateau as lambda increases, but do not reach the sustained validation behavior of the matched exponential anchor. This supports tail-decay shape, rather than simple global strength alone, as an important control factor in this Countdown external-validity setting.

## Evidence chain

- Shape screen: 16/16 cells, lambda 1/3/7/19, two reciprocal families.
- High-lambda extension: 16/16 cells, lambda 39/79/159/319, two reciprocal families.
- Dense Rec-Q curve: 32/32 cells, sixteen additional points through lambda 1279.
- Joint total: 64/64 cells; every terminal audit passed; numerical failures 0; test split unused.

## Claim boundary

All three records remain pilots. Two paired development seeds and a fixed 1200-step horizon do not establish convergence, steady state, statistical significance, or a formal method ranking. Valid rate remains a structural diagnostic and is not formal support recovery. Task performance, support/structure diagnostics, and NaN/Inf failure remain separate reports. The result does not establish universal exponential superiority across tasks, models, seeds, or parameterizations.

## Provenance resolution

The high-lambda run source commit is directly resolvable. The shape-screen and dense-run local source commits are not authoritative GitHub refs; their registered RunSpec snapshot commits were therefore audited against the result-recorded SHA-256 values for every active protected runtime source plus the exact grid config. The machine-readable mapping is in `E8_RECIPROCAL_JOINT_PROVENANCE_AUDIT.json`.

## Governance path

The user explicitly approved a manual V1 fallback and a one-time remote bootstrap workflow. The ordinary preparation/V1 fastpath supports one experiment entity per transaction, while this closure atomically registers three linked pilot entities. The workflow is removed in the same final closure commit after trusted-current-main normalization, authority verification, and repository gates.
