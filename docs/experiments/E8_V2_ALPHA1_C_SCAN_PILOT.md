# E8 V2 Alpha=1 High-c One-Parameter Scan Pilot

## Identity

- Experiment: `EXT-C-E8-ORACLE-OFFLINE-V2-ALPHA1-C-SCAN-0.5B-01`
- Registration update: `EXT-C-E8-V2-ALPHA1-C-SCAN-REGISTRATION-2026-07-14`
- Scientific status: `pilot`; no terminal result yet
- Execution state: server run started from launch commit `a54dc74b849561c15f6195336fca446ed36f0640`
- Role: Countdown external-validity tuning only

## Question

Determine whether the two-parameter controller `alpha*exp(-c*u^2)` can be simplified on this development environment to `exp(-c*u^2)` by fixing `alpha=1` and tuning only `c`.

## Frozen matrix

- Positive-only: `(alpha=0,c=0)`
- Previous-best same-seed control: `(alpha=0.5,c=1.0)`
- Alpha-one scan: `c={1.5,2.0,2.25,2.5,3.0,4.0}`
- Development seed offsets: `5000,6000,7000,8000`
- Total: `8 parameter points x 4 seeds = 32 cells`

## Inherited protocol

The model, frozen V2 bank, fresh-LoRA initialization, loss formula, unique-negative denominator, optimizer, learning rate, 1200-step horizon, evaluation cadence, validation set, and test prohibition are unchanged from the preceding continuous-EXP grid. Every first-occurrence unique negative participates. Near/far extreme selection, weight-sum normalization, hidden scaling, budget matching, dynamic alpha, SBRC, Hybrid, entropy bonus, SFT warmstart, on-policy sampling, replay refresh, and test access remain forbidden.

## Runtime

Eight candidate GPUs use two parent-controller runtime slots per GPU, for at most 16 concurrent cells and two expected waves. Runtime selection may not change the scientific matrix. A real two-step liveness gate, clean checkout, protected-source hashes, and identity-checked resume are required.

The server started before authoritative registration. Registration therefore leaves all launch-identity files unchanged while the run is active. The authoritative registry records the launch commit and the pre-registration runtime metadata explicitly.

## Reporting and non-claims

Report terminal and steps 800--1200 late-window metrics, Pass@64, Greedy, valid rate, weight/remoteness diagnostics, raw gradient norm, optimizer update norm, and NaN/Inf separately. Fixed 1200 steps are not convergence or steady state. Four development seeds do not establish a formal method ranking. Countdown does not replace C-U1/D-U1 controlled mechanism evidence, and no OOD claim is permitted.
