# EXT-H-E7-BENCH-01 joint-critic GAE tuning

This code-first pilot substage adds the manuscript threshold `tau` to the merged joint actor--critic GAE path and prepares the first common-`c` D4RL-9 response-curve screen. It does not authorize a formal run or a method-ranking claim.

The Gaussian coordinate remains the existing normalized squared standardized distance `D=(d/reference_distance)^2`. The new taper is `w(D)=exp(-taper_lambda*relu((D-tau)/c))`. Public `c` is a positive denominator; the existing exponential adapter receives only the derived slope `taper_lambda/c`.

P1 is frozen to all nine D4RL-v2 locomotion cells, development seeds `200,201`, GAE lambda `0.95`, taper lambda `1`, `tau=0`, Positive-only, nine common `c` values, and one uncontrolled anchor: `9 x 2 x 11 = 198` branches. Per-task retuning and held-out seeds `204--207` are forbidden.

Implementation must extend existing Python files only, preserve the historical squared-EXP and three-task GAE update semantics, and avoid implementing later tau, Global-alpha, or GAE-lambda sweeps before P1 closes. Exact dataset SHA-256 values, source RunSpec, implementation SHA, liveness, registration, and terminal aggregation remain launch gates. Task-performance collapse, support/variance boundary, rollout failure, and NaN/Inf numerical failure remain separate.
