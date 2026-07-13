# EXT-H-E7-PPO-W0-EXP-GRID-01 review checklist

- [ ] public config exposes only `w(0)` and `c` as negative-weight coordinates;
- [ ] branch identity and finalized diagnostics do not persist legacy scale/alpha fields;
- [ ] `w(0)=0.11` reproduces the historical scale-1 weighting exactly;
- [ ] 31 unique parameter points and 186 branches are materialized;
- [ ] only seeds 200 and 201 are used;
- [ ] held-out seeds 204--207 are absent from branch plans;
- [ ] PPO settings and canonical trainer settings remain frozen;
- [ ] runtime autotune changes only active subprocess count;
- [ ] selected workers are frozen into run identity;
- [ ] task degradation, support/variance boundary, and NaN/Inf remain separate;
- [ ] 500k is labeled screening rather than convergence;
- [ ] real-data liveness has been run on the server;
- [ ] authoritative handoff/registry delta has been materialized;
- [ ] RunSpec contains an exact reviewed commit and is promoted to ready;
- [ ] CI and governance checks pass;
- [ ] explicit user approval obtained before merge.
