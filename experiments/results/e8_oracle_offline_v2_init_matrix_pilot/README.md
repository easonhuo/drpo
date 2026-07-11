# E8 oracle-offline v2 init-matrix pilot

- Experiment: `EXT-C-E8-ORACLE-OFFLINE-V2-INIT-MATRIX-0.5B-01`
- Run commit: `fe214f010bd5fec1e0e6a83f8297132a9ae8882b` (`git_dirty=true`)
- Status: pilot only; no formal method ranking.

Base positive-only learns. In the tested 0.25/0.5/1/2 range, larger negative pressure generally worsens pass@8 and terminal validity; x1/x2 show severe task/output-validity degradation without NaN/Inf. The bank uses fixed tensor width 16: 4943 rows have 16 unique expressions and 1057 rows have 9--15 unique expressions padded by cycling exact duplicates.
