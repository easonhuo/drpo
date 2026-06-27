# Handoff deltas

Each directory contains one immutable Stage 3 shadow update. The directory name must equal `update_id` and contain `HANDOFF_DELTA.yaml`; successful shadow updates also retain `SHADOW_REPORT.json`. These records are governance provenance, not a second research master. `docs/handoff.md` remains authoritative throughout Stage 3.

The bootstrap directory is also a retained golden replay case, so it includes the exact gzip-compressed full candidate and the bootstrap Full-tier report. Normal successful updates do not retain a duplicate full candidate.
