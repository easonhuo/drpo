# Runtime worker-cap authorization records

This directory stores immutable, exact-run JSON approvals governed by
`docs/runtime_worker_cap_approval.md`.

Rules:

- `README.md` is documentation and is never an approval record.
- An AI agent may draft a proposed record but may not mark or use it as approved without explicit repository-owner approval.
- A usable record must be merged and exist byte-for-byte on trusted `origin/main`; a local commit, development branch, or unmerged pull request is not authorization.
- Every record must be committed, clean, and scoped to one experiment, one work directory, one exact cap, one observed CPU affinity set, and exact contract/run-spec/grid hashes.
- Changing, removing, or replacing a cap requires a new approval merged to `origin/main` and a new run/work directory.
- Do not add a placeholder approval record. Records are created only for a concrete launch after the owner approves the exact value and scope.
