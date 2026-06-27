# Provenance Note

The formal scientific run was launched from clean `main` commit
`eb5e12626026854f44f4698dbc8ed8829e74e0b0`. The run manifest records an
authoritative `origin/main` match at launch and no launch/end worktree changes.
The raw-complete artifact is `D-U1-E6-SEMANTIC-LONGRUN-01_RAW_COMPLETE.zip`,
SHA-256 `e098d4dd0483a661468db0cb1c4b67e4e563e2426a6aa078fe7b808f7ac691fa`.

That raw-complete ZIP is immutable recovery/evidence, not a code update. Its empty
`update.patch` is intentional under the artifact protocol. Repository closure is
performed separately from current `main` base
`a1672d95653139964debdd5c1baf00173722c071` by updating the handoff, registry,
compact outputs, updater diagnostics, and tests. The full trajectories and logs
remain in the raw artifact; the repository stores compact, hash-indexed evidence.
