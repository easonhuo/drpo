# Provenance Note

The formal scientific run used clean commit
`7a70278f3d6061379c81f33e82d93ead86484908` from the complete Git bundle
provided by the user. The bundle recorded complete history and `refs/heads/main` at
that commit. Container DNS could not resolve GitHub, so the hardened guard's
`git ls-remote origin` check was performed against that immutable uploaded bundle,
not against the live GitHub endpoint. Launch and end HEAD matched, the worktree was
clean at both boundaries, and provenance was not marked compromised.

The raw-complete artifact is `D-U1-E6-CONDITIONAL-GAP-01_RAW_COMPLETE.zip`,
SHA-256 `8c64f197e90e945f3a6bf8326c63abd6c4b3118e1c6c8bd614c73af6d1e5be93`. It is immutable recovery/evidence and is not itself a repository
update. The repository stores compact reviewed evidence. Manual terminal review
assigns `finite_step_validated`: all 200 runs and required audits are present, but
151/200 runs remain `formal_persistent_drift_or_inconclusive`, so steady-state or
universal method-ranking claims are not allowed.

The compact repository closure and minimum-change semantic-gap registration were rebased onto current `main` commit `1fa7f04d4830e4d562ab147dbb11dfa8cecc9b5d`; this does not alter the scientific run commit or raw evidence.
