# Stage 3 / Stage 5 authoritative-history compatibility maintenance

Base: `main@8a21fd16e63ea96a0cde2a477353d24562624655`

This maintenance repair fixes a governance compatibility mismatch exposed while landing the E8 wide-right-tail result closure. Stage 5 already accepts bounded pre-integration side-branch history and content-addresses one historical inert schema-v3 anomaly; Stage 3 still assumed a simpler history and attempted to require a materialization report for every schema-v3 delta.

The repair makes Stage 3 use the same bounded first-parent integration semantics and inherit only the exact `legacy_inert_update_ids` returned by the Stage 5 verifier. Every other missing materialization report, post-integration touch, ambiguous integration, or byte drift remains fail-closed. No handoff content, registry state, scientific implementation, experiment variable, seed, threshold, result, or scientific claim is changed by this maintenance update.

Because the canonical selector lints the complete modified Stage 3 file, this same maintenance after-image also normalizes the pre-existing Ruff findings exposed by that selector. Those edits are mechanical only and do not change Stage 3 or Stage 5 semantics.

A fresh Full Acceptance is required and persisted because both scheduled triggers were overdue and this is a critical semantic mismatch repair.
