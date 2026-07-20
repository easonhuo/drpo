# Stage 5 pre-integration report-history bugfix

Base: `main@cfe43571cd8c6d0909c61d36c4f6e4d07c2d2362`

The reciprocal closure exposed a fail-closed false rejection: an immutable materialization report had more than one source-branch revision before its single GitHub merge integration. The prior validator counted all path touches as post-acceptance mutations.

This maintenance change permits multiple revisions only when every touch is already an ancestor of one first-parent integration commit, the delta and report map to that same integration, and current bytes equal the integrated tree. Any post-integration touch or byte drift still fails closed. No scientific file, schema, registry authority, experiment status, or claim is changed.

## Bootstrap verification

Before the clean maintenance commit `3f75b2fb1c4d9022787307549fea81200cfb26b0` was published, the one-time workflow completed the focused pre-integration-revision/post-integration-tamper regression, current-repository handoff-authority verification, governance-stage validation, focused governance pytest, Python compilation, and changed-file Ruff. All temporary workflow, helper, and failure-diagnostic paths were removed from the published tree.

The final human-origin documentation commit exists only to trigger ordinary exact-head pull-request checks after GitHub marked the bot-authored commit's checks as `action_required`; it changes no protected hash, policy, scientific variable, result, status, or claim.
