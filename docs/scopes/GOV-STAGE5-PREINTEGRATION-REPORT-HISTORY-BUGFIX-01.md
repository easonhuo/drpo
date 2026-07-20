# Stage 5 pre-integration report-history bugfix

Base: `main@cfe43571cd8c6d0909c61d36c4f6e4d07c2d2362`

The reciprocal closure exposed a fail-closed false rejection: an immutable materialization report had more than one source-branch revision before its single GitHub merge integration. The prior validator counted all path touches as post-acceptance mutations.

This maintenance change permits multiple revisions only when every touch is already an ancestor of one first-parent integration commit, the delta and report map to that same integration, and current bytes equal the integrated tree. Any post-integration touch or byte drift still fails closed. No scientific file, schema, registry authority, experiment status, or claim is changed.
