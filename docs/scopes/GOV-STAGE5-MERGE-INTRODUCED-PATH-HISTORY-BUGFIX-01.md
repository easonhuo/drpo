# Stage 5 merge-introduced path-history bugfix

Base: `main@3e00ca8c2802724d509124541b7ca5a4de1eb90c`

The completed E8 backlog transaction introduced its materialization report in a two-parent source-merge result and then integrated that tree into first-parent `main`. Git records the source merge as the oldest path touch but reports no standalone `--diff-filter=A` commit because the path is absent from both source parents.

This maintenance fix accepts only that bounded history shape: exactly one oldest two-parent merge touch, no path on its first parent, one later first-parent integration, identical origin/integration/current bytes, and no post-integration touch. Multiple additions, non-merge origins, octopus merges, paths already present on the source first parent, byte drift, and post-integration mutation remain rejected. No handoff content, registry state, scientific variable, result, or claim changes.
