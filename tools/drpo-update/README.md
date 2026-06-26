# DRPO local update helper

This directory is the canonical, repository-backed source for the local
`drpo-update` command. It was imported after the user verified that the
uploaded helper and the installed `~/bin/drpo-update` were byte-identical
(SHA-256 `f344b0cffc163ecdeb80ec8d07b564d00c3538ad22e03887f87bd1ce2a85f4f3`).

## Install once

From the repository root:

```bash
bash tools/drpo-update/install.sh .
```

The default installation is a symlink, so future repository changes to the
helper become active automatically. `--copy` preserves the old copy-based
behavior.

## Use

```bash
drpo-update ~/Downloads/DRPO_*_UPDATE.zip --yes
```

Options:

```text
--no-push           commit locally without pushing
--message TEXT      override the commit message
--version           print the installed helper version
```

## Package compatibility

Legacy packages remain supported and contain:

- `BASE_COMMIT.txt`
- exactly one `*.patch` or `*.diff`
- `CHANGE_SUMMARY.md`
- optional `TEST_COMMANDS.sh`

A Stage 1 package may additionally contain both:

- `change.bundle`
- `PATCH_COMMIT.txt`

The pair is atomic: either both files are present or neither is present.

### Exact-base package

A legacy patch-only package is accepted only when `BASE_COMMIT.txt` equals the
current `main`. Integration and testing still occur in an isolated worktree,
so failures do not dirty `main`.

### Stale but ancestral package

When the package contains the Git bundle pair and the package base is an
ancestor of current `main`, the helper:

1. verifies the bundle;
2. verifies that the patch commit has exactly one parent equal to the package base;
3. proves that `change.bundle` and `update.patch` produce the same Git tree;
4. cherry-picks the patch commit in an isolated worktree;
5. runs package tests there;
6. fast-forwards real `main` only after success and user confirmation.

A real conflict, a non-ancestral base, a patch/bundle mismatch, or a failed test
stops the update without modifying `main`.

## Audit reports

Every run writes an `APPLY_REPORT.json`-style record under:

```text
~/.config/drpo-update/reports/
```

The report records original base, current head, integration mode, patch commit,
tests, conflicts, resulting commit, push state, and failure details.
