# DRPO update-package preflight v1

This directory contains a bootstrap release gate for DRPO update packages.
It is intentionally placed outside the current trusted control-plane paths so it
can be introduced by an ordinary content package under schema-v3 delta authority.

The gate checks a package before it is handed to `drpo-update`:

- bundle-backed package structure and patch/bundle equivalence;
- `BASE_COMMIT.txt` ancestry against the current repository HEAD;
- schema-v3 handoff-delta base metadata drift (`base.commit`, handoff hash,
  registry hash, and registry after-hash when present);
- dry-run source integration in an isolated worktree;
- trusted handoff normalization through the current repository's normalizer.

It does **not** fast-forward `main`, push, export main bundles, run package test
commands, or run repository test selection.  Those remain the job of
`drpo-update`; this tool is a pre-release gate to catch broken runnable-looking
packages earlier.

Usage:

```bash
python docs/update_packaging_hardening/preflight_update_package \
  --repo /path/to/drpo \
  --package /path/to/update.zip \
  --json
```

Self-test:

```bash
python docs/update_packaging_hardening/preflight_update_package --self-test --json
```
