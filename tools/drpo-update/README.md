# DRPO local update helper

This directory is the canonical, repository-backed source for the local
`drpo-update` command. It was imported after the user verified that the uploaded
helper and the installed `~/bin/drpo-update` were byte-identical (SHA-256
`f344b0cffc163ecdeb80ec8d07b564d00c3538ad22e03887f87bd1ce2a85f4f3`).

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
--no-push              commit locally without pushing
--message TEXT          override the commit message
--test-mode auto        focused tests unless risk requires the full suite
--test-mode fast        request focused tests; unsafe downgrades are rejected
--test-mode full        force full pytest and Ruff gates
--diagnostic-dir PATH   override the failure ZIP directory (default: ~/Downloads)
--main-bundle-dir PATH  override successful post-push main bundle directory
--no-export-main-bundle disable successful post-push main bundle export
--doctor                run non-destructive transactional self-tests
--version               print the installed helper version
```

## Optional macOS double-click launcher

On macOS, install the thin Finder launcher once:

```bash
bash tools/drpo-update-macos/install.sh
```

Future canonical ZIP packages may use the `.drpoupdate` suffix. Double-clicking
them opens Terminal and delegates to this same `drpo-update` command. The app
does not implement another validation or Git integration path. See
`tools/drpo-update-macos/README.md`.

## Package production and compatibility

### Canonical producer for every new package

New code-update packages must be bundle-backed. Prepare a staging directory with
`BASE_COMMIT.txt`, `update.patch`, `CHANGE_SUMMARY.md`, executable
`TEST_COMMANDS.sh`, and complete after-images under `modified_files/`, then run:

```bash
python3 scripts/package_update.py \
  --repo . \
  --package-root /path/to/staging \
  --output ~/Downloads/DRPO_UPDATE.zip

python3 scripts/verify_update_package.py \
  --repo . \
  --package ~/Downloads/DRPO_UPDATE.zip
```

The producer adds and verifies:

- `change.bundle`
- `PATCH_COMMIT.txt`
- `UPDATE_PACKAGE_MANIFEST.json`

It proves that the patch commit has the package base as its unique parent, that
the Git bundle and patch produce the same tree, and that every non-deleted
changed file and executable mode matches `modified_files/`.

### Legacy consumption compatibility

Historical patch-only exact-base packages remain accepted when they contain
`BASE_COMMIT.txt`, exactly one patch, `CHANGE_SUMMARY.md`, and optional
`TEST_COMMANDS.sh`. This is a consumption-only compatibility path. New packages
must not be manually produced in this format.

A legacy package is accepted only when `BASE_COMMIT.txt` equals current `main`.
Integration and testing still occur in an isolated worktree.

### Bundle-backed integration

When the package contains `change.bundle` and `PATCH_COMMIT.txt`, and the package
base is an ancestor of current `main`, the helper:

1. verifies the bundle;
2. verifies that the patch commit has exactly one parent equal to the package base;
3. proves that `change.bundle` and `update.patch` produce the same Git tree;
4. cherry-picks the patch commit in an isolated worktree;
5. runs package tests and the repository-selected integration gate there;
6. fast-forwards real `main` only after success and user confirmation.

A real conflict, a non-ancestral base, a patch/bundle mismatch, or a failed test
stops the update without modifying `main`.

## Test selection

`tools/drpo-update/test_impact_map.json` is the trusted changed-path map used by
the installed helper. It is loaded from the current real `main`, not from the
candidate worktree, so a package cannot weaken its own gate before being
accepted.

The default `--test-mode auto` policy is:

- low/medium-risk known paths: compile and Ruff-check changed Python files, run
  mapped validators, and run only mapped pytest targets;
- high-risk control/shared code: run full `pytest -q` and `ruff check .`;
- unknown paths: fail closed to the same full suite;
- an explicit `--test-mode fast` may not override a high-risk or unknown-path
  decision;
- `TEST_COMMANDS.sh` is still run first for package-specific checks.

The full gate is aggregated rather than fail-fast. It attempts compile checks,
shell syntax checks, governance validators, full pytest, and full Ruff even when
an earlier independent command fails. Each command receives its own complete
log, and the helper reports the combined failure set once all gates finish.

The standalone inspection command is:

```bash
python3 scripts/select_update_tests.py \
  --repo . --base <OLD_SHA> --head <NEW_SHA> --json
```

Use `--execute` to run the selected plan.

## Audit reports and timings

Every run writes an `APPLY_REPORT.json`-style record under:

```text
~/.config/drpo-update/reports/
```

The report records original base, current head, integration mode, patch commit,
package tests, selected test mode, matched impact groups, unknown paths,
conflicts, resulting commit, push state, failure details, and per-phase timing.
The terminal prints only a compact timing summary; the complete structured
record remains in the report file.

## Automatic failure diagnostic ZIP

Any package validation, stale-base, merge-conflict, package-test, repository
gate, review, or push failure creates one upload-ready archive by default at:

```text
~/Downloads/DRPO_DIAGNOSTIC_<HEAD>_<TIMESTAMP>_<ID>.zip
```

The location can be overridden with `--diagnostic-dir` or the
`DRPO_UPDATE_DIAGNOSTIC_DIR` environment variable. The archive contains:

- the original update package;
- the structured apply report;
- complete per-command package and repository-gate logs;
- a Git repository bundle with current/base/patch/candidate refs when available;
- candidate identity, changed files, and a current-to-candidate patch;
- base/ours/theirs/worktree copies for each merge conflict;
- Git status, refs, recent log, remotes, system information, and installed package
  versions;
- a SHA-256 diagnostic manifest.

The real `main` remains unchanged for package, integration, or test failures.
The diagnostic ZIP is created before the isolated worktree is removed, so
conflict stages and failed candidate state are preserved without extra user
commands.

## Exact-base preflight

Before applying a patch or running package tests, `drpo-update` fetches
`origin/main` and checks, in order:

1. the current branch is `main`;
2. the worktree is clean;
3. local `HEAD` equals `origin/main`;
4. package `BASE_COMMIT.txt` equals that synchronized HEAD.

Failures print `DRPO_UPDATE_PREFLIGHT_FAILED` immediately, followed by a stable
reason code, current values, dirty paths when applicable, and manual repair
commands. The same structured fields are retained in the apply report and
diagnostic ZIP. The helper never switches branches, stashes, discards local
changes, or rebases an outdated package automatically.

## Successful push main-bundle export

After tests pass, local `main` advances, and push succeeds, the helper runs
`git fetch origin main` and requires local `HEAD` to equal `origin/main`. It
then creates and verifies temporary bundles before atomically publishing these
files to `~/Downloads` by default:

```text
DRPO_MAIN_<12-char-SHA>.bundle
DRPO_MAIN_<12-char-SHA>.bundle.sha256
DRPO_MAIN_LATEST.bundle
DRPO_MAIN_LATEST.bundle.sha256
```

Use `--main-bundle-dir PATH` or `DRPO_UPDATE_MAIN_BUNDLE_DIR` to change the
directory. `--no-push` never creates an official main bundle. Use
`--no-export-main-bundle` to suppress export after a verified push.

If push succeeds but export fails, the pushed commit is not rolled back. The
command exits nonzero, prints `UPDATE_PUSHED_BUNDLE_FAILED`, and creates the
normal failure diagnostic ZIP.

## Local doctor

Run:

```bash
drpo-update --doctor
```

The doctor compiles the updater/producer, checks Shell syntax, and runs the
synthetic transactional tests for exact/stale bundle integration, conflict
protection, diagnostics, canonical packaging, and main-bundle export. It does
not mutate or push the real repository.


## Recovery experiment artifacts are evidence-only

`experiment-checkpoint`, `experiment-failed`, and `experiment-raw-complete`
packages preserve immutable run evidence before repository closure. Their
`update.patch` may intentionally be empty, so they must not be passed to
`drpo-update`. Preserve them for audit. After terminal audit and scientific
interpretation, update `docs/handoff.md`, `experiments/registry.yaml`, and the
compact result files, then build an `experiment-final` repository-closure
package. The updater detects these package kinds before patch validation and
reports this workflow explicitly.
