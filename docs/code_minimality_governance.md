# Code Minimality Governance

This document records the implementation guidance for `GOV-CODE-MIN-DIFF-01`.
It is an engineering-delivery note, not a second DRPO research master. The
research master remains `docs/handoff.md`; experiment status, terminology,
execution order, and handoff/registry authority continue to follow `AGENTS.md`,
`docs/handoff.md`, and the Stage 5 delta workflow.

## Purpose

Bug-intent requests should not expand into broad refactors, stale baseline
repairs, unrelated test modernization, or mixed experiment-protocol changes. A
user may describe a bug as small, but the assistant must inspect the repository
and determine the smallest sufficient root-cause closure rather than asking the
user to predict the number of files or lines.

The governing rule is: change size is determined by the verified root-cause
closure, not by the user's guess about line count. Larger changes are permitted
only when they are necessary, explained, and reclassified before delivery.

## Request classification

A bug-intent request is a concrete report of a bug, failing package, failing
test, or narrow repair request such as "just fix this". Such a request defaults
to Minimal Sufficient Diff mode.

Before delivery, classify the repair path:

- Green: the root cause is local, no protected files are touched, and the patch
  stays within the direct closure.
- Yellow: the bug is narrow but the closure spans multiple files, fixtures, or
  entry points. Explain the closure and the tests before delivery.
- Red: the repair touches protected governance, handoff/registry authority,
  experiment protocol, seeds, thresholds, formal results, or package authority.
  Reclassify the task before editing.
- Split: the current baseline is already red or the blocking failure is unrelated
  to the requested patch. Separate baseline repair from the requested bug fix
  unless the user explicitly authorizes a combined governance repair.

## Modified-file responsibility

Every modified file in a bug-intent package must have one of the following
roles:

- root-cause fix;
- direct caller compatibility;
- minimal regression test;
- directly tied fixture or package metadata.

The following do not belong in an ordinary bug-intent package: opportunistic
refactors, unrelated cleanup, experiment protocol changes, registry/handoff
updates, generated-shadow refreshes, paper or result updates, and unrelated
baseline repairs.

## First-failure rule

After the first failed package, CI run, or `drpo-update` attempt, stop and
classify the failure before producing another package. Use at least these
classes: integration conflict, package scope error, baseline gate failure,
package test failure, repository gate failure, protocol/package-format failure,
base freshness failure, artifact/provenance failure, and environment/dependency
failure.

If a package fails because it is a context or planning package rather than a
`drpo-update` package, classify it as `protocol/package-format failure`. Do not
retrofit fake `BASE_COMMIT.txt`, `update.patch`, `change.bundle`, or
`PATCH_COMMIT.txt` fields into planning material. Instead, generate a separate
canonical bundle-backed update package from a verified current base when a real
repository patch is intended.

## Relationship to DRPO science governance

This rule does not change scientific experiment IDs, result status, seeds,
thresholds, stopping criteria, or the distinction among C-U1, D-U1, Hopper,
Countdown, and historical mechanism environments. It only governs how narrow
engineering fixes are triaged, scoped, packaged, and reported.

## Control-plane note

`AGENTS.md` is a trusted control-plane file in the current updater workflow. A
normal content package must not modify it directly. If these rules should later
be promoted into `AGENTS.md`, do that as a separate control-plane/governance
update through the appropriate authorized path.
