# Manuscript Writing Skill Library

This directory contains the shared writing-skill layer for the manuscript system.
It is deliberately split from both the batch manuscript pipeline and the project
profile:

```text
core paper writing skills  -> cross-project writing rules
domain skills              -> field-level research-writing rules
project profile            -> project facts, terminology, evidence status, and claim boundaries
```

The batch pipeline and the interactive editor should load this directory rather
than copying temporary prompt rules into separate implementations.

## First-version boundary

The first version is intentionally light-weight:

- machine-readable skill schema;
- core and RL/ML domain skill YAML files;
- task router mapping writing tasks to the minimum necessary skills;
- interactive editing protocol;
- report-only / packet-only pipeline obligations;
- tests for schema validity, router references, and project-term leakage.

It does **not** implement automatic experience mining, a complex task classifier,
a full manuscript rewrite, or changes to scientific evidence status.

## Rollback posture

The default pipeline integration is report-only / packet-only.  It records skill
obligations in generation packets and validation reports but does not change the
existing prose-generation algorithm.  The integration can be disabled with the
pipeline's `--disable-skill-library` option, or by removing the loader/router
connection while leaving these documents in place.
