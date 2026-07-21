# GOV-DRPO-MAINTENANCE-RUNNER-REPLAYAB-01 — Stage 1/2 Scope

- Base: `main@d3f7d046f948108a3d837bdcff617eed5146a2f0`
- Scientific impact: none
- User authorization: continuously complete Stage 1 and Stage 2; merge only after the frozen final gate permits it.

Authorized:

- extend existing `scripts/run_workflow_replay.py` with one local `git-object-pair` evidence producer;
- extend existing ReplayAB tests;
- add non-Python E7/E8 scenario packets and Stage 1/2 evidence;
- create one Draft PR from an exact current-main atomic Git-object commit;
- run exact-head CI, controlled local ReplayAB, and final acceptance;
- merge only for `ADOPT_M0` or confirmed `NARROW_M0`.

Not authorized:

- new Python files, workflows, dependencies, services, publishers, backends, or E7/E8 adapters;
- handoff, registry, governance-ledger, authority, formal-execution, or scientific-code changes;
- M1, M2, default-route activation, experiment launch, force push, or automatic repair;
- merging a failed, stale, unreviewed, or non-exact PR head.

Executable budget:

- preferred: at most 100 changed executable lines;
- 101–140: yellow-zone review required;
- above 140: stop and redesign.

Rollback before default adoption is to stop using the packet procedure. No production component needs to be disabled.
