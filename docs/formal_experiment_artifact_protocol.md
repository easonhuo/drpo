# Formal Experiment Supervision and Durable Artifact Protocol

**Governance IDs:** `GOV-EXP-ARTIFACT-01`, `GOV-EXP-ARTIFACT-02`
**Scope:** every formal DRPO / SNA2C experiment, including C-U1, D-U1, Hopper/D4RL, Countdown/Transformer, recommendation, and future registered environments.
**Authority:** `AGENTS.md` and Section 0 of `docs/handoff.md` override this operational detail if they conflict.

## 1. Why this protocol exists

A process finishing in an ephemeral runtime is not the same as a result being preserved. Files written only to a temporary container can disappear when the runtime is recycled. Chat messages, process counters, and statements such as `20/20 completed` are not experiment evidence.

A formal experiment is complete only after all of the following are true:

1. the registered computation finished or was explicitly classified as failed;
2. raw outputs and logs were audited;
3. required terminal-state checks were completed;
4. `docs/handoff.md` and `experiments/registry.yaml` were updated when the scientific status changed;
5. a durable downloadable artifact was generated and verified;
6. the artifact was delivered to the user or stored on a persistent system;
7. repository application is reported separately and only after an actual commit or push.

## 2. Two independent status axes

The scientific result status remains one of the statuses allowed by `AGENTS.md`:

- `analytically_proven`
- `long_run_validated`
- `finite_step_validated`
- `pilot`
- `not_run`
- `rejected`
- `superseded`

Execution and evidence use a separate lifecycle:

1. `registered`
2. `running`
3. `raw_complete`
4. `terminal_audited`
5. `packaged`
6. `delivered`
7. `applied_to_repository`

`raw_complete` does not authorize the phrase “formal experiment completed.” The minimum completion claim is `packaged + delivered`. Repository closure additionally requires `applied_to_repository`.

## 3. Supervision rules for ephemeral runtimes

A formal experiment in an ephemeral runtime must run through `scripts/run_experiment_guard_hardened.py` or an equivalent foreground supervisor with the same guarantees.

The supervisor must:

- remain attached to the experiment until process exit;
- write a heartbeat containing UTC time, PID, elapsed time, output activity, and completed-unit counters when available;
- stream stdout and stderr to both the console and a log file;
- record the command, repository SHA, environment metadata, start time, and exit code;
- detect nonzero exit, signal termination, stale output, and missing required outputs;
- preserve partial results and create `RUN_FAILED.json` on failure;
- create `RUN_RAW_COMPLETE.json` on successful computation before scientific audit;
- generate a durable recovery package immediately on success or failure when `--artifact-output` is supplied.

Starting a background process and ending the working turn without a supervisor is prohibited for formal experiments.

## 4. Stage boundaries

Each registered experiment ID is a persistence boundary.

- E3 must be packaged and delivered before E4 begins.
- A completed external-validity probe must be packaged before the downstream method comparison begins.
- For a run expected to exceed 30 minutes, create a checkpoint artifact at least every five formal seeds or at another interval explicitly registered before launch.
- A failed final plotting, aggregation, or reporting step must not erase or invalidate already written raw trajectories. The partial run must be packaged before repair or rerun.

## 5. Required durable package types

### 5.1 Checkpoint package

Purpose: interruption recovery. It may be scientifically incomplete.

Required labels:

- `package_kind: experiment-checkpoint`
- completed and pending seeds or units
- latest heartbeat
- source commit and command
- partial raw outputs and logs

### 5.2 Failed-run package

Purpose: preserve evidence and diagnose failure.

Required labels:

- `package_kind: experiment-failed`
- `RUN_FAILED.json`
- exit code or signal
- traceback or error log
- completed and missing outputs
- partial raw outputs

### 5.3 Raw-complete package

Purpose: preserve computation immediately, before interpretation and handoff editing.

Required labels:

- `package_kind: experiment-raw-complete`
- `RUN_RAW_COMPLETE.json`
- raw outputs, trajectories, summaries, logs, source snapshot, and manifest
- explicit statement that terminal audit or scientific acceptance may still be pending

### 5.4 Final experiment update package

Purpose: durable scientific result and repository update. It must be compatible with the local `drpo-update` workflow.

Required top-level files:

- `update.patch`
- `BASE_COMMIT.txt`
- `CHANGE_SUMMARY.md`
- `TEST_COMMANDS.sh`
- `modified_files/`
- `ARTIFACT_MANIFEST.json`
- `SHA256SUMS.txt`

Required result content:

- raw per-seed summaries;
- trajectories or raw curves;
- aggregate CSV/JSON;
- report and figures;
- run manifest;
- `RUN_COMPLETE.json`;
- `TERMINAL_AUDIT.json` or `terminal_audit.json`;
- logs and failed-run index;
- source snapshot or source-file hashes.

`BASE_COMMIT.txt` contains exactly one full 40-character Git SHA and a trailing newline.

### 5.5 Governance/code-only update package

Purpose: change rules or code without claiming a new experiment result. It uses the same `drpo-update` top-level structure, but does not require result markers.

## 6. Result reporting separation

Every terminal audit must separately report:

1. task-performance failure or collapse;
2. support, entropy, variance, or structure-coverage boundary events;
3. NaN/Inf or other nonfinite numerical failure.

A support-boundary event is not automatically a numerical crash. A low finite reward is not automatically NaN/Inf collapse.

## 7. Handoff and registry updates

After scientific audit and before final packaging:

- update the newest increment record in `docs/handoff.md` without deleting prior claims;
- record old statement, problem, new evidence, and replacement conclusion when correcting a result;
- update `experiments/registry.yaml` execution and evidence fields;
- keep the scientific status separate from execution lifecycle;
- record the artifact filename and SHA-256 after package generation;
- do not record `applied_commit` until the repository change actually exists.

If raw computation finished but the final artifact has not been delivered, record:

`raw_complete, not durably delivered`

and do not call the experiment complete.

## 8. Artifact size policy

The default warning threshold is 25 MiB per final experiment package. Size reduction must not delete essential evidence.

Prefer:

- compressed CSV/JSON/JSONL;
- evaluation-point trajectories rather than every minibatch tensor;
- source hashes rather than duplicate large source trees when the exact commit is public;
- one necessary checkpoint rather than optimizer state at every step.

Do not include large foundation-model checkpoints, D4RL datasets, or redundant optimizer states unless a registered recovery need requires them. Large external files should be referenced by immutable checksum and persistent location.

## 9. Canonical commands

Run a supervised formal process:

```bash
python3 scripts/run_experiment_guard_hardened.py \
  --experiment-id C-U1-E3 \
  --repo-root . \
  --output-root experiments/results/C-U1-E3/run_001 \
  --artifact-output artifacts/C-U1-E3_RAW_COMPLETE.zip \
  -- python3 src/drpo/drpo_cu1_e1_e4_oneclick.py
```

Build a final update artifact after terminal audit and handoff/registry edits:

```bash
python3 scripts/package_experiment_hardened.py \
  --repo-root . \
  --experiment-id C-U1-E3 \
  --package-kind experiment-final \
  --result-dir experiments/results/C-U1-E3/run_001 \
  --output artifacts/DRPO_CU1_E3_RESULTS_AND_UPDATE.zip \
  --summary-file experiments/results/C-U1-E3/run_001/REPORT.md \
  --test-command "python3 -m pytest -q tests/test_experiment_artifact_protocol.py"
```

Verify before delivery:

```bash
python3 scripts/verify_experiment_package_hardened.py \
  artifacts/DRPO_CU1_E3_RESULTS_AND_UPDATE.zip \
  --repo-root .
```

## 10. Completion language

Allowed statements:

- “The process is running; heartbeat is current.”
- “Raw computation finished; terminal audit is pending.”
- “The final package was verified and delivered.”
- “The package was applied and committed as `<SHA>`.”

Prohibited statements without corresponding evidence:

- “The result is safe because it was written to `/mnt/data`.”
- “The experiment is complete” when no durable package exists.
- “The repository was updated” when no commit or push succeeded.

## 11. Commit identity and formal-run provenance (`GOV-EXP-ARTIFACT-02`)

A web commits page, search result, or cached repository view is not authoritative for the current `main` SHA. Resolve commit identity in this order:

1. `git ls-remote --exit-code origin refs/heads/main`;
2. `git rev-parse HEAD` in the checked-out repository;
3. an explicitly supplied full expected SHA.

The local HEAD and expected SHA must match. If an authoritative remote query succeeds, it must also match. If remote resolution is unavailable, record the failure and do not claim that remote `main` was independently verified.

Formal runs require a clean worktree at launch. Editing may continue freely during development, but each formal attempt or rerun must start from a committed snapshot. Dirty execution is allowed only for an explicitly labelled pilot with `--allow-dirty`; before launch, capture tracked and staged binary patches, bounded untracked source/config files, and hashes.

At process exit, re-check HEAD and worktree status. A changed HEAD or dirty formal worktree sets `provenance_compromised: true` and forces failed-run packaging even when the child process exits successfully.

## 12. Atomic candidate verification

`package_experiment_hardened.py` must not publish directly to the requested final filename. It must:

1. build a candidate ZIP;
2. verify safe member paths, required files, checksums, result markers, base SHA, and `git apply --check`;
3. enforce the hard main-package size limit;
4. delete the candidate on any failure;
5. atomically rename the candidate only after all checks pass.

This internal verification is mandatory; a separate verifier remains available for independent re-checking.

## 13. Large-result and sidecar policy

The default main ZIP hard limit is 25 MiB and the default single-file main-package limit is 10 MiB. These are gates, not warnings.

Failed, checkpoint, and raw-complete packages use lightweight evidence mode by default. They retain manifests, commands, logs, tracebacks, compact metrics, completed/missing-unit inventories, source provenance, and checksums. Large checkpoint-like files are excluded from the main ZIP and recorded in `LARGE_FILE_INDEX.json` with path, role, byte size, SHA-256, inclusion decision, and persistence status.

When a checkpoint is required for continuation or audit, deliver it as a separate sidecar. The main manifest records the sidecar filename, SHA-256, size, and status. A final experiment package that declares a required large file may not claim complete durable delivery until the sidecar is also durable.

Formal large-model experiments must pre-register:

- main artifact budget;
- maximum checkpoint count;
- best/latest retention policy;
- optimizer-state policy;
- sidecar requirement.

For Countdown, the default is at most two retained checkpoints per method (`best` and `latest`), no copied foundation-model weights, and no optimizer state unless a registered recovery requirement overrides it.

## 14. Symbolic-link policy

Packaging must never follow symbolic links. An external symlink can silently import a model cache, dataset, secret, or many gigabytes of unrelated files; it is therefore rejected. An internal symlink is recorded as a reference, and the real target is packaged no more than once. Broken symlinks are rejected.

## 15. Fail-closed entry points and pre-delivery compatibility gate

The canonical scripts `run_experiment_guard_hardened.py`, `package_experiment_hardened.py`, and `verify_experiment_package_hardened.py` must import the same hardened implementation. If the shared implementation is absent, each entry point exits nonzero with an actionable error. A missing module must never trigger a legacy fallback, because mixed producer/verifier versions can create packages that pass one environment and fail another.

Before a ChatGPT-generated update ZIP is delivered, it must be validated in a fresh clean checkout reconstructed from the confirmed base commit:

1. preserve the base repository's Git file modes, especially executable scripts;
2. run `git apply --check update.patch`;
3. apply the patch and confirm that only declared files changed;
4. run the ZIP's own `TEST_COMMANDS.sh`;
5. run `git diff --check` and reject whitespace errors;
6. independently verify ZIP structure and SHA-256 inventory;
7. publish the downloadable ZIP only after all steps pass.

Local `drpo-update` tests remain the final environment-specific safety check, not the first execution of the compatibility suite.

## 16. Exact-base acquisition and merge-equivalent pre-delivery gate

The base repository used to generate and test an update package must be obtained from an authoritative source pinned to the full SHA:

1. a real `git clone`/`git fetch` followed by `git checkout <full-sha>`; or
2. the GitHub source archive for that exact full SHA.

A repository reconstructed from parsed web pages, copied snippets, manually recreated files, or an unrelated local commit is not a valid base, even when `BASE_COMMIT.txt` contains the desired SHA. A package generated from such a synthetic base must not be described as verified.

Before publishing the downloadable ZIP, perform a merge-equivalent test in a second fresh checkout of the same authoritative base:

1. confirm `git rev-parse HEAD` equals `BASE_COMMIT.txt`;
2. confirm the worktree is clean;
3. run `git apply --check update.patch`;
4. apply `update.patch`;
5. run the package's own `TEST_COMMANDS.sh`;
6. run `git diff --check`;
7. compare the changed paths and executable modes with `ARTIFACT_MANIFEST.json` and `modified_files/`;
8. independently verify ZIP checksums and required members.

If authoritative source acquisition or any merge-equivalent step is unavailable, the package must be withheld rather than delegated to the user's machine as the first real compatibility test.

## 17. Canonical hardened commands

Resolve the commit before a formal run:

```bash
python3 scripts/resolve_main_commit.py \
  --repo-root . \
  --expected-sha "$(git rev-parse HEAD)"
```

Run a formal experiment from a clean commit:

```bash
python3 scripts/run_experiment_guard_hardened.py \
  --run-class formal \
  --expected-commit "$(git rev-parse HEAD)" \
  --experiment-id C-U1-E3 \
  --repo-root . \
  --output-root experiments/results/C-U1-E3/run_001 \
  --artifact-output artifacts/C-U1-E3_RAW_COMPLETE.zip \
  -- python3 src/drpo/drpo_cu1_e1_e4_oneclick.py
```

For a large-model run that needs a recovery checkpoint, add a sidecar:

```bash
python3 scripts/run_experiment_guard_hardened.py \
  --run-class formal \
  --expected-commit "$(git rev-parse HEAD)" \
  --experiment-id EXT-C-E8-V4 \
  --repo-root . \
  --output-root experiments/results/EXT-C-E8-V4/run_001 \
  --artifact-output artifacts/EXT-C-E8-V4_FAILED_EVIDENCE.zip \
  --sidecar-output artifacts/EXT-C-E8-V4_CHECKPOINT_SIDECAR.zip \
  -- python3 src/drpo/countdown_qwen_arena_onefile.py
```
