# Formal Experiment Supervision and Durable Artifact Protocol

**Governance IDs:** `GOV-EXP-ARTIFACT-01`, `GOV-EXP-ARTIFACT-02`, `GOV-EXP-ARTIFACT-03`
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

This internal verification is mandatory; a separate verifier remains available for independent re-checking. The patch apply gate uses a temporary Git index loaded from `BASE_COMMIT.txt`, so staged packaging edits do not create a false conflict and the caller's real index is never mutated.

## 13. Large-result and sidecar policy

The default main ZIP hard limit is 25 MiB and the default single-file main-package limit is 10 MiB. These are gates, not warnings.

Failed, checkpoint, and raw-complete packages use lightweight evidence mode by default. They retain manifests, commands, logs, tracebacks, compact metrics, completed/missing-unit inventories, source provenance, and checksums. Large checkpoint-like files are excluded from the main ZIP and recorded in `LARGE_FILE_INDEX.json` with path, role, byte size, SHA-256, inclusion decision, and persistence status.

Real weights and checkpoints remain on persistent training-server storage by default. The main package records their storage path, byte size, SHA-256, role, and persistence status. Sidecar delivery is disabled by default. A sidecar may be created only for explicitly selected files whose cross-machine transfer, restart, or independent audit requirement was pre-registered. The sidecar has independent file-count and size limits and may never silently sweep all checkpoint-like files.

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
2. a verified Git bundle/source capsule that contains the expected commit object and complete tree.

A plain GitHub source archive may support static inspection, but it does not contain Git commit objects and therefore does not by itself satisfy the commit-bound formal-run or merge-equivalent pre-delivery gate. A repository reconstructed from parsed web pages, copied snippets, manually recreated files, or an unrelated local commit is not a valid base, even when `BASE_COMMIT.txt` contains the desired SHA. A package generated from such a synthetic base must not be described as verified.

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
  --output-root experiments/results/EXT-C-E8-V4.1/run_001 \
  --artifact-output artifacts/EXT-C-E8-V4.1_FAILED_EVIDENCE.zip \
  --large-file-persistence persistent_local \
  --sidecar-output artifacts/EXT-C-E8-V4.1_CHECKPOINT_SIDECAR_run001.zip \
  --sidecar-purpose restart \
  --sidecar-file method/latest/adapter_model.safetensors \
  -- python3 src/drpo/countdown_qwen_arena_onefile.py
```

## 18. Transactional replacement and launch-failure recovery (`GOV-EXP-ARTIFACT-03`)

A previously verified artifact is immutable until a replacement candidate has passed every gate. The packager must never unlink the final output before verification. Candidate failure removes only candidate files and leaves the prior final package untouched. Explicit sidecars use a new versioned path. If sidecar publication succeeds but the subsequent main-package replace fails, the newly published sidecar is removed and the prior main package remains untouched.

A supervised attempt must use a new or empty `--output-root`. Reusing a non-empty run directory is rejected before process launch because stale files could satisfy `--required-output`, contaminate logs, or be packaged as if they came from the new attempt. A resumed run reads its input checkpoint from a separately declared persistent path and writes to a fresh run directory.

The foreground supervisor treats command-start failure as an experiment failure, not as an uncaught tool crash. Missing executables, permission errors, invalid working directories, and equivalent `Popen` failures must produce:

- `RUN_FAILED.json`;
- `run_manifest.json`;
- a traceback-bearing log under `logs/`;
- launch commit and end-state provenance;
- an attempted lightweight `experiment-failed` package.

## 19. Launch-commit binding for recovery packages

A recovery artifact describes the code that actually launched. Its `BASE_COMMIT.txt`, manifest `base_commit`, and any source snapshot therefore use the launch commit even when the repository HEAD changes before packaging. The packaging-time HEAD is recorded separately. Recovery packages do not contain an update patch assembled from the contaminated end-state worktree. Source files are read from the launch commit with Git object access.

## 20. Final-evidence completeness gate

Before `experiment-final` publication, the result directory must contain:

- parseable `RUN_COMPLETE.json`;
- parseable `run_manifest.json`;
- parseable `TERMINAL_AUDIT.json` or `terminal_audit.json`;
- at least one regular log file under `logs/`.

The experiment ID and base commit in all three identity-bearing JSON records must match the package request. These checks are repeated against the finished ZIP. Raw metrics and summaries remain experiment-specific, but omission of the generic provenance and audit evidence is a hard failure.

## 21. Persistent-local checkpoint default and explicit sidecars

The generic policy applies to Countdown, Hopper, recommendation, and every future large-file-producing experiment:

1. foundation-model weights are never copied;
2. real adapters/checkpoints stay on persistent training-server storage by default;
3. the main ZIP contains only the large-file index with storage path, size, SHA-256, role, and persistence status;
4. sidecar generation is off by default;
5. `--sidecar-output` requires one or more explicit `--sidecar-file` selections and an explicit `--sidecar-purpose` (`cross_machine_transfer`, `restart`, or `independent_audit`);
6. sidecar filenames are versioned and must not already exist; overwriting a previous sidecar is rejected;
7. sidecar file-count and total-size limits are hard gates;
8. optimizer state is excluded unless a registered recovery requirement explicitly needs it;
9. foundation-model weights are never copied to either the main ZIP or a sidecar;
10. checkpoint/model-state files are indexed rather than embedded in the main ZIP even when they are below the generic single-file size threshold.

A path on an ephemeral container is not `persistent_local`. The run manifest must use `ephemeral` or `unknown` unless the storage survives the runtime and remains accessible to the project.

## 22. Formal source-availability preflight

The strict exact-source rule explains why an environment may browse GitHub yet still be unable to launch a formal experiment: browser rendering is not a Git checkout and cannot prove the complete import/configuration closure. Earlier sessions sometimes ran directly reconstructed one-file code before this gate existed; the new rule prevents repeating that weaker provenance path and does not by itself erase previously accepted evidence.

Before proposing a formal run, check source availability in this order:

1. an existing clean Git checkout containing the expected full SHA;
2. shell-accessible clone/fetch;
3. an environment-provided download bridge for a full-SHA Git bundle or a verified source capsule that includes the expected commit object and complete tree;
4. a project-persistent exact-SHA bundle/capsule already delivered by a previous trusted workflow.

Browser access and shell access are separate capabilities: seeing a GitHub page does not place executable files or Git objects in the shell. A plain source ZIP can be used for read-only review, but it is not sufficient for the formal guard. Exhaust all automated acquisition paths before involving the user. If none succeeds, stop at preflight and, only as a last resort, request one complete Git bundle or verified source capsule; never request arbitrary individual files or describe a plain Source code ZIP as a formal checkout. Browser-only inspection remains allowed for review, planning, and static analysis, but not for a new commit-bound formal result. This source gate must not be weakened to solve a network/tooling limitation.
User upload is therefore optional rather than preferred: it is unnecessary whenever an existing checkout, clone/fetch, environment bridge, or project-persistent bundle/capsule supplies the exact commit object and complete tree. A no-upload formal run is valid after the same commit, cleanliness, and source-file preflights pass.

The formal guard accepts exactly two source-identity paths:

1. pass an explicit full `--expected-commit` that exists as a commit object in the local checkout, which is the normal offline/server path; or
2. omit `--expected-commit` only when a live authoritative `git ls-remote origin refs/heads/main` succeeds and matches local `HEAD`.

An offline checkout must therefore use the explicit full SHA. The guard never silently promotes an arbitrary local `HEAD` into a formal source identity. Any requested `--source-file` is also checked against the launch commit before process start, so a typo or missing committed entry point fails before compute begins.

## 23. Defensive identity, path, and mutation gates

The following controls apply to every package and supervised run:

1. experiment IDs must match the restricted identifier grammar and may not contain path separators or traversal;
2. result roots, output roots, artifact paths, and sidecar paths may not be symlinks or pass through symlink components;
3. runtime output/artifact paths inside the repository may not overlap tracked files, because ignored runtime paths must never hide source changes;
4. `run_manifest.json` and the relevant failed/raw-complete/final marker must agree on experiment ID and launch commit for every result package kind;
5. small `.npy`/`.npz` raw evidence is embedded when it is below the normal file-size limit, while model/checkpoint/optimizer state remains index-only regardless of size;
6. files are hashed during scan and rechecked after copy; concurrent mutation invalidates the candidate;
7. the verifier rejects malformed JSON-object manifests, unknown package kinds, unsafe or duplicate paths, undeclared `modified_files/` members, and malformed large-file indexes;
8. a stale process receives one SIGTERM and then SIGKILL after `--termination-grace-seconds` if it remains alive;
9. failures during signal-handler setup, log-reader startup, monitoring, or end-provenance resolution use the same failed-evidence path as child-process failures.
10. sidecar verification checks the exact manifest-to-payload inventory, including experiment ID, full base commit, declared purpose, canonical path, size, and SHA-256 for every selected member;
11. all size and sidecar-count limits must be positive, and the generic large-file persistence default is `persistent_local`; ephemeral runtimes must explicitly set `ephemeral` or `unknown`.
