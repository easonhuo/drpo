# Formal Experiment Supervision and Durable Artifact Protocol

**Governance ID:** `GOV-EXP-ARTIFACT-01`
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

A formal experiment in an ephemeral runtime must run through `scripts/run_experiment_guard.py` or an equivalent foreground supervisor with the same guarantees.

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
python3 scripts/run_experiment_guard.py \
  --experiment-id C-U1-E3 \
  --repo-root . \
  --output-root experiments/results/C-U1-E3/run_001 \
  --artifact-output artifacts/C-U1-E3_RAW_COMPLETE.zip \
  -- python3 src/drpo/drpo_cu1_e1_e4_oneclick.py
```

Build a final update artifact after terminal audit and handoff/registry edits:

```bash
python3 scripts/package_experiment.py \
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
python3 scripts/verify_experiment_package.py \
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
