# DRPO RunSpec executor contract

RunSpec is a file-based contract between the online planning/review AI and a
server-local Claude Code executor. It removes local guessing about which
experiment to run, which checked-in entrypoint to execute, which lane owns the
work, and which result files may be published.

## Active server lanes

Only two server lanes are active:

```text
lane=e7 -> D4RL / Hopper offline-RL experiments -> EXT-H-E7-*
lane=e8 -> Countdown experiments              -> EXT-C-E8-*
```

Historical E1 experiments remain in the scientific registry, but `e1` is not an
active server executor lane.

## One-time server setup

After this feature is merged into `main`, run one command in each clean checkout:

```bash
# E7 D4RL/Hopper server
python scripts/agent/configure_claude_workspace.py --lane e7

# E8 Countdown server
python scripts/agent/configure_claude_workspace.py --lane e8
```

The configurator:

- creates or switches to the fixed lane branch (`dev/server-e7` or
  `dev/server-e8`);
- writes `.agent_lane.yaml` with the lane, prefix boundary, and publish branch;
- writes `CLAUDE.local.md` with the persistent executor role;
- installs a strict Claude Code `PreToolUse` hook in
  `.claude/settings.local.json`.

Tracked changes in the worktree or index block configuration. Ignored output
files do not block it. Restart Claude Code after configuration.

From then on, the human instruction can be:

```text
执行当前 lane 的下一个 READY RunSpec。
```

The canonical command is:

```bash
python scripts/agent/run_lane.py --once
```

## One-command runtime resources

Runtime placement can be attached to that same command without editing the tracked
RunSpec. For example:

```bash
python scripts/agent/run_lane.py --once \
  --cpu-pool 0-95,192-295 \
  --minimum-available-cpu-cores 60 \
  --max-workers 60
```

This invocation:

1. claims the RunSpec through the normal lane rules;
2. restricts the executor and all descendants to the declared Linux CPU list;
3. measures CPU availability only inside that affinity and visible cgroup quotas;
4. waits in the foreground until the declared minimum capacity is available;
5. exports `DRPO_RUNTIME_MAX_WORKERS=60` for compatible launchers;
6. continues through the existing execution, recovery, packaging, and delivery path.

The default resource policy uses `--resource-cpu-fraction 0.85`, one-second
measurements, a 300-second poll interval, and unlimited foreground waiting. The
complete opt-in controls are:

```text
--cpu-pool
--resource-cpu-fraction
--minimum-available-cpu-cores
--resource-wait-timeout-seconds
--resource-poll-seconds
--resource-sample-seconds
--max-workers
```

The CPU pool is the hard placement boundary. The minimum available cores are the
launch floor. `--max-workers` is only a ceiling exported to a compatible runner;
it does not let the generic RunSpec layer infer arbitrary per-worker CPU or memory
demand. Existing workload-specific auto launchers may perform their more detailed
representative-worker probe after inheriting the restricted pool. Sequential or
fixed-width liveness runners may ignore the worker ceiling while still receiving
the pool and capacity wait.

Use non-overlapping CPU pools for independently launched E7 and E8 workloads.
The wrapper never changes affinity of unrelated processes and does not dynamically
resize workers after launch. Runtime identity and wait evidence are written below
`.runspec_state/logs/<run_id>/`.

## Environment prefixes

`entrypoint.command` and `recovery.resume_command` may start with literal
shell-style environment assignments before the executable:

```yaml
entrypoint:
  command: >-
    WORK_DIR=outputs/e8/example
    GRID_CONFIG='configs/example grid.yaml'
    bash scripts/e8/run_example_one_click.sh
```

The executor separates the leading `NAME=value` tokens, adds them to a copy of
the process environment, and executes the remaining argv without `shell=True`.
Shell expansion, command substitution, pipes, redirects, and other arbitrary
shell syntax are not introduced by this compatibility rule. At least one
executable token must follow the assignments.

## State model and reruns

Tracked READY specifications live under `runspecs/ready/`. Local state is stored
under `.runspec_state/`:

```text
claimed/
running/
done/
failed/
published/
logs/
```

A `run_id` is single-use. A run already present in `claimed`, `running`, `done`,
`failed`, or `published` is not claimed again. A new scientific rerun still uses
a new `run_id`; this prevents accidental repeat training and result ambiguity.

## Bounded transient recovery

Recovery is optional and must be declared by the online planner in the same
RunSpec before execution. It does not let the local AI infer a repair, edit a
configuration, change hyperparameters, or invent a resume command.

A recovery-enabled RunSpec declares:

```yaml
recovery:
  enabled: true
  max_attempts: 2
  resume_command: bash scripts/e8/resume_example_one_click.sh
  retryable_exit_codes: [75, 137, 143]
  checkpoint_globs:
    - outputs/e8/example_run/checkpoints/resume_state.json
  backoff_seconds: 60
```

The bounded recovery contract is:

- `max_attempts` counts the initial attempt and is limited to 2 or 3;
- the first attempt uses `entrypoint.command`;
- later attempts use only the checked-in `recovery.resume_command`;
- the failed exit code must be explicitly allow-listed;
- at least one fresh checkpoint matching `checkpoint_globs` must have been
  written by the failed attempt;
- checkpoint paths may not be symbolic links or escape the repository;
- logs containing out-of-memory or NaN/non-finite signatures stop immediately;
- command-start failures, success-criteria failures, packaging failures, and
  publication failures are not training retries.

All attempts remain inside the same single-use `run_id`. The executor stays in
the foreground until the attempt sequence reaches a terminal state. Per-attempt
logs and the machine-readable decision trail are written under:

```text
.runspec_state/logs/<run_id>/attempt-01/
.runspec_state/logs/<run_id>/attempt-02/
.runspec_state/logs/<run_id>/RECOVERY_REPORT.json
```

If the policy does not authorize another attempt, the run moves to `failed` and
the local AI reports the failure instead of changing code or parameters.

## Provenance contract

A production RunSpec pins `repo_commit`. The default provenance policy is
`protected_paths_unchanged`:

- the pinned commit must be the current HEAD or an ancestor;
- the checked-in entrypoint is always protected;
- additional scientific files are listed in `provenance.protected_paths`;
- documentation-only descendant commits are allowed;
- changes to protected scripts/configs after the pinned commit block execution.

`exact_head` is also supported when a separately generated RunSpec can pin the
exact execution HEAD.

## Artifact policy

Artifact packaging is allow-list first. Only files matching `artifacts.include`
can enter the ZIP. Model/checkpoint/optimizer files remain denied by default.
Any symbolic-link path, including a file beneath a symlinked directory, is
rejected rather than followed. Package size limits are enforced before delivery.

## Controlled publication

A completed RunSpec may declare `publish.enabled: true`. In a configured
workspace, `publish.dev_branch` must equal the fixed lane branch recorded in
`.agent_lane.yaml`.

The publisher commits only exact small evidence paths and a generated delivery
manifest. It never commits the result ZIP or model-like files. It pushes the lane
branch and creates or updates a Draft PR. A publish failure returns `PARTIAL` but
keeps the successful computation in `done`, so publication can be retried safely.

The Draft PR is a reviewer handoff, not merge authorization. The online reviewer
must verify provenance, scientific-variable drift, result classification,
registry/handoff changes, and required gates before selective integration from
current `main`.
