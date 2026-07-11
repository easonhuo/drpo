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
`failed`, or `published` is not claimed again. A rerun or retry must use a new
`run_id`; this prevents accidental repeat training and result ambiguity.

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
