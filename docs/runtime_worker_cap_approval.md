# User-approved runtime worker caps

**Policy ID:** `GOV-RUNTIME-WORKER-CAP-HUMAN-APPROVAL-01`  
**Parent:** `GOV-E7-RUNTIME-AUTOTUNE-ADAPTIVE-SEARCH-V3-01`  
**Status:** implementation review; no scientific run is authorized by this document.

## Responsibility split

The three runtime controls have different responsibilities:

- the **resource pool** defines where a workload may execute and the CPU/cgroup capacity visible to it;
- **autotune** measures worker cost and throughput inside that resource boundary and selects concurrency;
- `MAX_WORKERS` is an optional human risk ceiling that may censor autotune's search space.

`MAX_WORKERS` is not an autotune recommendation and is not derived automatically from the pool size. The default is **unset**. When it is unset, autotune owns the concurrency decision subject to measured CPU, memory, task-count, and growth constraints.

## Human-approval rule

An AI agent may inspect evidence and recommend a worker cap. It may not, without explicit repository-owner approval:

- set a cap for the first time;
- increase or decrease an existing cap;
- remove an existing cap from the same run identity;
- substitute a different approval record;
- silently inherit an unapproved environment value;
- describe a cap-censored result as the uncapped autotune optimum.

Every approved cap is exact-run evidence. The approval must be stored as a committed JSON record under `docs/runtime_worker_cap_authorizations/` and must already exist byte-for-byte on trusted `origin/main`. A local commit, local branch, or unmerged pull request cannot authorize a cap. A command-line flag or environment variable is only an assertion of the approved value; it is not itself authorization.

## Default-unset behavior

For a new work directory with no worker cap:

```text
MAX_WORKERS = unset
approval file = unset
autotune selects concurrency inside the resource pool
```

The canonical E7 wrappers write `USER_APPROVED_WORKER_CAP.json` with mode `unset_autotune_controls_concurrency`. Once written, the cap mode is immutable for that work directory. Unset mode does not require an approval record or network access.

## Approved-cap behavior

When a cap is requested, both a value and an approval path are required:

```bash
export DRPO_RUNTIME_MAX_WORKERS=64
export DRPO_RUNTIME_MAX_WORKERS_APPROVAL_FILE=docs/runtime_worker_cap_authorizations/<record>.json
```

E7-specific variables may be used instead, but both must be supplied together:

```bash
export E7_SQUARED_EXP_MAX_WORKERS=64
export E7_SQUARED_EXP_MAX_WORKERS_APPROVAL_FILE=docs/runtime_worker_cap_authorizations/<record>.json
```

Before launch, the server must fetch the authoritative remote state so that `refs/remotes/origin/main` resolves the merged approval. The validator rejects a cap when the approval is missing, untracked, dirty, outside the authorization directory, absent from trusted `origin/main`, byte-different from trusted `origin/main`, malformed, scoped to another experiment/work directory/resource pool, or based on runtime code that changed after approval.

## Approval record schema

Each record is a JSON object:

```json
{
  "schema_version": 1,
  "authorization_id": "E7-WORKER-CAP-YYYYMMDD-01",
  "status": "approved",
  "approved_by": "repository_owner",
  "approval_reference": "durable PR/comment/reference containing explicit user approval",
  "reason": "why an absolute cap is required instead of uncapped autotune",
  "scope": {
    "experiment_id": "EXT-H-...",
    "work_dir": "/absolute/work/directory",
    "max_workers": 64,
    "affinity_cpu_ids": [0, 1, 2],
    "contract_sha256": "...",
    "run_spec_sha256": "...",
    "grid_sha256": "...",
    "approved_code_commit": "40-character commit SHA"
  }
}
```

The approved code commit must be an ancestor of both the launch commit and trusted `origin/main`. The cap validator additionally requires the canonical E7 wrappers, liveness wrapper, auto runner, runtime selector, scientific runner, and tracked grid to remain unchanged since that approved commit.

## Change and resume rules

- The first cap for a run requires a new approval record merged to `origin/main`.
- Any different cap requires a new approval record and a new run/work directory.
- Removing a cap from an already identified work directory is prohibited; use a newly approved run identity whose default is unset.
- Resume must reproduce the same normalized cap identity byte-for-byte.
- Existing V2/V3 work directories created before this policy remain historical artifacts and may resume only under their original source and identity; they are not silently upgraded.

## Reporting

A capped selection must be reported as constrained by a user-approved hard ceiling. It must not be described as an uncapped machine optimum merely because the best tested candidate equals the cap.

This policy changes runtime governance only. It does not change datasets, branches, seeds, training steps, algorithms, scientific thresholds, convergence gates, or experiment responsibilities.
