# GOV-EXPERIMENT-LAUNCH-SIMPLIFICATION-01 — Global Experiment Launch Simplification

**Status:** user-authorized for implementation and Draft PR  
**Base commit:** `ea2c6d60084acd3e87d6f4118161747364bfe9b4`  
**Authorization record:** `docs/governance_stage_authorizations/GOV-EXPERIMENT-LAUNCH-SIMPLIFICATION-2026-08-22.yaml`  
**Scientific experiment impact:** none; no task, dataset, seed, coefficient, optimizer, horizon, metric, result status, or execution ordering is changed

## 1. Purpose

Reduce DRPO experiment-launch overhead by permanently retiring the non-scientific pre-launch activation stack for future experiments and reruns.

The following mechanisms are no longer allowed to be mandatory launch prerequisites after this policy merges:

1. READY RunSpec promotion or READY-directory activation;
2. `experiments/registry.yaml` `execution_gate` / implementation-identity activation used as an execution license;
3. Stage-5/schema-v3 registration transactions performed only to satisfy the two launch gates above;
4. `validate_formal_execution_channel.py` or equivalent registry-completeness/channel-activation validation used as a GPU/workload launch gate;
5. lane activation or duplicate identity activation whose sole purpose is to prove READY/registry state before workload execution.

Historical RunSpecs, registry records, handoff deltas, activation records, and completed experiment provenance are preserved and must not be destructively deleted.

## 2. New default launch contract

The default scientific launch path becomes:

```text
scientific protocol documented and frozen
→ exact source commit selected
→ clean checkout / source identity check
→ frozen config + data/seeds/threshold identity check
→ Run ID + run manifest / source provenance
→ foreground guard and recovery
→ workload
→ terminal audit
→ durable packaging / delivery
→ registry / handoff status recording as a non-blocking bookkeeping step when needed
```

A launch may not be blocked merely because a registry entity, READY RunSpec, schema-v3 registration transaction, or channel activation record is absent or stale.

## 3. What remains mandatory

This change does **not** weaken scientific or evidence integrity. The following remain hard requirements where applicable:

- document-before-experiment scientific protocol freeze;
- exact Git commit and canonical repository identity;
- clean checkout for formal runs, or an explicitly captured dirty-pilot snapshot where already permitted;
- frozen scientific configuration, data source/version, seeds, thresholds, coefficient grid, optimizer, horizon, and stopping/terminal criteria;
- no silent scientific-variable modification during recovery;
- foreground supervision for long/ephemeral formal runs;
- recovery/checkpoint identity binding;
- terminal-state audit before steady-state, collapse, or method-ranking claims;
- separate reporting of task-performance collapse, support/variance-boundary events, and NaN/Inf numerical failure;
- durable result packaging, checksums, source provenance, and final artifact verification;
- explicit user approval for scientific protocol changes and repository merges under the existing GitHub development route.

## 4. Registry and RunSpec roles after simplification

`experiments/registry.yaml` is retained as a project index/history source. A RunSpec may be retained as an optional immutable execution snapshot or server convenience. Neither is an execution license.

For new work:

- registry entries may be created or refreshed asynchronously and may be finalized after launch/result collection;
- RunSpecs do not require promotion to `runspecs/ready/` before execution;
- a stale or missing registry `execution_gate` does not block an otherwise source/config-valid launch;
- experiment identity should be captured in the run manifest and result provenance even when registry bookkeeping is deferred.

## 5. Stage-5 distinction

Stage-5/schema-v3 handoff authority may continue to be used when the repository actually needs to materialize `docs/handoff.md` or `experiments/registry.yaml` changes. This policy retires **schema-v3 registration as a launch permission**, not the lossless handoff-authority machinery itself.

No experiment must wait for a handoff/registry materialization transaction solely to obtain permission to touch GPUs or start a frozen workload.

## 6. Compatibility policy

Existing historical runners and documents may contain READY/registry/channel terminology. They are preserved as provenance. Active or newly modified launchers must follow this policy and must not reintroduce those terms as launch prerequisites.

The hardened guard/package/verify tooling remains reusable safety infrastructure. It protects source identity, supervision, evidence, and artifact durability; it must not depend on registry/READY activation to decide whether a frozen workload can start.

## 7. PR/CI policy

Routine PR CI must not execute registry/channel activation validation merely to approve ordinary code or experiment-launch changes. Compile, shell syntax, tests, Ruff, handoff-authority checks when handoff-related files change, and governance-stage checks when governance-protected files change remain available.

Historical formal-channel tests may remain as compatibility tests for legacy registry data, but they are not a prerequisite for new experiment registration or launch.

## 8. Rollback

Rollback is intentionally simple and non-destructive:

1. revert the policy-routing changes in `AGENTS.md`, `runspecs/README.md`, and routine PR workflow to the exact base behavior at `ea2c6d60084acd3e87d6f4118161747364bfe9b4`;
2. mark this scope as reverted while preserving this file and the authorization record;
3. re-enable READY/registry/channel activation only through a new reviewed governance change;
4. do not modify or invalidate experiments that ran under this simplified policy; their exact commit/config/run-manifest provenance remains authoritative evidence;
5. preserve all historical registry, RunSpec, handoff, result, and artifact records.

## 9. Acceptance criteria

The implementation is acceptable only if:

- no scientific Python implementation or scientific config is changed;
- current running experiments are untouched;
- `AGENTS.md` clearly makes registration/READY/channel activation non-blocking for future experiments;
- `runspecs/README.md` clearly reclassifies READY as legacy/optional rather than required activation;
- the old pilot-registration fastpath is marked superseded, not deleted;
- routine PR CI no longer runs formal execution-channel activation validation;
- compile, shell syntax, relevant focused tests, full pytest, Ruff, governance inventory/stage checks, and handoff authority verification are reviewed after the change;
- no historical file or result is destructively deleted;
- merge is authorized by the user's explicit 2026-08-22 instruction for this exact scope once the reviewed diff remains within scope and required CI passes; any scope expansion requires new approval.
