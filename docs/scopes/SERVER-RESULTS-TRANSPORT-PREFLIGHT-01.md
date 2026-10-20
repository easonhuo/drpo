# SERVER-RESULTS-TRANSPORT-PREFLIGHT-01

## Objective

Add a read-only server preflight that separates network, SSH/remote configuration, and
repository-authentication blockers before an E7/E8 results-delivery shadow RunSpec is
promoted to `runspecs/ready/`.

The preflight supports `SERVER-RESULTS-REPO-DELIVERY-01` but is independently useful on
current `main`. It does not depend on the unmerged delivery implementation and does not
change scientific or experiment authority.

## In scope

- `scripts/agent/diagnose_results_repo_transport.py`
- deterministic unit tests for redaction, SSH effective-config parsing, transport target
  derivation, blocker classification, and the read-only command boundary
- JSON and concise human-readable output
- default remote `git@github.com:easonhuo/drpo-results.git`
- optional `DRPO_RESULTS_REMOTE_URL` or `--remote` override
- inspection of credential presence without printing token or key contents
- `ssh -G`, DNS, TCP, and `git ls-remote` probes with bounded timeouts
- explicit detection of non-standard GitHub SSH ports such as the observed port `36000`

## Explicitly excluded

- claiming or executing a RunSpec
- copying a template into `runspecs/ready/`
- cloning either repository
- committing or pushing either repository
- testing remote write authority
- creating deploy keys, tokens, or other credentials
- changing SSH, proxy, Git, firewall, or network configuration
- changing `docs/handoff.md`, `experiments/registry.yaml`, scientific code, datasets,
  methods, seeds, thresholds, budgets, or result status
- merging PR #51 or this scope's PR

## Status vocabulary

The probe emits exactly one primary status:

- `READY_FOR_SHADOW_READ_PREFLIGHT`: `git ls-remote` succeeded; read authentication is
  available, but push authority remains unverified.
- `BLOCKED_BY_NETWORK`: the configured transport cannot establish the required network
  connection or name resolution.
- `BLOCKED_BY_CREDENTIAL`: the transport is reachable but repository authentication or
  authorization failed.
- `BLOCKED_BY_CONFIGURATION`: the remote or SSH configuration failed before a reliable
  network/authentication decision.

## Security and side-effect boundaries

- token values, URL userinfo, and recognized GitHub token forms are redacted;
- environment variables are reported only as present/absent;
- SSH-agent evidence is reduced to status and identity count, not fingerprints;
- the default invocation performs no repository write;
- an optional `--output` path writes only the generated JSON report;
- `--strict` may return exit code `2` for a blocked status but does not add side effects;
- `git ls-remote` is the strongest remote operation permitted.

## Required validation

- Python compilation
- targeted unit tests
- full repository pytest through the normal PR gate
- Ruff through the normal PR gate
- handoff authority and governance validators through the normal PR gate
- manual server execution before the preserved no-training results-delivery shadow

## Interpretation boundary

A successful read preflight is necessary but insufficient for the real shadow. It does
not prove write permission. The first append-only shadow push remains the authority for
repository-scoped write capability.
