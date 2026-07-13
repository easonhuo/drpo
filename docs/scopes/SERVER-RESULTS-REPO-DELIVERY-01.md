# SERVER-RESULTS-REPO-DELIVERY-01

## Objective

Replace the default server result handoff path with an opt-in, append-only delivery
channel to a separate results-only Git repository. The local executor may upload
validated review evidence, but it must not commit experiment results to the DRPO code
repository or authorize scientific integration.

## In scope

- `scripts/agent/runspec_results_delivery.py`
- `scripts/agent/runspec_delivery_policy.py`
- `scripts/agent/upload_runspec_result.py`
- RunSpec validation for the new `delivery` block and fixed V1 size caps
- automatic delivery after successful execution and artifact packaging
- strict Claude executor allowance for the canonical upload command
- text-first result export, branch-JSON compaction, SHA-256 manifesting, size limits,
  symlink rejection, append-only conflict checks, and idempotent retry
- graceful `RESULT_TOO_LARGE` downgrade without retraining or remote writes
- local bare-repository tests and operator documentation

## Explicitly excluded

- changes to scientific code, datasets, methods, seeds, horizons, or hyperparameters
- changes to registry or handoff scientific state
- model, checkpoint, optimizer, dataset, or cache upload
- Git LFS, Google Drive, object storage, Release assets, or self-hosted Actions runners
- automatic scientific acceptance or merge
- removal of the legacy publish implementation

## Security and governance boundaries

- delivery is disabled unless explicitly declared in a RunSpec;
- `delivery.branch` is fixed to `ingest/<lane>`;
- delivery and legacy `publish` cannot both be enabled;
- V1 permits at most 10 MiB per review-package file and 30 MiB in total;
- a RunSpec may choose stricter limits but cannot raise those caps;
- oversize review packages remain local, keep the RunSpec in `done`, and return
  `RESULT_TOO_LARGE` without attempting commit or push;
- the uploader derives an SSH remote from `delivery.repository` and never stores a
  credential in the RunSpec;
- source artifact paths, sizes, and SHA-256 values are revalidated before export;
- a remote `runs/<lane>/<run_id>/` directory is immutable after first delivery;
- same-manifest retries are idempotent and different-manifest retries fail closed;
- upload failure preserves the completed local run and never retriggers training.

## Required validation

- Python compilation
- Ruff
- targeted delivery tests against a local bare Git remote
- fixed-cap and oversize-downgrade tests
- strict executor guard coverage
- full pytest
- handoff authority, formal execution channel, and governance gates

## Real-shadow boundary

The private `easonhuo/drpo-results` repository exists and the online GitHub connector
has read/write access. A real remote shadow remains blocked only until an E7 or E8
execution environment has a repository-scoped write credential. The first shadow must
use a previously completed result package and must not rerun training.
