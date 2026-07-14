# SERVER-RESULTS-REPO-DELIVERY-01

## Objective

Replace the default server result handoff path with an opt-in, append-only delivery
channel to a separate results-only Git repository. The local executor may upload
validated review evidence, but it must not commit experiment results to the DRPO code
repository or authorize scientific integration.

## In scope

- `scripts/agent/runspec_results_delivery.py`
- `scripts/agent/runspec_delivery_policy.py`
- `scripts/agent/runspec_registration.py`
- `scripts/agent/upload_runspec_result.py`
- `scripts/agent/prepare_results_repo_delivery_shadow.py`
- `runspecs/templates/E8_RESULTS_REPO_DELIVERY_SHADOW_20260714_01.yaml`
- RunSpec validation for the new `delivery` block and fixed V1 size caps
- dual RunSpec registration timing: default `pre_registered` and code-first `deferred`
- lane claim, execution, packaging, and upload for deferred-registration RunSpecs
- automatic delivery after successful execution and artifact packaging
- strict Claude executor allowance for the canonical upload command
- text-first result export, branch-JSON compaction, SHA-256 manifesting, size limits,
  symlink rejection, append-only conflict checks, and idempotent retry
- graceful `RESULT_TOO_LARGE` downgrade without retraining or remote writes
- a no-training real-remote shadow based on previously completed compact E8 evidence
- local bare-repository tests and operator documentation

## Explicitly excluded

- changes to scientific code, datasets, methods, seeds, horizons, or hyperparameters
- changes to registry or handoff scientific state
- a separate code-first execution pipeline outside RunSpec/lane
- automatic post-hoc alteration of a completed RunSpec contract
- model, checkpoint, optimizer, dataset, or cache upload
- Git LFS, Google Drive, object storage, Release assets, or self-hosted Actions runners
- automatic scientific acceptance or merge
- removal of the legacy publish implementation

## Security and governance boundaries

- delivery is disabled unless explicitly declared in a RunSpec;
- absent `registration`, RunSpec behavior remains `pre_registered` and registry-gated;
- `registration.mode: deferred` bypasses only the pre-execution registry lookup;
- deferred registration requires a full 40-character `repo_commit` and
  `closure_required: true`;
- deferred timing does not downgrade or upgrade formal/pilot scientific status;
- a later registry/handoff closure must reference the immutable run identity and
  evidence rather than rewriting delivered results;
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
- upload failure preserves the completed local run and never retriggers training;
- the first real shadow hashes and packages already completed compact evidence only;
- the shadow is engineering evidence, performs no training, and is excluded from
  scientific aggregation.

## Required validation

- Python compilation
- Ruff
- targeted delivery tests against a local bare Git remote
- fixed-cap and oversize-downgrade tests
- pre-registered backward-compatibility tests
- deferred-registration validation, lane-claim, and explicit override tests
- no-training shadow preparation tests, including source immutability, symlink rejection,
  and model-like-file rejection
- strict executor guard coverage
- full pytest
- handoff authority, formal execution channel, and governance gates
- real remote first push returns `PASS`
- identical manual retry returns `ALREADY_DELIVERED`
- remote `READY_FOR_REVIEW.json` and `RESULT_MANIFEST.json` agree on manifest SHA
- remote package records `registration.mode: deferred` and contains no model-like files

## Real-shadow boundary

The private `easonhuo/drpo-results` repository exists and the online GitHub connector
has read/write access. The no-training E8 shadow RunSpec is prepared as a template and
uses the already completed compact result under
`experiments/results/e8_oracle_offline_v2_init_matrix_pilot`. Promotion from template to
`runspecs/ready/` remains blocked until an E8 execution environment has a
repository-scoped write credential. The shadow must not rerun training or alter the
source evidence.
