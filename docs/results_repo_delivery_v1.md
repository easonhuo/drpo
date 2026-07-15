# Results repository delivery V1

## Purpose

Server-side E7/E8 executors should execute, validate, package, and deliver experiment
results without committing result evidence to the DRPO code repository. V1 uploads a
text-first review package to a separate private Git repository such as
`easonhuo/drpo-results`.

The code repository remains the authority for experiment code, RunSpecs, registry,
handoff, and paper changes. The results repository is append-only execution evidence
for online review.

## One-time external setup

1. Create a private repository named `easonhuo/drpo-results`.
2. Grant the online GitHub connector read access to that repository.
3. Give each server a credential that can write only to `drpo-results`.
   A repository-scoped SSH deploy key or fine-grained token is preferred.
4. Verify that the server can clone the SSH URL:

   ```text
   git@github.com:easonhuo/drpo-results.git
   ```

The uploader never reads a token from a RunSpec. Git or SSH authentication is supplied
by the server environment.

## RunSpec contract

```yaml
registration:
  mode: deferred
  closure_required: true

delivery:
  enabled: true
  auto: true
  mode: results_repo
  repository: easonhuo/drpo-results
  branch: ingest/e7
  export_profile: manifest_text_v1
  max_total_size_mb: 30
  max_file_size_mb: 10

publish:
  enabled: false
  auto: false
```

The branch is lane-bound:

```text
e7 -> ingest/e7
e8 -> ingest/e8
```

Legacy `publish.enabled` and results-repository delivery cannot both be enabled.

## Registration timing

RunSpec remains the single lane execution contract and supports two registration
modes:

```yaml
registration:
  mode: pre_registered
```

`pre_registered` is the default when the block is absent. Normal lane claim validation
requires the experiment ID to already exist in `experiments/registry.yaml`.

```yaml
registration:
  mode: deferred
  closure_required: true
```

`deferred` is the code-first route. The lane may claim and execute the RunSpec before
registry or handoff materialization completes. Deferred mode requires `repo_commit` to
be a full 40-character Git SHA, and `closure_required` is always true. It changes only
registration timing: it does not change the RunSpec's scientific class, formal/pilot
status, method matrix, seeds, budget, or reporting duties.

A completed deferred RunSpec follows the same packaging and delivery path as a
pre-registered RunSpec. Result upload must not wait for registry/handoff closure and
must not downgrade a formal run merely because registration is still pending. Later
registration should reference the immutable `run_id`, source commit, and delivered
manifest SHA. A registration conflict is reviewed separately; it does not delete or
rewrite already delivered evidence.

The explicit validator option `--no-registry-check` remains an operator/debug override.
Normal production code-first execution should declare `registration.mode: deferred`
instead of relying on that flag.

## Real remote shadow validation

The first real remote check is intentionally not a new training run. The template
`runspecs/templates/E8_RESULTS_REPO_DELIVERY_SHADOW_20260714_01.yaml` uses the already
completed compact evidence under
`experiments/results/e8_oracle_offline_v2_init_matrix_pilot`. Its entrypoint hashes the
source files, records that no training occurred, and leaves the source unchanged.

The shadow validates these boundaries together:

1. an unregistered `registration.mode: deferred` RunSpec is accepted by the normal E8
   lane claim path;
2. the completed RunSpec is packaged without model/checkpoint evidence;
3. the package is pushed to `drpo-results` branch `ingest/e8`;
4. the remote result contains matching `RESULT_MANIFEST.json` and
   `READY_FOR_REVIEW.json` files;
5. a second identical upload returns `ALREADY_DELIVERED` and creates no new result
   content;
6. the original compact evidence remains byte-identical and no training is executed.

Before promoting the template, configure the repository-scoped credential in the E8
executor environment and verify read authentication:

```bash
git ls-remote git@github.com:easonhuo/drpo-results.git
```

Then use the reviewed PR branch and validate the deferred RunSpec without an operator
registry bypass:

```bash
git fetch origin agent/results-repo-delivery-v1
git checkout agent/results-repo-delivery-v1
python scripts/agent/validate_runspec.py \
  --lane e8 \
  runspecs/templates/E8_RESULTS_REPO_DELIVERY_SHADOW_20260714_01.yaml
```

Promote exactly that reviewed template and execute one lane task:

```bash
cp \
  runspecs/templates/E8_RESULTS_REPO_DELIVERY_SHADOW_20260714_01.yaml \
  runspecs/ready/E8_RESULTS_REPO_DELIVERY_SHADOW_20260714_01.yaml
python scripts/agent/run_lane.py \
  --lane e8 \
  --run-id E8_RESULTS_REPO_DELIVERY_SHADOW_20260714_01 \
  --once \
  --json
```

The first command must return an execution payload with `status: PASS`,
`delivery_status: PASS`, a results commit, and a manifest SHA. The local audit file is:

```text
.runspec_state/delivery/E8_RESULTS_REPO_DELIVERY_SHADOW_20260714_01/DELIVERY_REPORT.json
```

Run the canonical manual retry against the completed state:

```bash
python scripts/agent/upload_runspec_result.py \
  --run-id E8_RESULTS_REPO_DELIVERY_SHADOW_20260714_01 \
  --json
```

The retry must return `ALREADY_DELIVERED` with the same result path and manifest SHA.
The online review then verifies the remote directory
`runs/e8/E8_RESULTS_REPO_DELIVERY_SHADOW_20260714_01/`, confirms that all manifest
hashes match, and confirms that the package records deferred registration timing and no
model-like files.

A credential or network failure is not retried by rerunning the entrypoint. The RunSpec
must remain in `.runspec_state/done/`; after fixing authentication, use only the manual
uploader command above.

## Simple size policy

V1 intentionally uses one fixed, simple rule instead of LFS, Release assets, Drive,
or object storage:

```text
single review-package file <= 10 MiB
whole review package       <= 30 MiB
```

A RunSpec may choose stricter limits, but it may not raise either V1 cap. The existing
full local result ZIP is not uploaded by this channel and may be larger than 30 MiB.
Only the generated text-first review package is measured against these limits.

When either limit is exceeded:

- no result-repository commit or push is attempted;
- the experiment remains successfully completed in `.runspec_state/done/`;
- the existing local artifact ZIP and SHA-256 remain available;
- the partial review-package directory is removed;
- `DELIVERY_REPORT.json` records `status: RESULT_TOO_LARGE`;
- the canonical executor command exits successfully, so training is not retried.

This is a normal delivery downgrade, not an experiment failure. Manual upload remains
a human fallback when the complete result is needed.

## Review package

The uploader uses the already validated RunSpec artifact manifest as its source. It
rechecks every source file's path, symlink status, size, and SHA-256 before export.
Model, checkpoint, optimizer, and dataset-like evidence remain outside this channel.

V1 copies normal text files under `files/` while compacting JSON files below a
`branches/` directory into `BRANCH_RESULTS.jsonl`. `FAILED.json` entries are compacted
into `FAILURES.jsonl`. Non-text files are omitted and recorded in
`SOURCE_ARTIFACT_MANIFEST.json`.

Each result directory has this shape:

```text
runs/<lane>/<run_id>/
├── files/...
├── BRANCH_RESULTS.jsonl       # when branch JSON exists
├── FAILURES.jsonl             # when failed branch JSON exists
├── SOURCE_ARTIFACT_MANIFEST.json
├── README.md
├── RESULT_MANIFEST.json
└── READY_FOR_REVIEW.json
```

`RESULT_MANIFEST.json` records every review-package file with size and SHA-256.
`READY_FOR_REVIEW.json` references its manifest SHA and is written only after export is
complete. The Git commit makes the remote update atomic.

## Append-only and retry behavior

The remote path is fixed as `runs/<lane>/<run_id>/`.

- Missing path: create, commit, and push it.
- Existing path with the same manifest SHA: return `ALREADY_DELIVERED`.
- Existing path with a different manifest SHA: fail with `RESULT_CONFLICT`.
- Upload failure: keep the RunSpec in `done`; do not rerun training.
- Oversize package: return `RESULT_TOO_LARGE`; do not attempt upload or rerun training.
- Manual retry:

  ```bash
  python scripts/agent/upload_runspec_result.py --run-id <run_id>
  ```

A machine-readable local report is written to:

```text
.runspec_state/delivery/<run_id>/DELIVERY_REPORT.json
```

## Repository cache and test overrides

The uploader caches a checkout under:

```text
.runspec_state/results_repo/<owner>__<repository>/
```

Two environment variables exist for controlled testing and unusual server layouts:

```text
DRPO_RESULTS_REMOTE_URL
DRPO_RESULTS_CACHE_DIR
```

Normal production RunSpecs should not depend on these variables. Without a remote URL
override, the uploader derives `git@github.com:<owner>/<repository>.git`.

## V1 boundaries

V1 intentionally does not provide:

- Git LFS;
- Google Drive or object-storage delivery;
- GitHub Release assets;
- checkpoint or model upload;
- automatic code-repository commits or Draft PRs;
- automatic scientific acceptance;
- a cross-project result service.

The online reviewer remains responsible for result interpretation and any later
registry, handoff, paper, or code changes in the DRPO repository.
