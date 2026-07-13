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
