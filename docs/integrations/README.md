# Dev integration request records

Each reviewed integration may create one repository directory:

```text
docs/integrations/<integration-id>/
  INTEGRATION_REQUEST.yaml
  REVIEW_DECISION.yaml
```

Use the templates under `docs/templates/`. These files are reviewer inputs, not machine transaction state. Machine-generated `SOURCE_LOCK.json`, `SCOPE_AUDIT.json`, `TRANSACTION.json`, `DIAGNOSTIC.json`, normalization/gate reports, logs, and ready-commit records belong under the untracked persistent `--transaction-root` supplied to the CLI.

After Batch 2A reaches `PREPARED`, a transaction may optionally add the following untracked inputs directly inside its attempt directory:

```text
REGISTRATION_INTENT.yaml
REGISTRATION_APPROVAL.yaml
```

Both files must be present together. The approval binds the exact intent bytes to the already locked request, reviewer decision, reviewer identity, and review token. The target experiment must already be named in the reviewed request. Absence of both files means code-only normalization; the tool does not infer or invent registration content.

Do not commit credentials, GitHub tokens, model weights, datasets, large result artifacts, or local absolute paths in an integration request. `reviewer.decision_token` is a review-specific identifier, not an authentication secret.

Batch 1 ends at `REVIEWED`, Batch 2A ends at `PREPARED`, and Batch 2B ends at local `READY`. See `docs/dev_branch_integration_protocol.md` for the commands and current boundaries.
