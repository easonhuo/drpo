# Dev integration request records

Each reviewed integration may create one directory:

```text
docs/integrations/<integration-id>/
  INTEGRATION_REQUEST.yaml
  REVIEW_DECISION.yaml
```

Use the templates under `docs/templates/`. These files are reviewer inputs, not machine transaction state. Machine-generated `SOURCE_LOCK.json`, `SCOPE_AUDIT.json`, `TRANSACTION.json`, `DIAGNOSTIC.json`, gate reports, and ready-commit records belong under the untracked persistent `--transaction-root` supplied to the CLI.

Do not commit credentials, GitHub tokens, model weights, datasets, large result artifacts, or local absolute paths in an integration request. `reviewer.decision_token` is a review-specific identifier, not an authentication secret.

Batch 1 is read-only and ends at `REVIEWED`. See `docs/dev_branch_integration_protocol.md` for commands and current limitations.
