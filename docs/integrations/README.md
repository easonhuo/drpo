# Dev-integration records

Each real integration request should use its own repository directory:

```text
docs/integrations/<integration-id>/
  INTEGRATION_REQUEST.yaml
  REVIEW_DECISION.yaml
```

The request and decision are reviewer-owned inputs. Runtime transaction records belong in a persistent path outside the tracked source repository.

Current implementation states:

- Batch 1 `plan` produces a `REVIEWED` transaction with source and scope audits.
- Batch 2A `scripts/dev_integration_write_path.py` produces a local `PREPARED` source commit.
- Batch 2B normalization, gates, and final ready commit are not implemented yet.

Do not commit runtime transaction directories, temporary audit repositories, or local integration repositories here. Evidence summaries or closure records may be added later only under an explicitly approved schema and scope.
