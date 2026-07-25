# Provenance correction

Archive:
`EXT-H-E7-SQEXP-GAE-FROZEN-CRITIC-DIAGNOSTIC-2026-07-18`

The initial compact archive stated that the reviewed ZIP did not expose a
resolvable repository source commit. That statement described the locally
reviewed ZIP only.

The durable delivery in `easonhuo/drpo-results` contains an authoritative
`RESULT_MANIFEST.json` that records source commit
`14fceaf18ce827acabc4a04a4662ae0fa99cb9a7`. The commit is remotely resolvable
in `easonhuo/drpo` and is therefore the source commit recorded by the evidence
locator.

Raw-result identity:

- run ID: `E7_SQEXP_GAE_PILOT_20260717_04`;
- results repository: `easonhuo/drpo-results`;
- results branch: `ingest/e7`;
- results commit: `8a06e4ed9b5ae282fffd4bac1f6ac99b21feb197`;
- result path: `runs/e7/E7_SQEXP_GAE_PILOT_20260717_04`;
- manifest SHA-256:
  `2449b1872bf70533c63d5b4446884bdb967ba1cba4961c63d38f584b4c86fc10`.

This correction changes provenance resolution only. It does not change the
scientific boundary: the result remains a superseded frozen-critic,
precomputed-advantage development diagnostic and does not satisfy the current
joint-critic `EXT-H-E7-SQEXP-GAE-01` protocol.
