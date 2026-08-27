# Results-repository transport preflight

Use this read-only probe before promoting the no-training E7/E8 results-delivery shadow
RunSpec. It does not claim a lane task, clone a repository, create a commit, or push.

## Canonical command

```bash
python scripts/agent/diagnose_results_repo_transport.py \
  --json \
  --output /tmp/DRPO_RESULTS_TRANSPORT_PREFLIGHT.json
```

To make a blocked result fail a shell gate without changing probe behavior:

```bash
python scripts/agent/diagnose_results_repo_transport.py \
  --strict \
  --json \
  --output /tmp/DRPO_RESULTS_TRANSPORT_PREFLIGHT.json
```

The default remote is:

```text
git@github.com:easonhuo/drpo-results.git
```

An executor may select a reviewed alternate transport through
`DRPO_RESULTS_REMOTE_URL` or `--remote`. Do not place a token in a RunSpec. When an
HTTPS URL includes userinfo, the probe redacts it from output, but environment-scoped or
credential-helper authentication remains preferable.

## What the probe checks

- effective `ssh -G github.com` host, port, proxy, and identity-file paths;
- DNS resolution;
- TCP reachability for the configured endpoint;
- fallback reachability for `github.com:22`, `ssh.github.com:443`, and
  `github.com:443`;
- presence, not values, of relevant credential environment variables;
- availability of `git`, `ssh`, and `gh`;
- SSH-agent identity count without fingerprints;
- read authentication through timeout-bounded, non-interactive `git ls-remote`.

The probe specifically warns when SSH configuration maps GitHub to a non-standard port,
such as the observed `36000`.

## Status interpretation

### `BLOCKED_BY_NETWORK`

Repair routing, firewall, proxy, DNS, or port selection first. A credential cannot fix a
TCP timeout.

### `BLOCKED_BY_CONFIGURATION`

Correct the remote URL, host-key setup, SSH config, ProxyCommand, or ProxyJump before
installing credentials.

### `BLOCKED_BY_CREDENTIAL`

The transport is reachable but repository authentication failed. Provision a
repository-scoped deploy key or fine-grained token outside the RunSpec, then repeat the
probe.

### `READY_FOR_SHADOW_READ_PREFLIGHT`

`git ls-remote` succeeded. This proves only read access. Confirm the intended
repository-scoped write setup and then execute the preserved no-training shadow exactly
once. The shadow push is the first write-authority test.

## Required handoff to the reviewer

Provide the generated JSON report. Do not provide private keys, tokens, credential
helper payloads, or an unredacted remote URL.
