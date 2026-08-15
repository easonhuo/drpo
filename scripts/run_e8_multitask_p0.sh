#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${repo_root}/src${PYTHONPATH:+:${PYTHONPATH}}"

exec python -m drpo.e8_multitask_p0 \
  --config "${repo_root}/configs/e8_multitask_p0.yaml" \
  "$@"
