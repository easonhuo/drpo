#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

if [[ $# -ne 2 || "$1" != "--profile" ]]; then
  echo "Usage: bash scripts/run_runtime_resource_acceptance_partitioned_one_click.sh" >&2
  echo "  --profile /absolute/profile.json" >&2
  exit 2
fi
if [[ "$2" != /* ]]; then
  echo "Profile path must be absolute: $2" >&2
  exit 2
fi

export PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
cd "${REPO_ROOT}"

python3 scripts/run_runtime_resource_acceptance_partitioned.py \
  --profile "$2" \
  --check-only

exec python3 scripts/run_runtime_resource_acceptance_partitioned.py \
  --profile "$2"
