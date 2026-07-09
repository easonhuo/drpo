#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  scripts/run_e7_canonical_two_dataset.sh <profile> --data-dir /ABS/D4RL_HDF5_DIR --work-dir /ABS/OUTPUT_DIR [extra args]

Profiles:
  smoke        20k-step liveness, original ExpRank_MR passthrough only.
  reproduce   1M-step original ExpRank_MR passthrough only. Use this to check the old backbone scale.
  taper-pilot 300k-step small taper/control grid. Use only after reproduce is sane.
  full-grid    1M-step broad exploratory grid. Do not launch until the small pilot is reviewed.

The script uses the vendored canonical old D4RL code under
src/drpo/e7_canonical_vendor/d4rl by default.  Override with --canonical-root
only for source-lineage audits.
EOF
}

if [[ $# -lt 1 || "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

PROFILE="$1"
shift
case "$PROFILE" in
  smoke|reproduce|taper-pilot|full-grid) ;;
  *)
    echo "Unknown profile: $PROFILE" >&2
    usage >&2
    exit 2
    ;;
esac

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

python scripts/run_e7_canonical_two_dataset.py run \
  --profile "$PROFILE" \
  "$@"
