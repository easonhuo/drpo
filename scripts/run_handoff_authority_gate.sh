#!/usr/bin/env bash
set -euo pipefail

repo_root="."
base_sha=""
head_sha=""

usage() {
  cat <<'EOF'
Usage:
  bash scripts/run_handoff_authority_gate.sh \
    --repo-root <path> \
    --base <full-git-sha> \
    --head <full-git-sha>

The gate is read-only. It classifies the exact base-to-head diff and runs the
Stage 5 handoff-authority checks only when authority-controlled paths changed.
EOF
}

while (($#)); do
  case "$1" in
    --repo-root)
      repo_root="$2"
      shift 2
      ;;
    --base)
      base_sha="$2"
      shift 2
      ;;
    --head)
      head_sha="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "ERROR: unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ ! "$base_sha" =~ ^[0-9a-f]{40}$ ]]; then
  echo "ERROR: --base must be a full lowercase Git SHA" >&2
  exit 2
fi
if [[ ! "$head_sha" =~ ^[0-9a-f]{40}$ ]]; then
  echo "ERROR: --head must be a full lowercase Git SHA" >&2
  exit 2
fi

repo_root="$(cd "$repo_root" && pwd)"
cd "$repo_root"

git cat-file -e "${base_sha}^{commit}"
git cat-file -e "${head_sha}^{commit}"

mapfile -t changed_paths < <(git diff --name-only "$base_sha" "$head_sha")
authority_changed=false
for path in "${changed_paths[@]}"; do
  case "$path" in
    .github/workflows/handoff-authority.yml|\
    scripts/run_handoff_authority_gate.sh|\
    scripts/handoff_authority.py|\
    scripts/handoff_delta_shadow.py|\
    scripts/build_stage4_context.py|\
    scripts/validate_governance_pipeline_stage_status.py|\
    docs/handoff.md|\
    docs/handoff_deltas/*|\
    docs/handoff_versions/*|\
    docs/handoff_shadow/stage4/minimal/generated/*|\
    docs/handoff_shadow/stage4/minimal/MODULES.yaml|\
    docs/handoff_shadow/stage4/minimal/DEPENDENCIES.yaml|\
    docs/handoff_delta_policy.yaml|\
    docs/handoff_delta_protocol.md|\
    docs/handoff_delta_state_machines.yaml|\
    docs/governance_pipeline_stage_status.yaml|\
    docs/governance_stage5_versioned_handoff_spec.md|\
    docs/governance_stage_authorizations/GOV-HANDOFF-AUTHORITY-*.yaml|\
    docs/scopes/GOV-HANDOFF-AUTHORITY-*.md|\
    experiments/registry.yaml)
      authority_changed=true
      ;;
  esac
done

if [[ "$authority_changed" != true ]]; then
  echo "Handoff authority gate: SKIP (no authority-controlled path changed)"
  exit 0
fi

delta_count="$(
  git diff --name-status "$base_sha" "$head_sha" -- \
    'docs/handoff_deltas/*/HANDOFF_DELTA.yaml' |
    awk '$1 == "A" {count += 1} END {print count + 0}'
)"
materialized_path_count="$(
  git diff --name-only "$base_sha" "$head_sha" -- \
    docs/handoff.md \
    experiments/registry.yaml \
    'docs/handoff_shadow/stage4/minimal/generated/**' |
    awk 'NF {count += 1} END {print count + 0}'
)"

if (( materialized_path_count > 0 && delta_count != 1 )); then
  echo "ERROR: handoff/registry/generated-view changes require exactly one newly added schema-v3 delta; found ${delta_count}" >&2
  exit 2
fi
if (( delta_count == 1 && materialized_path_count == 0 )); then
  echo "ERROR: a new schema-v3 delta is present but no materialized handoff, registry, or generated-view change is committed" >&2
  exit 2
fi

echo "Handoff authority gate: RUN"
echo "Base: $base_sha"
echo "Head: $head_sha"
echo "Changed paths:"
printf '  - %s\n' "${changed_paths[@]}"

python scripts/handoff_authority.py verify --repo-root . --json
python scripts/build_stage4_context.py --repo-root . --json check
python scripts/validate_governance_pipeline_stage_status.py --repo-root .

if [[ -n "${GITHUB_STEP_SUMMARY:-}" ]]; then
  {
    echo "### Handoff authority gate"
    echo
    echo "- Base: \`$base_sha\`"
    echo "- Head: \`$head_sha\`"
    echo "- New schema-v3 deltas: \`$delta_count\`"
    echo "- Materialized authority paths changed: \`$materialized_path_count\`"
    echo "- Result: **PASS**"
  } >> "$GITHUB_STEP_SUMMARY"
fi
