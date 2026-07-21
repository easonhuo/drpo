#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage: check_human_approval_record.sh \
  --comments-json PATH \
  --approval-author LOGIN \
  [--new-python-paths PATH] \
  --approval-reason REASON [--approval-reason REASON ...]

Validate that owner-authored pull-request discussion contains a durable record of
an already-granted oral approval. Supported reasons are:
  new_python_file
  hard_gate_policy_change
  large_or_structural_python_change
USAGE
}

comments_json=""
approval_author=""
new_python_paths=""
approval_reasons=()

while (($#)); do
  case "$1" in
    --comments-json)
      comments_json=${2:?missing value for --comments-json}
      shift 2
      ;;
    --approval-author)
      approval_author=${2:?missing value for --approval-author}
      shift 2
      ;;
    --new-python-paths)
      new_python_paths=${2:?missing value for --new-python-paths}
      shift 2
      ;;
    --approval-reason)
      approval_reasons+=("${2:?missing value for --approval-reason}")
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ -z "$comments_json" || -z "$approval_author" ]]; then
  echo "--comments-json and --approval-author are required" >&2
  usage >&2
  exit 2
fi
if ((${#approval_reasons[@]} == 0)); then
  echo "at least one --approval-reason is required" >&2
  usage >&2
  exit 2
fi
if [[ ! -f "$comments_json" ]]; then
  echo "comments JSON does not exist: $comments_json" >&2
  exit 2
fi
if ! jq -e 'type == "array"' "$comments_json" >/dev/null; then
  echo "comments JSON must be an array" >&2
  exit 2
fi
if [[ -n "$new_python_paths" && ! -f "$new_python_paths" ]]; then
  echo "new-Python-path file does not exist: $new_python_paths" >&2
  exit 2
fi

owner_bodies=$(mktemp)
trap 'rm -f "$owner_bodies"' EXIT
jq -r --arg author "$approval_author" \
  '.[] | select((.user.login // "") == $author) | (.body // "")' \
  "$comments_json" >"$owner_bodies"

marker='DRPO-ORAL-APPROVAL: GOV-NEW-PYTHON-FILE-ORAL-APPROVAL-02'

has_exact_line() {
  local expected=$1
  grep -Fqx -- "$expected" "$owner_bodies"
}

has_nonempty_prefixed_line() {
  local prefix=$1
  awk -v prefix="$prefix" '
    index($0, prefix) == 1 && length($0) > length(prefix) { found = 1 }
    END { exit(found ? 0 : 1) }
  ' "$owner_bodies"
}

legacy_path_approved() {
  local path=$1
  jq -e --arg author "$approval_author" --arg path "$path" '
    any(.[]?;
      ((.user.login // "") == $author)
      and ((.body // "") | contains("Human approval record"))
      and ((.body // "") | contains("GOV-NEW-PYTHON-FILE-HUMAN-APPROVAL-01"))
      and ((.body // "") | contains("`" + $path + "`"))
      and ((((.body // "") | ascii_downcase) | contains("responsibility")))
    )
  ' "$comments_json" >/dev/null
}

structured_path_approved() {
  local path=$1
  has_exact_line "$marker" \
    && has_exact_line 'DRPO-APPROVED-REASON: new_python_file' \
    && has_exact_line "DRPO-APPROVED-PATH: $path" \
    && has_nonempty_prefixed_line "DRPO-RESPONSIBILITY: $path :: " \
    && has_nonempty_prefixed_line "DRPO-REUSE-RATIONALE: $path :: "
}

validate_new_python_paths() {
  if [[ -z "$new_python_paths" || ! -s "$new_python_paths" ]]; then
    echo "new_python_file approval requires at least one detected path" >&2
    exit 1
  fi

  while IFS= read -r path || [[ -n "$path" ]]; do
    if [[ -z "$path" || "$path" == *$'\n'* || "$path" == *$'\r'* || "$path" == *$'\t'* ]]; then
      echo "invalid governed Python path in approval input" >&2
      exit 1
    fi
    if structured_path_approved "$path"; then
      continue
    fi
    if legacy_path_approved "$path"; then
      continue
    fi
    echo "missing durable oral-approval record for new Python path: $path" >&2
    exit 1
  done <"$new_python_paths"
}

validate_scoped_reason() {
  local reason=$1
  if ! has_exact_line "$marker"; then
    echo "missing structured oral-approval marker for reason: $reason" >&2
    exit 1
  fi
  if ! has_exact_line "DRPO-APPROVED-REASON: $reason"; then
    echo "missing structured oral-approval reason: $reason" >&2
    exit 1
  fi
  if ! has_nonempty_prefixed_line 'DRPO-APPROVED-SCOPE: '; then
    echo "missing non-empty approved scope for reason: $reason" >&2
    exit 1
  fi
}

seen_reasons=()
for reason in "${approval_reasons[@]}"; do
  for seen in "${seen_reasons[@]:-}"; do
    if [[ "$seen" == "$reason" ]]; then
      echo "duplicate approval reason: $reason" >&2
      exit 2
    fi
  done
  seen_reasons+=("$reason")

  case "$reason" in
    new_python_file)
      validate_new_python_paths
      ;;
    hard_gate_policy_change|large_or_structural_python_change)
      validate_scoped_reason "$reason"
      ;;
    *)
      echo "unsupported approval reason: $reason" >&2
      exit 2
      ;;
  esac
done

echo "human approval record verified for: ${approval_reasons[*]}"
