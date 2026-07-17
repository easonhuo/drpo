#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage: check_new_python_file_gate.sh --repo-root PATH --base COMMIT --head COMMIT [--output PATH]

Print each new Python destination path on its own line. A path is governed when it
exists at HEAD, does not exist at BASE, and ends in .py case-insensitively.
USAGE
}

repo_root="."
base=""
head=""
output=""

while (($#)); do
  case "$1" in
    --repo-root)
      repo_root=${2:?missing value for --repo-root}
      shift 2
      ;;
    --base)
      base=${2:?missing value for --base}
      shift 2
      ;;
    --head)
      head=${2:?missing value for --head}
      shift 2
      ;;
    --output)
      output=${2:?missing value for --output}
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

if [[ -z "$base" || -z "$head" ]]; then
  echo "--base and --head are required" >&2
  usage >&2
  exit 2
fi

if [[ ! -d "$repo_root/.git" ]]; then
  echo "repo root is not a Git worktree: $repo_root" >&2
  exit 2
fi

base_sha=$(git -C "$repo_root" rev-parse --verify "${base}^{commit}") || {
  echo "could not resolve base commit: $base" >&2
  exit 2
}
head_sha=$(git -C "$repo_root" rev-parse --verify "${head}^{commit}") || {
  echo "could not resolve head commit: $head" >&2
  exit 2
}

if ! git -C "$repo_root" merge-base --is-ancestor "$base_sha" "$head_sha"; then
  echo "base commit is not an ancestor of head: $base_sha -> $head_sha" >&2
  exit 2
fi

diff_file=$(mktemp)
result_file=$(mktemp)
trap 'rm -f "$diff_file" "$result_file"' EXIT

git -C "$repo_root" diff --name-status -z -M -C --find-copies-harder \
  "$base_sha" "$head_sha" -- >"$diff_file"

while IFS= read -r -d '' status; do
  source_path=""
  destination_path=""
  case "$status" in
    A*)
      IFS= read -r -d '' destination_path || {
        echo "malformed added-path record" >&2
        exit 2
      }
      ;;
    C*|R*)
      IFS= read -r -d '' source_path || {
        echo "malformed source-path record for $status" >&2
        exit 2
      }
      IFS= read -r -d '' destination_path || {
        echo "malformed destination-path record for $status" >&2
        exit 2
      }
      ;;
    *)
      IFS= read -r -d '' source_path || {
        echo "malformed path record for $status" >&2
        exit 2
      }
      continue
      ;;
  esac

  if [[ "$destination_path" == *$'\n'* || "$destination_path" == *$'\r'* ]]; then
    echo "new-path gate rejects control characters in path names" >&2
    exit 2
  fi

  lowercase=${destination_path,,}
  [[ "$lowercase" == *.py ]] || continue

  if git -C "$repo_root" cat-file -e "$base_sha:$destination_path" 2>/dev/null; then
    continue
  fi
  if ! git -C "$repo_root" cat-file -e "$head_sha:$destination_path" 2>/dev/null; then
    echo "diff reports a new Python destination missing at head: $destination_path" >&2
    exit 2
  fi
  printf '%s\n' "$destination_path" >>"$result_file"
done <"$diff_file"

LC_ALL=C sort -u "$result_file" -o "$result_file"
if [[ -n "$output" ]]; then
  mkdir -p "$(dirname "$output")"
  cp "$result_file" "$output"
else
  cat "$result_file"
fi
