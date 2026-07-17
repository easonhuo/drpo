#!/usr/bin/env bash
set -euo pipefail

root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
gate="$root/scripts/check_new_python_file_gate.sh"
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
repo="$tmp/repo"

git init -q "$repo"
git -C "$repo" config user.name test
git -C "$repo" config user.email test@example.com
mkdir -p "$repo/pkg"
printf 'value = 1\n' >"$repo/pkg/existing.py"
printf 'text\n' >"$repo/note.txt"
git -C "$repo" add .
git -C "$repo" commit -qm base
base=$(git -C "$repo" rev-parse HEAD)

assert_paths() {
  local expected=$1
  local head=$2
  local output="$tmp/output.txt"
  bash "$gate" --repo-root "$repo" --base "$base" --head "$head" --output "$output"
  actual=$(cat "$output")
  if [[ "$actual" != "$expected" ]]; then
    printf 'expected:\n%s\nactual:\n%s\n' "$expected" "$actual" >&2
    exit 1
  fi
}

printf 'value = 2\n' >"$repo/pkg/existing.py"
git -C "$repo" add .
git -C "$repo" commit -qm modify-existing
assert_paths "" "$(git -C "$repo" rev-parse HEAD)"
git -C "$repo" reset -q --hard "$base"

printf 'new = 1\n' >"$repo/pkg/new_file.py"
git -C "$repo" add .
git -C "$repo" commit -qm add-python
assert_paths "pkg/new_file.py" "$(git -C "$repo" rev-parse HEAD)"
git -C "$repo" reset -q --hard "$base"

printf 'upper = 1\n' >"$repo/pkg/UPPER.PY"
git -C "$repo" add .
git -C "$repo" commit -qm add-upper-python
assert_paths "pkg/UPPER.PY" "$(git -C "$repo" rev-parse HEAD)"
git -C "$repo" reset -q --hard "$base"

git -C "$repo" mv note.txt pkg/renamed.py
git -C "$repo" commit -qm rename-to-python
assert_paths "pkg/renamed.py" "$(git -C "$repo" rev-parse HEAD)"
git -C "$repo" reset -q --hard "$base"

cp "$repo/pkg/existing.py" "$repo/pkg/copied.py"
git -C "$repo" add .
git -C "$repo" commit -qm copy-python
assert_paths "pkg/copied.py" "$(git -C "$repo" rev-parse HEAD)"
git -C "$repo" reset -q --hard "$base"

git -C "$repo" rm -q pkg/existing.py
git -C "$repo" commit -qm delete-python
assert_paths "" "$(git -C "$repo" rev-parse HEAD)"

if bash "$gate" --repo-root "$repo" --base deadbeef --head HEAD >/dev/null 2>&1; then
  echo "unresolved base must fail closed" >&2
  exit 1
fi

echo "new Python file gate tests: PASS"
