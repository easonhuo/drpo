#!/usr/bin/env bash
set -euo pipefail

root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
gate="$root/scripts/check_human_approval_record.sh"
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
paths="$tmp/paths.txt"
printf '%s\n' 'tests/test_example.py' >"$paths"

expect_pass() {
  local comments=$1
  shift
  bash "$gate" \
    --comments-json "$comments" \
    --approval-author easonhuo \
    --new-python-paths "$paths" \
    "$@" >/dev/null
}

expect_fail() {
  local comments=$1
  shift
  if bash "$gate" \
    --comments-json "$comments" \
    --approval-author easonhuo \
    --new-python-paths "$paths" \
    "$@" >/dev/null 2>&1; then
    echo "expected approval validation failure: $comments" >&2
    exit 1
  fi
}

cat >"$tmp/good.json" <<'JSON'
[
  {
    "user": {"login": "easonhuo"},
    "body": "DRPO-ORAL-APPROVAL: GOV-NEW-PYTHON-FILE-ORAL-APPROVAL-02\nDRPO-APPROVED-REASON: new_python_file\nDRPO-APPROVED-PATH: tests/test_example.py\nDRPO-RESPONSIBILITY: tests/test_example.py :: focused regression coverage\nDRPO-REUSE-RATIONALE: tests/test_example.py :: existing tests own different responsibilities"
  }
]
JSON
expect_pass "$tmp/good.json" --approval-reason new_python_file

cat >"$tmp/wrong-author.json" <<'JSON'
[
  {
    "user": {"login": "automation"},
    "body": "DRPO-ORAL-APPROVAL: GOV-NEW-PYTHON-FILE-ORAL-APPROVAL-02\nDRPO-APPROVED-REASON: new_python_file\nDRPO-APPROVED-PATH: tests/test_example.py\nDRPO-RESPONSIBILITY: tests/test_example.py :: focused regression coverage\nDRPO-REUSE-RATIONALE: tests/test_example.py :: existing tests own different responsibilities"
  }
]
JSON
expect_fail "$tmp/wrong-author.json" --approval-reason new_python_file

cat >"$tmp/missing-responsibility.json" <<'JSON'
[
  {
    "user": {"login": "easonhuo"},
    "body": "DRPO-ORAL-APPROVAL: GOV-NEW-PYTHON-FILE-ORAL-APPROVAL-02\nDRPO-APPROVED-REASON: new_python_file\nDRPO-APPROVED-PATH: tests/test_example.py\nDRPO-REUSE-RATIONALE: tests/test_example.py :: existing tests own different responsibilities"
  }
]
JSON
expect_fail "$tmp/missing-responsibility.json" --approval-reason new_python_file

cat >"$tmp/legacy.json" <<'JSON'
[
  {
    "user": {"login": "easonhuo"},
    "body": "Human approval record for `GOV-NEW-PYTHON-FILE-HUMAN-APPROVAL-01`: the user explicitly approved creation of exactly `tests/test_example.py`. Its responsibility is limited to focused regression coverage."
  }
]
JSON
expect_pass "$tmp/legacy.json" --approval-reason new_python_file

printf '%s\n' 'tests/test_example.py' 'tests/test_unapproved.py' >"$paths"
expect_fail "$tmp/good.json" --approval-reason new_python_file
printf '%s\n' 'tests/test_example.py' >"$paths"

cat >"$tmp/policy-good.json" <<'JSON'
[
  {
    "user": {"login": "easonhuo"},
    "body": "DRPO-ORAL-APPROVAL: GOV-NEW-PYTHON-FILE-ORAL-APPROVAL-02\nDRPO-APPROVED-REASON: hard_gate_policy_change\nDRPO-APPROVED-SCOPE: replace the duplicate Environment click with durable oral-approval verification"
  }
]
JSON
expect_pass "$tmp/policy-good.json" --approval-reason hard_gate_policy_change

cat >"$tmp/policy-missing-scope.json" <<'JSON'
[
  {
    "user": {"login": "easonhuo"},
    "body": "DRPO-ORAL-APPROVAL: GOV-NEW-PYTHON-FILE-ORAL-APPROVAL-02\nDRPO-APPROVED-REASON: hard_gate_policy_change"
  }
]
JSON
expect_fail "$tmp/policy-missing-scope.json" --approval-reason hard_gate_policy_change

cat >"$tmp/large-good.json" <<'JSON'
[
  {
    "user": {"login": "easonhuo"},
    "body": "DRPO-ORAL-APPROVAL: GOV-NEW-PYTHON-FILE-ORAL-APPROVAL-02\nDRPO-APPROVED-REASON: large_or_structural_python_change\nDRPO-APPROVED-SCOPE: approved bounded refactor of existing Python files"
  }
]
JSON
expect_pass "$tmp/large-good.json" --approval-reason large_or_structural_python_change

if bash "$gate" \
  --comments-json "$tmp/good.json" \
  --approval-author easonhuo \
  --new-python-paths "$paths" \
  --approval-reason unsupported >/dev/null 2>&1; then
  echo "unsupported reasons must fail closed" >&2
  exit 1
fi

printf '%s\n' '{}' >"$tmp/not-array.json"
expect_fail "$tmp/not-array.json" --approval-reason new_python_file

echo "human approval record gate tests: PASS"
