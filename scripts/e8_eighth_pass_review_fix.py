#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path


def replace_once(path: Path, old: str, new: str) -> None:
    text = path.read_text(encoding="utf-8")
    count = text.count(old)
    if count != 1:
        raise SystemExit(
            f"{path}: expected one target, found {count}: {old[:120]!r}"
        )
    path.write_text(text.replace(old, new, 1), encoding="utf-8")


bootstrap = Path("scripts/bootstrap_e8_multitask_exp_coldstart.sh")
tests = Path("tests/test_e8_multitask_p0.py")

replace_once(
    bootstrap,
    '''if [[ "${MODE}" == "full" ]]; then
  TARGET_REF="${E8_COLDSTART_TARGET_REF:-refs/heads/main}"
  git check-ref-format "${TARGET_REF}" >/dev/null 2>&1 || fail "invalid full target ref: ${TARGET_REF}"
  [[ "${TARGET_REF}" == refs/heads/* ]] || fail "full target ref must be under refs/heads/: ${TARGET_REF}"
  LOCAL_FETCH_REF="refs/e8-coldstart-bootstrap/full-target"
else
  TARGET_REF="refs/pull/309/head"
  LOCAL_FETCH_REF="refs/e8-coldstart-bootstrap/pr-309-head"
fi
''',
    '''resolve_target_ref() {
  if [[ "${MODE}" == "full" ]]; then
    TARGET_REF="${E8_COLDSTART_TARGET_REF:-refs/heads/main}"
    git check-ref-format "${TARGET_REF}" >/dev/null 2>&1 || \
      fail "invalid full target ref: ${TARGET_REF}"
    [[ "${TARGET_REF}" == refs/heads/* ]] || \
      fail "full target ref must be under refs/heads/: ${TARGET_REF}"
    LOCAL_FETCH_REF="refs/e8-coldstart-bootstrap/full-target"
    return
  fi

  TARGET_REF="${E8_COLDSTART_TARGET_REF:-}"
  [[ -n "${TARGET_REF}" ]] || \
    fail "self-test requires E8_COLDSTART_TARGET_REF"
  git check-ref-format "${TARGET_REF}" >/dev/null 2>&1 || \
    fail "invalid self-test target ref: ${TARGET_REF}"
  if [[ "${TARGET_REF}" == refs/heads/* ]]; then
    :
  elif [[ "${TARGET_REF}" =~ ^refs/pull/[1-9][0-9]*/head$ ]]; then
    :
  else
    fail "self-test target ref must be refs/heads/* or refs/pull/<number>/head: ${TARGET_REF}"
  fi
  LOCAL_FETCH_REF="refs/e8-coldstart-bootstrap/self-test-target"
}

resolve_target_ref
''',
)

text = tests.read_text(encoding="utf-8")
marker = '''def test_runner_and_bootstrap_derive_experiment_id_from_config() -> None:
'''
extra = r'''def test_bootstrap_self_test_target_ref_is_explicit_and_current() -> None:
    import re

    bootstrap = Path("scripts/bootstrap_e8_multitask_exp_coldstart.sh").read_text(
        encoding="utf-8"
    )
    assert "refs/pull/309/head" not in bootstrap
    assert "pr-309-head" not in bootstrap

    match = re.search(
        r"(resolve_target_ref\(\) \{.*?\n\})\n\nresolve_target_ref",
        bootstrap,
        flags=re.S,
    )
    assert match is not None
    function = match.group(1)

    def execute(*, mode: str, target: str | None) -> subprocess.CompletedProcess[str]:
        target_line = (
            "unset E8_COLDSTART_TARGET_REF"
            if target is None
            else f"E8_COLDSTART_TARGET_REF={target!r}"
        )
        script = f'''set -u
fail() {{ echo "ERROR: $*" >&2; exit 2; }}
MODE={mode!r}
{target_line}
TARGET_REF=""
LOCAL_FETCH_REF=""
{function}
resolve_target_ref
printf '%s\n' "$TARGET_REF" "$LOCAL_FETCH_REF"
'''
        return subprocess.run(
            ["bash", "-c", script],
            check=False,
            capture_output=True,
            text=True,
        )

    full_default = execute(mode="full", target=None)
    assert full_default.returncode == 0, full_default.stderr
    assert full_default.stdout.splitlines() == [
        "refs/heads/main",
        "refs/e8-coldstart-bootstrap/full-target",
    ]

    missing = execute(mode="self-test", target=None)
    assert missing.returncode == 2
    assert "self-test requires E8_COLDSTART_TARGET_REF" in missing.stderr

    branch = execute(
        mode="self-test",
        target="refs/heads/dev/e8-config-driven-sweep-01",
    )
    assert branch.returncode == 0, branch.stderr
    assert branch.stdout.splitlines() == [
        "refs/heads/dev/e8-config-driven-sweep-01",
        "refs/e8-coldstart-bootstrap/self-test-target",
    ]

    pr_head = execute(mode="self-test", target="refs/pull/340/head")
    assert pr_head.returncode == 0, pr_head.stderr
    assert pr_head.stdout.splitlines() == [
        "refs/pull/340/head",
        "refs/e8-coldstart-bootstrap/self-test-target",
    ]

    for rejected in (
        "refs/tags/v1",
        "refs/pull/0/head",
        "refs/pull/340/merge",
        "main",
    ):
        result = execute(mode="self-test", target=rejected)
        assert result.returncode == 2


'''
if text.count(marker) != 1 or "test_bootstrap_self_test_target_ref_is_explicit_and_current" in text:
    raise SystemExit("cannot insert eighth-pass bootstrap target test")
tests.write_text(text.replace(marker, extra + marker, 1), encoding="utf-8")
