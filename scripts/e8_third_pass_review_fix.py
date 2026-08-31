#!/usr/bin/env python3
from __future__ import annotations

import textwrap
from pathlib import Path


def replace_once(path: Path, old: str, new: str) -> None:
    text = path.read_text(encoding="utf-8")
    count = text.count(old)
    if count != 1:
        raise SystemExit(
            f"{path}: expected one target, found {count}: {old[:100]!r}"
        )
    path.write_text(text.replace(old, new, 1), encoding="utf-8")


py = Path("src/drpo/e8_multitask_exp_tuning.py")
runner = Path("scripts/run_e8_multitask_exp_coldstart.sh")
bootstrap = Path("scripts/bootstrap_e8_multitask_exp_coldstart.sh")
tests = Path("tests/test_e8_multitask_p0.py")

# Recovery/resume must bind the exact canonical liveness identity, not only
# generic completion/finiteness flags.
marker = "def _require_liveness_gate(\n"
helper = '''def _matches_canonical_cold_liveness_manifest(
    result: Mapping[str, Any],
) -> bool:
    cell = result.get("cell")
    if not isinstance(cell, Mapping):
        return False
    expected = _canonical_cold_liveness_cell()
    try:
        rho = float(cell.get("rho"))
        coefficient = float(cell.get("lambda"))
    except (TypeError, ValueError):
        return False
    return (
        cell.get("task") == expected.task
        and cell.get("method") == expected.method
        and cell.get("seed") == expected.seed
        and cell.get("stage") == expected.stage
        and expected.rho is not None
        and math.isclose(rho, expected.rho, rel_tol=0.0, abs_tol=1.0e-15)
        and math.isclose(
            coefficient,
            COUNTDOWN_LIVENESS_COEFFICIENT,
            rel_tol=0.0,
            abs_tol=1.0e-15,
        )
        and result.get("canonical_dispatch")
        == "countdown_e8_alpha1_highc_scan_runtime.smoke"
    )


'''
text = py.read_text(encoding="utf-8")
if text.count(marker) != 1 or "def _matches_canonical_cold_liveness_manifest" in text:
    raise SystemExit("cannot insert canonical liveness manifest matcher")
py.write_text(text.replace(marker, helper + marker, 1), encoding="utf-8")

replace_once(
    py,
    '''        canonical_cold = (
            _is_coldstart(config)
            and result.get("canonical_dispatch_verified") is True
            and result.get("finite_old_core_updates") is True
''',
    '''        canonical_cold = (
            _is_coldstart(config)
            and _matches_canonical_cold_liveness_manifest(result)
            and result.get("canonical_dispatch_verified") is True
            and result.get("finite_old_core_updates") is True
''',
)

# The engineering self-test must produce the same exact liveness identity that
# the real recovery gate now requires.
replace_once(
    py,
    '''        "canonical_dispatch_verified": True,
        "finite_old_core_updates": True,
        "optimizer_update_norm": 1.0,
        "initial_adapter_weight_sha256": "0" * 64,
        "terminal_adapter_weight_sha256": "1" * 64,
        "cell": {"task": "countdown"},
''',
    '''        "canonical_dispatch": "countdown_e8_alpha1_highc_scan_runtime.smoke",
        "canonical_dispatch_verified": True,
        "finite_old_core_updates": True,
        "optimizer_update_norm": 1.0,
        "initial_adapter_weight_sha256": "0" * 64,
        "terminal_adapter_weight_sha256": "1" * 64,
        "cell": {
            "task": "countdown",
            "method": METHOD_EXPONENTIAL,
            "rho": math.exp(-COUNTDOWN_LIVENESS_COEFFICIENT),
            "lambda": COUNTDOWN_LIVENESS_COEFFICIENT,
            "seed": COUNTDOWN_LIVENESS_SEED_OFFSET,
            "stage": "liveness",
        },
''',
)

# Bootstrap cannot assume PyYAML is installed. Keep one tiny stdlib parser in
# runner/bootstrap and accept only the safe experiment-ID scalar forms that the
# Python config validator accepts: plain, single-quoted, or double-quoted,
# optionally followed by a YAML comment.
parser_function = '''read_config_experiment_id() {
  command -v python3 >/dev/null || return 127
  python3 - "$1" <<'PY_EXPERIMENT_ID'
from pathlib import Path
import re
import sys

safe = r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}"
plain = re.compile(rf"^({safe})(?:[ \\t]+#.*)?$")
single = re.compile(rf"^'({safe})'(?:[ \\t]+#.*)?$")
double = re.compile(rf'^"({safe})"(?:[ \\t]+#.*)?$')
values = []
malformed = False
for raw in Path(sys.argv[1]).read_text(encoding="utf-8").splitlines():
    if raw[:1].isspace() or not raw.startswith("experiment_id:"):
        continue
    rhs = raw.split(":", 1)[1].strip()
    match = plain.fullmatch(rhs) or single.fullmatch(rhs) or double.fullmatch(rhs)
    if match is None:
        malformed = True
    else:
        values.append(match.group(1))
if malformed or len(values) != 1:
    raise SystemExit(2)
print(values[0])
PY_EXPERIMENT_ID
}
'''

text = runner.read_text(encoding="utf-8")
anchor = "resolve_experiment_id() {\n"
if text.count(anchor) != 1 or "read_config_experiment_id()" in text:
    raise SystemExit("cannot insert runner config ID parser")
text = text.replace(anchor, parser_function + "\n" + anchor, 1)
old = '''resolve_experiment_id() {
  local matches=()
  mapfile -t matches < <(
    sed -nE 's/^experiment_id:[[:space:]]*([A-Za-z0-9][A-Za-z0-9._-]{0,127})[[:space:]]*$/\\1/p' "${CONFIG_PATH}"
  )
  [[ "${#matches[@]}" -eq 1 ]] || \\
    fail "config must contain exactly one well-formed top-level experiment_id: ${CONFIG_REPO_PATH}"
  local config_experiment_id="${matches[0]}"
'''
new = '''resolve_experiment_id() {
  local config_experiment_id
  config_experiment_id="$(read_config_experiment_id "${CONFIG_PATH}")" || \\
    fail "config must contain exactly one well-formed top-level experiment_id: ${CONFIG_REPO_PATH}"
'''
if text.count(old) != 1:
    raise SystemExit("cannot replace runner sed identity parser")
runner.write_text(text.replace(old, new, 1), encoding="utf-8")

text = bootstrap.read_text(encoding="utf-8")
insert_before = 'CURRENT_STAGE="resolve_config_identity"\n'
if text.count(insert_before) != 1 or "read_config_experiment_id()" in text:
    raise SystemExit("cannot insert bootstrap config ID parser")
text = text.replace(insert_before, parser_function + "\n" + insert_before, 1)
old = '''mapfile -t config_experiment_ids < <(
  sed -nE 's/^experiment_id:[[:space:]]*([A-Za-z0-9][A-Za-z0-9._-]{0,127})[[:space:]]*$/\\1/p' "${CONFIG_PATH}"
)
[[ "${#config_experiment_ids[@]}" -eq 1 ]] || \\
  fail "config must contain exactly one well-formed top-level experiment_id: ${CONFIG_REPO_PATH}"
CONFIG_EXPERIMENT_ID="${config_experiment_ids[0]}"
'''
new = '''CONFIG_EXPERIMENT_ID="$(read_config_experiment_id "${CONFIG_PATH}")" || \\
  fail "config must contain exactly one well-formed top-level experiment_id: ${CONFIG_REPO_PATH}"
'''
if text.count(old) != 1:
    raise SystemExit("cannot replace bootstrap sed identity parser")
bootstrap.write_text(text.replace(old, new, 1), encoding="utf-8")

# Direct regression coverage for stale liveness reuse.
text = tests.read_text(encoding="utf-8")
marker = "def test_canonical_coldstart_liveness_identity_is_independent_of_sweep_grid() -> None:\n"
extra = '''def test_canonical_liveness_manifest_matcher_rejects_stale_identity() -> None:
    from drpo import e8_multitask_exp_tuning as exp_tuning

    cell = exp_tuning._canonical_cold_liveness_cell()
    good = {
        "canonical_dispatch": "countdown_e8_alpha1_highc_scan_runtime.smoke",
        "cell": {
            "task": cell.task,
            "method": cell.method,
            "rho": cell.rho,
            "lambda": cell.lambda_value,
            "seed": cell.seed,
            "stage": cell.stage,
        },
    }
    assert exp_tuning._matches_canonical_cold_liveness_manifest(good)
    for key, value in (
        ("lambda", 0.916290732),
        ("rho", math.exp(-0.916290732)),
        ("seed", 5000),
        ("stage", "countdown_sentinel"),
        ("method", exp_tuning.METHOD_GLOBAL),
    ):
        stale = copy.deepcopy(good)
        stale["cell"][key] = value
        assert not exp_tuning._matches_canonical_cold_liveness_manifest(stale)
    stale = copy.deepcopy(good)
    stale["canonical_dispatch"] = "countdown_e8_alpha1_highc_scan_runtime.worker"
    assert not exp_tuning._matches_canonical_cold_liveness_manifest(stale)


'''
if text.count(marker) != 1 or "test_canonical_liveness_manifest_matcher_rejects_stale_identity" in text:
    raise SystemExit("cannot insert liveness manifest matcher test")
text = text.replace(marker, extra + marker, 1)

# Exercise the exact embedded parser rather than a separately reimplemented
# approximation. Import sys locally because the existing test module does not
# import it at module scope.
marker = "def test_runner_and_bootstrap_derive_experiment_id_from_config() -> None:\n"
extra = r'''def test_runner_and_bootstrap_config_id_parser_accepts_yaml_presentation(tmp_path: Path) -> None:
    import re
    import sys

    scripts = [
        Path("scripts/run_e8_multitask_exp_coldstart.sh").read_text(encoding="utf-8"),
        Path("scripts/bootstrap_e8_multitask_exp_coldstart.sh").read_text(encoding="utf-8"),
    ]
    parser_bodies = []
    for script in scripts:
        match = re.search(
            r"python3 - \"\$1\" <<'PY_EXPERIMENT_ID'\n(.*?)\nPY_EXPERIMENT_ID",
            script,
            flags=re.S,
        )
        assert match is not None
        parser_bodies.append(match.group(1))
    assert parser_bodies[0] == parser_bodies[1]

    valid = (
        "experiment_id: EXT-C-E8-PLAIN-01\n",
        "experiment_id: 'EXT-C-E8-SINGLE-01'\n",
        'experiment_id: "EXT-C-E8-DOUBLE-01"\n',
        "experiment_id: EXT-C-E8-COMMENT-01  # metadata\n",
        "experiment_id: 'EXT-C-E8-SINGLE-COMMENT-01' # metadata\n",
        'experiment_id: "EXT-C-E8-DOUBLE-COMMENT-01"\t# metadata\n',
    )
    for index, payload in enumerate(valid):
        config = tmp_path / f"valid-{index}.yaml"
        config.write_text(payload, encoding="utf-8")
        completed = subprocess.run(
            [sys.executable, "-c", parser_bodies[0], str(config)],
            check=False,
            capture_output=True,
            text=True,
        )
        assert completed.returncode == 0, completed.stderr
        assert completed.stdout.strip().startswith("EXT-C-E8-")

    invalid = (
        "experiment_id: bad id\n",
        "experiment_id: 'bad id'\n",
        "  experiment_id: NESTED-NOT-TOPLEVEL\n",
        "experiment_id: FIRST\nexperiment_id: SECOND\n",
        "experiment_id: `BAD`\n",
    )
    for index, payload in enumerate(invalid):
        config = tmp_path / f"invalid-{index}.yaml"
        config.write_text(payload, encoding="utf-8")
        completed = subprocess.run(
            [sys.executable, "-c", parser_bodies[0], str(config)],
            check=False,
            capture_output=True,
            text=True,
        )
        assert completed.returncode != 0


'''
extra = textwrap.dedent(extra)
if text.count(marker) != 1 or "test_runner_and_bootstrap_config_id_parser_accepts_yaml_presentation" in text:
    raise SystemExit("cannot insert config parser test")
text = text.replace(marker, extra + marker, 1)

# Two pre-existing assertions still described the old sed/mapfile parser. They
# should follow the config-authority invariant rather than pinning the removed
# implementation detail.
stale_assert = "    assert 'CONFIG_EXPERIMENT_ID=\"${config_experiment_ids[0]}\"' in bootstrap\n"
updated_assert = "    assert 'CONFIG_EXPERIMENT_ID=\"$(read_config_experiment_id \"${CONFIG_PATH}\")\"' in bootstrap\n"
if text.count(stale_assert) != 2:
    raise SystemExit(
        f"expected two stale bootstrap parser assertions, found {text.count(stale_assert)}"
    )
text = text.replace(stale_assert, updated_assert)

tests.write_text(text, encoding="utf-8")
