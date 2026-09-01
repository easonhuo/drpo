#!/usr/bin/env bash
set -euo pipefail

python - <<'PY'
from pathlib import Path


def replace_once(text: str, old: str, new: str, label: str) -> str:
    if old not in text:
        raise SystemExit(f"missing anchor: {label}")
    return text.replace(old, new, 1)


path = Path('src/drpo/e8_multitask_exp_tuning.py')
text = path.read_text(encoding='utf-8')
text = replace_once(
    text,
    'COUNTDOWN_LIVENESS_SEED_OFFSET = 4000\n',
    'COUNTDOWN_LIVENESS_SEED_OFFSET = 4000\nEXECUTION_IDENTITY_ENV = "E8_COLDSTART_EXECUTION_IDENTITY_PATH"\n',
    'execution identity constant',
)
text = replace_once(
    text,
    '''def _repo_root() -> Path:\n    return Path(__file__).resolve().parents[2]\n\n\ndef _canonical_paths''',
    '''def _repo_root() -> Path:\n    return Path(__file__).resolve().parents[2]\n\n\ndef effective_execution_config_hash(config: Mapping[str, Any]) -> str:\n    """Hash the resolved cell/runtime plan that the cold-start runner will consume."""\n\n    cells = build_cells(config)\n    active_tasks = sorted({cell.task for cell in cells})\n    runtime = (\n        {\n            task: experiment_config.effective_coldstart_runtime(config, task)\n            for task in active_tasks\n        }\n        if _is_coldstart(config)\n        else {}\n    )\n    matrix = [\n        {\n            "cell_key": cell.key,\n            "task": cell.task,\n            "method": cell.method,\n            "rho": cell.rho,\n            "lambda": cell.lambda_value,\n            "seed": cell.seed,\n            "stage": cell.stage,\n        }\n        for cell in cells\n    ]\n    return stable_hash(\n        {\n            "semantic_config_hash": stable_config_hash(config),\n            "active_tasks": active_tasks,\n            "effective_runtime": runtime,\n            "matrix": matrix,\n        }\n    )\n\n\ndef _execution_identity_payload(*, required: bool = False) -> dict[str, Any] | None:\n    value = os.environ.get(EXECUTION_IDENTITY_ENV, "").strip()\n    if not value:\n        if required:\n            raise RuntimeError(f"{EXECUTION_IDENTITY_ENV} is required for cold-start execution")\n        return None\n    path = Path(value).resolve()\n    if not path.is_file():\n        raise FileNotFoundError(f"Execution identity file is missing: {path}")\n    payload = json.loads(path.read_text(encoding="utf-8"))\n    if not isinstance(payload, dict):\n        raise TypeError("Execution identity root must be a mapping")\n    recorded = str(payload.get("execution_identity_hash", ""))\n    unhashed = {key: item for key, item in payload.items() if key != "execution_identity_hash"}\n    if len(recorded) != 64 or stable_hash(unhashed) != recorded:\n        raise RuntimeError("Execution identity hash is missing or corrupt")\n    return payload\n\n\ndef _execution_identity_hash(*, required: bool = False) -> str | None:\n    payload = _execution_identity_payload(required=required)\n    return None if payload is None else str(payload["execution_identity_hash"])\n\n\ndef _validate_launch_execution_identity(\n    config_path: Path,\n    config: Mapping[str, Any],\n    *,\n    command: str,\n) -> dict[str, Any]:\n    """Bind direct Python execution to the runner-materialized reviewed identity."""\n\n    payload = _execution_identity_payload(required=True)\n    assert payload is not None\n    resolved, relative, blob = experiment_config.require_tracked_config(config_path, _repo_root())\n    del resolved\n    reviewed = payload.get("reviewed_config")\n    if not isinstance(reviewed, Mapping):\n        raise RuntimeError("Execution identity reviewed_config is malformed")\n    source_commit = subprocess.check_output(\n        ["git", "rev-parse", "HEAD^{commit}"], cwd=_repo_root(), text=True\n    ).strip()\n    expected = {\n        "experiment_id": experiment_id(config),\n        "repo_path": relative,\n        "git_blob_sha": blob,\n        "semantic_config_hash": stable_config_hash(config),\n        "effective_config_hash": effective_execution_config_hash(config),\n        "source_commit": source_commit,\n    }\n    observed = {\n        "experiment_id": payload.get("experiment_id"),\n        "repo_path": reviewed.get("repo_path"),\n        "git_blob_sha": reviewed.get("git_blob_sha"),\n        "semantic_config_hash": reviewed.get("semantic_config_hash"),\n        "effective_config_hash": payload.get("effective_config_hash"),\n        "source_commit": payload.get("source_commit"),\n    }\n    if observed != expected:\n        raise RuntimeError("Execution identity does not match the selected tracked config/source")\n    backend = payload.get("backend")\n    if backend == "engineering_placeholder":\n        allowed = {"engineering-self-test", "import-recovery", "recovery-plan"}\n        if command not in allowed:\n            raise RuntimeError(\n                f"engineering-placeholder identity may not execute scientific command {command}"\n            )\n    elif backend == "real_canonical":\n        if command == "engineering-self-test":\n            raise RuntimeError("real-canonical identity may not satisfy engineering self-test")\n    else:\n        raise RuntimeError(f"Unsupported execution identity backend: {backend}")\n    return payload\n\n\ndef _canonical_paths''',
    'execution identity helpers',
)
text = replace_once(
    text,
    '''        "config_hash": stable_config_hash(config),\n        "cell": {''',
    '''        "config_hash": stable_config_hash(config),\n        "execution_identity_hash": _execution_identity_hash(required=False),\n        "cell": {''',
    'cell identity propagation',
)
text = replace_once(
    text,
    '''    expected_hash = stable_config_hash(config)\n    expected_id = experiment_id(config)\n    for cell in build_cells(config):''',
    '''    expected_hash = stable_config_hash(config)\n    expected_id = experiment_id(config)\n    expected_execution = _execution_identity_hash(required=False)\n    for cell in build_cells(config):''',
    'reusable expected execution',
)
text = replace_once(
    text,
    '''                value.get("experiment_id") != expected_id\n                or value.get("config_hash") != expected_hash\n                or value.get("complete") is not True''',
    '''                value.get("experiment_id") != expected_id\n                or value.get("config_hash") != expected_hash\n                or (\n                    expected_execution is not None\n                    and value.get("execution_identity_hash") != expected_execution\n                )\n                or value.get("complete") is not True''',
    'reusable execution comparison',
)
text = replace_once(
    text,
    '''    config = _effective_recovery_config(config, output_root)\n    prepare_error: str | None = None''',
    '''    config = _effective_recovery_config(config, output_root)\n    execution_hash = _execution_identity_hash(required=False)\n    identity_error: str | None = None\n    if execution_hash is not None:\n        try:\n            provenance = _read_json_object(output_root / "source_provenance.json")\n            if provenance.get("execution_identity_hash") != execution_hash:\n                raise RuntimeError("source provenance execution identity mismatch")\n        except Exception as exc:\n            identity_error = f"{type(exc).__name__}: {exc}"\n    prepare_error: str | None = None''',
    'recovery plan identity precheck',
)
text = replace_once(
    text,
    '''    try:\n        _load_prepared(output_root, config)\n        prepare_complete = True''',
    '''    try:\n        if identity_error is not None:\n            raise RuntimeError(identity_error)\n        _load_prepared(output_root, config)\n        prepare_complete = True''',
    'recovery plan identity gate',
)
text = replace_once(
    text,
    '''        "config_hash": stable_config_hash(config),\n        "output_root": str(output_root.resolve()),''',
    '''        "config_hash": stable_config_hash(config),\n        "execution_identity_hash": execution_hash,\n        "output_root": str(output_root.resolve()),''',
    'recovery plan identity output',
)
text = replace_once(
    text,
    '''    provenance = _read_json_object(source_output_root / "source_provenance.json")\n    if provenance.get("source_commit") != source_commit:\n        raise RuntimeError("Recovery source commit does not match the reviewed execution commit")''',
    '''    provenance = _read_json_object(source_output_root / "source_provenance.json")\n    if provenance.get("source_commit") != source_commit:\n        raise RuntimeError("Recovery source commit does not match the reviewed execution commit")\n    current_execution = _execution_identity_hash(required=False)\n    if (\n        current_execution is not None\n        and provenance.get("execution_identity_hash") != current_execution\n    ):\n        raise RuntimeError("Recovery source belongs to another execution identity")''',
    'recovery import identity gate',
)
text = replace_once(
    text,
    '''        "source_commit": source_commit,\n        "source_output_root": source_text,''',
    '''        "source_commit": source_commit,\n        "execution_identity_hash": current_execution,\n        "source_output_root": source_text,''',
    'recovery import manifest identity',
)
text = replace_once(
    text,
    '''        "base_commit": source_commit,\n        "config_hash": stable_config_hash(config),\n        "output_root": str(output_root.resolve()),''',
    '''        "base_commit": source_commit,\n        "config_hash": stable_config_hash(config),\n        "execution_identity_hash": _execution_identity_hash(required=False),\n        "output_root": str(output_root.resolve()),''',
    'recovery snapshot identity',
)
text = replace_once(
    text,
    '''        "--source-file",\n        "src/drpo/e8_multitask_exp_tuning.py",\n    ]''',
    '''        "--source-file",\n        "src/drpo/e8_multitask_exp_tuning.py",\n        "--source-file",\n        "src/drpo/e8_experiment_config.py",\n        "--source-file",\n        "scripts/preflight_e8_multitask_config.py",\n    ]\n    identity = _execution_identity_payload(required=False)\n    if identity is not None:\n        command.extend(["--source-file", str(identity["reviewed_config"]["repo_path"])])''',
    'checkpoint provenance source closure',
)
text = replace_once(
    text,
    '''        "experiment_id": experiment_id(config),\n        "scheduler": "dynamic_slot_queue",''',
    '''        "experiment_id": experiment_id(config),\n        "execution_identity_hash": _execution_identity_hash(required=False),\n        "scheduler": "dynamic_slot_queue",''',
    'scheduler identity',
)
text = replace_once(
    text,
    '''        "source_commit": source_commit,\n        "task": task,\n        "expected_cells": len(rows),''',
    '''        "source_commit": source_commit,\n        "execution_identity_hash": _execution_identity_hash(required=False),\n        "task": task,\n        "expected_cells": len(rows),''',
    'task result identity',
)
text = replace_once(
    text,
    '''        "run_id": run_id,\n        "source_commit": source_commit,\n        "cell_count": len(rows),''',
    '''        "run_id": run_id,\n        "source_commit": source_commit,\n        "execution_identity_hash": _execution_identity_hash(required=False),\n        "cell_count": len(rows),''',
    'aggregate identity',
)
text = replace_once(
    text,
    '''    cells = build_cells(config)\n    missing: list[str] = []''',
    '''    cells = build_cells(config)\n    execution_hash = _execution_identity_hash(required=False)\n    if execution_hash is not None and provenance.get("execution_identity_hash") != execution_hash:\n        raise RuntimeError("Terminal audit source provenance execution identity mismatch")\n    missing: list[str] = []''',
    'audit provenance identity',
)
text = replace_once(
    text,
    '''        aggregate_complete = (\n            aggregate_path.is_file()\n            and int(json.loads(aggregate_path.read_text(encoding="utf-8")).get("cell_count", 0))\n            == len(cells)\n            and reproduction_gate_status == expected_protocol_status\n        )''',
    '''        aggregate_value = (\n            json.loads(aggregate_path.read_text(encoding="utf-8"))\n            if aggregate_path.is_file()\n            else {}\n        )\n        scheduler_path = output_root / "scheduler" / "dynamic_run.json"\n        scheduler_value = (\n            json.loads(scheduler_path.read_text(encoding="utf-8"))\n            if scheduler_path.is_file()\n            else {}\n        )\n        identity_consistent = execution_hash is None or (\n            aggregate_value.get("execution_identity_hash") == execution_hash\n            and scheduler_value.get("execution_identity_hash") == execution_hash\n        )\n        aggregate_complete = (\n            aggregate_path.is_file()\n            and scheduler_path.is_file()\n            and int(aggregate_value.get("cell_count", 0)) == len(cells)\n            and reproduction_gate_status == expected_protocol_status\n            and identity_consistent\n        )''',
    'audit terminal identity closure',
)
text = replace_once(
    text,
    '''        "base_commit": base_commit,\n        "expected_cells": len(cells),''',
    '''        "base_commit": base_commit,\n        "execution_identity_hash": execution_hash,\n        "expected_cells": len(cells),''',
    'audit identity output',
)
text = replace_once(
    text,
    '''    source_commit = str(provenance.get("source_commit", ""))\n    if len(source_commit) != 40 or any(char not in "0123456789abcdef" for char in source_commit):\n        raise RuntimeError("source_provenance.json must contain one full lowercase Git SHA")\n    expected_cells = len(build_cells(config))''',
    '''    source_commit = str(provenance.get("source_commit", ""))\n    if len(source_commit) != 40 or any(char not in "0123456789abcdef" for char in source_commit):\n        raise RuntimeError("source_provenance.json must contain one full lowercase Git SHA")\n    execution_hash = _execution_identity_hash(required=False)\n    if execution_hash is not None:\n        for label, value in (\n            ("source_provenance", provenance),\n            ("scheduler", scheduler),\n            ("aggregate", aggregate),\n            ("terminal_audit", audit),\n        ):\n            if value.get("execution_identity_hash") != execution_hash:\n                raise RuntimeError(f"{label} execution identity is not current")\n    expected_cells = len(build_cells(config))''',
    'completion current identity closure',
)
text = replace_once(
    text,
    '''        "config_hash": stable_config_hash(config),\n        "expected_cells": expected_cells,''',
    '''        "config_hash": stable_config_hash(config),\n        "execution_identity_hash": execution_hash,\n        "expected_cells": expected_cells,''',
    'run manifest identity',
)
text = replace_once(
    text,
    '''        "artifact_kind": (\n            "engineering_self_test" if _is_engineering_self_test(config) else "pilot_results"\n        ),''',
    '''        "artifact_kind": (\n            "engineering_self_test" if _is_engineering_self_test(config) else "pilot_results"\n        ),\n        "execution_identity_hash": _execution_identity_hash(required=False),''',
    'package identity',
)
text = replace_once(
    text,
    '''        "artifact_state": "raw_complete",\n        "canonical_archive_owner": "scripts/run_experiment_guard_hardened.py",''',
    '''        "artifact_state": "raw_complete",\n        "execution_identity_hash": _execution_identity_hash(required=False),\n        "canonical_archive_owner": "scripts/run_experiment_guard_hardened.py",''',
    'finalize identity',
)
text = replace_once(
    text,
    '''    output_root = validate_work_dir(output_root)\n    fresh_run = not (output_root / "prepare_manifest.json").is_file()''',
    '''    output_root = validate_work_dir(output_root)\n    execution_identity = _execution_identity_payload(required=False)\n    if execution_identity is not None:\n        atomic_json(output_root / "execution_identity.json", execution_identity)\n    fresh_run = not (output_root / "prepare_manifest.json").is_file()''',
    'engineering identity copy',
)
text = replace_once(
    text,
    '''                "source_commit": source_commit,\n                "model_repo": "engineering-placeholder-no-model-loaded",''',
    '''                "source_commit": source_commit,\n                "execution_identity_hash": _execution_identity_hash(required=False),\n                "model_repo": "engineering-placeholder-no-model-loaded",''',
    'engineering provenance identity',
)
text = replace_once(
    text,
    '''            "source_commit": source_commit,\n            "cell_key": cell.key,''',
    '''            "source_commit": source_commit,\n            "execution_identity_hash": _execution_identity_hash(required=False),\n            "cell_key": cell.key,''',
    'engineering placeholder cell identity',
)
text = replace_once(
    text,
    '''    config = load_config(config_path)\n    output_root = validate_work_dir(args.output_root)''',
    '''    config = load_config(config_path)\n    if _is_coldstart(config) and args.command != "plan":\n        _validate_launch_execution_identity(config_path, config, command=str(args.command))\n    output_root = validate_work_dir(args.output_root)''',
    'direct CLI identity gate',
)
path.write_text(text, encoding='utf-8')
PY

python - <<'PY'
from pathlib import Path


def replace_once(text: str, old: str, new: str, label: str) -> str:
    if old not in text:
        raise SystemExit(f"missing shell anchor: {label}")
    return text.replace(old, new, 1)


path = Path('scripts/run_e8_multitask_exp_coldstart.sh')
text = path.read_text(encoding='utf-8')
text = replace_once(
    text,
    '''  RECOVERY_PACKAGE="${RECOVERY_ROOT}/latest_checkpoint.zip"\n  DELIVERY_PREFLIGHT_PACKAGE="${RECOVERY_ROOT}/delivery_preflight.zip"\n  export RUN_ID ATTEMPTS_ROOT GUARD_ROOT OUTPUT_ROOT P0_WORK_DIR COUNTDOWN_WORK_DIR\n  export GUARD_ARTIFACT RECOVERY_ROOT RECOVERY_PACKAGE DELIVERY_PREFLIGHT_PACKAGE''',
    '''  RECOVERY_PACKAGE="${RECOVERY_ROOT}/latest_checkpoint.zip"\n  DELIVERY_PREFLIGHT_PACKAGE="${RECOVERY_ROOT}/delivery_preflight.zip"\n  EXECUTION_IDENTITY_PATH="${RECOVERY_ROOT}/EXECUTION_IDENTITY.json"\n  export RUN_ID ATTEMPTS_ROOT GUARD_ROOT OUTPUT_ROOT P0_WORK_DIR COUNTDOWN_WORK_DIR\n  export GUARD_ARTIFACT RECOVERY_ROOT RECOVERY_PACKAGE DELIVERY_PREFLIGHT_PACKAGE\n  export EXECUTION_IDENTITY_PATH\n  export E8_COLDSTART_EXECUTION_IDENTITY_PATH="${EXECUTION_IDENTITY_PATH}"''',
    'identity path',
)
identity_helpers = r'''
materialize_execution_identity() {
  local backend="$1"
  local run_class="$2"
  local model_root="$3"
  local action="$4"
  local python_bin="${VENV_DIR}/bin/python"
  if [[ "${backend}" == "engineering_placeholder" ]]; then
    python_bin="${SELFTEST_VENV_DIR}/bin/python"
  fi
  [[ -x "${python_bin}" ]] || fail "execution-identity Python is unavailable: ${python_bin}"
  mkdir -p "${RECOVERY_ROOT}"
  "${python_bin}" - "${ROOT_DIR}" "${CONFIG_PATH}" "${EXPECTED_COMMIT}" \
    "${MODEL_REPO}" "${MODEL_REVISION}" "${model_root}" "${backend}" "${run_class}" \
    "${EXECUTION_IDENTITY_PATH}" "${action}" <<'PY_IDENTITY'
import json
import platform
import sys
from pathlib import Path

from drpo import e8_experiment_config as experiment_config
from drpo import e8_multitask_exp_tuning as tuning
from drpo.e8_multitask_tasks import stable_hash

(
    root_text,
    config_text,
    source_commit,
    model_repo,
    model_revision,
    model_root_text,
    backend,
    run_class,
    identity_text,
    action,
) = sys.argv[1:]
root = Path(root_text).resolve()
config_path, relative, blob = experiment_config.require_tracked_config(config_text, root)
config = experiment_config.load_strict_yaml(config_path)
tuning.validate_config(config)
experiment_config.validate_historical_config_identity(config_path, config, repo_root=root)
semantic_hash = stable_hash(config)
effective_hash = tuning.effective_execution_config_hash(config)
if backend == "real_canonical":
    model_snapshot_hash = experiment_config.model_snapshot_identity(model_root_text)[
        "model_snapshot_hash"
    ]
    runtime = experiment_config.runtime_fingerprint()
elif backend == "engineering_placeholder":
    model_snapshot_hash = stable_hash(
        {
            "backend": backend,
            "source_commit": source_commit,
            "model_repo": "engineering-placeholder-no-model-loaded",
        }
    )
    runtime = {
        "python": platform.python_version(),
        "backend": "engineering_placeholder_non_gpu",
    }
else:
    raise SystemExit(f"unsupported backend: {backend}")
identity = experiment_config.execution_identity(
    reviewed_config_path=relative,
    reviewed_config_git_blob_sha=blob,
    reviewed_config_hash=semantic_hash,
    effective_config_hash=effective_hash,
    experiment_id_value=experiment_config.experiment_id(config),
    source_commit=source_commit,
    model_repo=model_repo if backend == "real_canonical" else "engineering-placeholder-no-model-loaded",
    model_revision=model_revision if backend == "real_canonical" else "not_applicable",
    model_snapshot_hash=model_snapshot_hash,
    runtime=runtime,
    backend=backend,
    run_class=run_class,
)
path = Path(identity_text)
if action == "write":
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(identity, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)
elif action == "verify":
    if not path.is_file():
        raise SystemExit("execution identity file is missing")
    observed = json.loads(path.read_text(encoding="utf-8"))
    if observed != identity:
        raise SystemExit("stored execution identity is stale")
else:
    raise SystemExit(f"unsupported execution identity action: {action}")
print(identity["execution_identity_hash"])
PY_IDENTITY
}

current_execution_identity_hash() {
  [[ -f "${EXECUTION_IDENTITY_PATH}" ]] || return 1
  python3 - "${EXECUTION_IDENTITY_PATH}" <<'PY_ID_HASH'
import json
import sys
value = json.loads(open(sys.argv[1], encoding="utf-8").read())
digest = str(value.get("execution_identity_hash", ""))
if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
    raise SystemExit(2)
print(digest)
PY_ID_HASH
}

workload_matches_execution_identity() {
  local workload="$1"
  local current
  current="$(current_execution_identity_hash)" || return 1
  python3 - "${workload}" "${current}" <<'PY_WORKLOAD_ID'
import json
import sys
from pathlib import Path
root = Path(sys.argv[1])
expected = sys.argv[2]
for name in ("source_provenance.json", "RUN_COMPLETE.json"):
    path = root / name
    if not path.is_file():
        if name == "RUN_COMPLETE.json":
            continue
        raise SystemExit(1)
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("execution_identity_hash") != expected:
        raise SystemExit(1)
raise SystemExit(0)
PY_WORKLOAD_ID
}
'''
text = replace_once(
    text,
    '''config_preflight() {\n  [[ -x "${VENV_DIR}/bin/python" ]] || fail "runtime Python is unavailable for config preflight"\n  "${VENV_DIR}/bin/python" "${ROOT_DIR}/scripts/preflight_e8_multitask_config.py" \\\n    --repo-root "${ROOT_DIR}" \\\n    --config "${CONFIG_PATH}"\n}\n\nsetup() {''',
    '''config_preflight() {\n  [[ -x "${VENV_DIR}/bin/python" ]] || fail "runtime Python is unavailable for config preflight"\n  "${VENV_DIR}/bin/python" "${ROOT_DIR}/scripts/preflight_e8_multitask_config.py" \\\n    --repo-root "${ROOT_DIR}" \\\n    --config "${CONFIG_PATH}"\n}\n''' + identity_helpers + '''\nsetup() {''',
    'identity helper insertion',
)
text = replace_once(
    text,
    '''  mkdir -p "${RECOVERY_ROOT}"\n  python - <<PY\nimport json\nfrom pathlib import Path\npath = Path("${RECOVERY_ROOT}") / "SETUP_COMPLETE.json"\npath.write_text(json.dumps({\n    "schema_version": 1,\n    "experiment_id": "${EXPERIMENT_ID}",\n    "source_commit": "${EXPECTED_COMMIT}",\n    "model_revision": "${MODEL_REVISION}",\n    "venv": str(Path("${VENV_DIR}").resolve()),\n    "model": str(Path("${MODEL_DIR}").resolve()),\n    "complete": True,\n}, indent=2, sort_keys=True) + "\\n", encoding="utf-8")\nPY''',
    '''  mkdir -p "${RECOVERY_ROOT}"\n  materialize_execution_identity real_canonical "${RUN_CLASS}" "${MODEL_DIR}" write >/dev/null\n  python - <<PY\nimport json\nfrom pathlib import Path\nidentity = json.loads(Path("${EXECUTION_IDENTITY_PATH}").read_text(encoding="utf-8"))\npath = Path("${RECOVERY_ROOT}") / "SETUP_COMPLETE.json"\npath.write_text(json.dumps({\n    "schema_version": 1,\n    "experiment_id": "${EXPERIMENT_ID}",\n    "source_commit": "${EXPECTED_COMMIT}",\n    "model_revision": "${MODEL_REVISION}",\n    "execution_identity_hash": identity["execution_identity_hash"],\n    "model_snapshot_hash": identity["model"]["model_snapshot_hash"],\n    "venv": str(Path("${VENV_DIR}").resolve()),\n    "model": str(Path("${MODEL_DIR}").resolve()),\n    "complete": True,\n}, indent=2, sort_keys=True) + "\\n", encoding="utf-8")\nPY''',
    'setup identity materialization',
)
text = replace_once(
    text,
    '''  preflight_gpu\n}\n\nensure_setup() {''',
    '''  materialize_execution_identity real_canonical "${RUN_CLASS}" "${MODEL_DIR}" verify >/dev/null || return 1\n  local current_identity\n  current_identity="$(current_execution_identity_hash)" || return 1\n  python - "${RECOVERY_ROOT}/SETUP_COMPLETE.json" "${current_identity}" <<'PY_SETUP_ID' >/dev/null\nimport json\nimport sys\nvalue = json.loads(open(sys.argv[1], encoding="utf-8").read())\nassert value.get("execution_identity_hash") == sys.argv[2]\nPY_SETUP_ID\n  preflight_gpu\n}\n\nensure_setup() {''',
    'runtime ready identity verification',
)
text = replace_once(
    text,
    '''    [[ -f "${attempt}/workload/source_provenance.json" ]] || continue\n    printf '%s\\n' "${attempt}/workload"''',
    '''    [[ -f "${attempt}/workload/source_provenance.json" ]] || continue\n    workload_matches_execution_identity "${attempt}/workload" || continue\n    printf '%s\\n' "${attempt}/workload"''',
    'recoverable execution identity',
)
text = replace_once(
    text,
    '''  [[ -f "${artifact}" && -f "${latest}/workload/aggregate/plot_curve_points.csv" ]] || return 1\n  python "${ROOT_DIR}/scripts/verify_experiment_package_hardened.py"''',
    '''  [[ -f "${artifact}" && -f "${latest}/workload/aggregate/plot_curve_points.csv" ]] || return 1\n  workload_matches_execution_identity "${latest}/workload" || return 1\n  python "${ROOT_DIR}/scripts/verify_experiment_package_hardened.py"''',
    'successful attempt execution identity',
)
text = replace_once(
    text,
    '''    printf 'source_commit=%q\\n' "${EXPECTED_COMMIT}"\n    printf 'guard_root=%q\\n' "${GUARD_ROOT}"''',
    '''    printf 'source_commit=%q\\n' "${EXPECTED_COMMIT}"\n    printf 'execution_identity_hash=%q\\n' "$(current_execution_identity_hash)"\n    printf 'guard_root=%q\\n' "${GUARD_ROOT}"''',
    'attempt state identity',
)
text = replace_once(
    text,
    '''  RECOVERY_ROOT="${RUNTIME_ROOT}/self-test-recovery/${RUN_ID}"\n  RECOVERY_PACKAGE="${RECOVERY_ROOT}/latest_checkpoint.zip"\n  command -v flock''',
    '''  RECOVERY_ROOT="${RUNTIME_ROOT}/self-test-recovery/${RUN_ID}"\n  RECOVERY_PACKAGE="${RECOVERY_ROOT}/latest_checkpoint.zip"\n  EXECUTION_IDENTITY_PATH="${RECOVERY_ROOT}/EXECUTION_IDENTITY.json"\n  export RECOVERY_ROOT RECOVERY_PACKAGE EXECUTION_IDENTITY_PATH\n  export E8_COLDSTART_EXECUTION_IDENTITY_PATH="${EXECUTION_IDENTITY_PATH}"\n  materialize_execution_identity engineering_placeholder pilot "placeholder" write >/dev/null\n  command -v flock''',
    'self-test identity',
)
text = replace_once(
    text,
    '''  run_module prepare \\\n    --p0-work-dir "${P0_WORK_DIR}" \\\n    --p0-config "${P0_CONFIG_PATH}" \\\n    --countdown-bank "${COUNTDOWN_WORK_DIR}/data/offline_bank_v2.jsonl" \\\n    --countdown-validation "${COUNTDOWN_WORK_DIR}/data/val.jsonl"\n  python - <<PY\nimport json\nfrom pathlib import Path\nvalue = {''',
    '''  run_module prepare \\\n    --p0-work-dir "${P0_WORK_DIR}" \\\n    --p0-config "${P0_CONFIG_PATH}" \\\n    --countdown-bank "${COUNTDOWN_WORK_DIR}/data/offline_bank_v2.jsonl" \\\n    --countdown-validation "${COUNTDOWN_WORK_DIR}/data/val.jsonl"\n  cp "${EXECUTION_IDENTITY_PATH}" "${OUTPUT_ROOT}/execution_identity.json"\n  python - <<PY\nimport json\nfrom pathlib import Path\nidentity = json.loads(Path("${EXECUTION_IDENTITY_PATH}").read_text(encoding="utf-8"))\nvalue = {''',
    'prepare identity copy',
)
text = replace_once(
    text,
    '''    "source_commit": "${EXPECTED_COMMIT}",\n    "model_repo": "${MODEL_REPO}",''',
    '''    "source_commit": "${EXPECTED_COMMIT}",\n    "execution_identity_hash": identity["execution_identity_hash"],\n    "model_repo": "${MODEL_REPO}",''',
    'formal provenance identity',
)
text = replace_once(
    text,
    '''    --source-file src/drpo/e8_multitask_exp_tuning.py \\\n    --source-file "${CONFIG_REPO_PATH}"''',
    '''    --source-file src/drpo/e8_multitask_exp_tuning.py \\\n    --source-file src/drpo/e8_experiment_config.py \\\n    --source-file scripts/preflight_e8_multitask_config.py \\\n    --source-file "${CONFIG_REPO_PATH}"''',
    'formal guard provenance closure',
)
text = replace_once(
    text,
    '''    src/drpo/e8_multitask_exp_tuning.py\n    src/drpo/e8_multitask_p0.py''',
    '''    src/drpo/e8_multitask_exp_tuning.py\n    src/drpo/e8_experiment_config.py\n    scripts/preflight_e8_multitask_config.py\n    src/drpo/e8_multitask_p0.py''',
    'registered protected paths closure',
)
path.write_text(text, encoding='utf-8')
PY

python - <<'PY'
from pathlib import Path

path = Path('tests/test_e8_multitask_p0.py')
text = path.read_text(encoding='utf-8')
text += r'''


def test_e8_execution_identity_distinguishes_backend_and_run_class() -> None:
    from drpo import e8_experiment_config as experiment_config

    base = dict(
        reviewed_config_path="configs/example.yaml",
        reviewed_config_git_blob_sha="1" * 40,
        reviewed_config_hash="semantic",
        effective_config_hash="effective",
        experiment_id_value="EXT-C-E8-EXAMPLE-01",
        source_commit="2" * 40,
        model_repo="repo/model",
        model_revision="rev",
        model_snapshot_hash="3" * 64,
        runtime={"python": "3.x"},
    )
    real_formal = experiment_config.execution_identity(
        **base, backend="real_canonical", run_class="formal"
    )
    real_pilot = experiment_config.execution_identity(
        **base, backend="real_canonical", run_class="pilot"
    )
    placeholder_pilot = experiment_config.execution_identity(
        **base, backend="engineering_placeholder", run_class="pilot"
    )
    assert real_formal["execution_identity_hash"] != real_pilot["execution_identity_hash"]
    assert real_pilot["execution_identity_hash"] != placeholder_pilot["execution_identity_hash"]


def test_e8_model_snapshot_hash_detects_weight_drift(tmp_path: Path) -> None:
    from drpo import e8_experiment_config as experiment_config

    root = tmp_path / "model"
    root.mkdir()
    weights = root / "model.safetensors"
    weights.write_bytes(b"first")
    (root / "config.json").write_text("{}")
    first = experiment_config.model_snapshot_identity(root)
    weights.write_bytes(b"second")
    second = experiment_config.model_snapshot_identity(root)
    assert first["model_snapshot_hash"] != second["model_snapshot_hash"]


def test_e8_reusable_cells_reject_other_execution_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from drpo import e8_multitask_exp_tuning as tuning

    config = tuning._engineering_self_test_config(
        tuning.load_config(Path("configs/e8_multitask_exp_coldstart.yaml"))
    )
    identity_path = tmp_path / "identity.json"
    payload = {
        "schema_version": 1,
        "experiment_id": tuning.experiment_id(config),
        "reviewed_config": {},
        "effective_config_hash": "effective",
        "source_commit": "1" * 40,
        "model": {},
        "runtime": {},
        "backend": "engineering_placeholder",
        "run_class": "pilot",
    }
    payload["execution_identity_hash"] = tuning.stable_hash(payload)
    identity_path.write_text(json.dumps(payload))
    monkeypatch.setenv(tuning.EXECUTION_IDENTITY_ENV, str(identity_path))
    cell = tuning.build_cells(config)[0]
    root = tmp_path / "run"
    manifest = root / "cells" / cell.key / "cell_manifest.json"
    manifest.parent.mkdir(parents=True)
    manifest.write_text(
        json.dumps(
            {
                "experiment_id": tuning.experiment_id(config),
                "config_hash": tuning.stable_config_hash(config),
                "execution_identity_hash": "0" * 64,
                "complete": True,
                "evaluation_status": "complete",
                "nan_inf_failure": False,
                "engineering_placeholder_backend": True,
            }
        )
    )
    reusable, rejected = tuning._reusable_cell_manifests(config, root)
    assert cell.key not in reusable
    assert "mismatch" in rejected[cell.key]


def test_e8_recovery_import_rejects_stale_execution_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from drpo import e8_multitask_exp_tuning as tuning

    config = tuning._engineering_self_test_config(
        tuning.load_config(Path("configs/e8_multitask_exp_coldstart.yaml"))
    )
    identity_path = tmp_path / "identity.json"
    payload = {
        "schema_version": 1,
        "experiment_id": tuning.experiment_id(config),
        "reviewed_config": {},
        "effective_config_hash": "effective",
        "source_commit": "1" * 40,
        "model": {},
        "runtime": {},
        "backend": "engineering_placeholder",
        "run_class": "pilot",
    }
    payload["execution_identity_hash"] = tuning.stable_hash(payload)
    identity_path.write_text(json.dumps(payload))
    monkeypatch.setenv(tuning.EXECUTION_IDENTITY_ENV, str(identity_path))
    source = tmp_path / "source"
    source.mkdir()
    (source / "source_provenance.json").write_text(
        json.dumps({"source_commit": "1" * 40, "execution_identity_hash": "0" * 64})
    )
    with pytest.raises(RuntimeError, match="another execution identity"):
        tuning.cmd_import_recovery(
            config,
            tmp_path / "destination",
            source_output_root=source,
            base_model_path="placeholder",
            source_commit="1" * 40,
        )


def test_e8_runner_binds_execution_identity_and_complete_provenance() -> None:
    runner = Path("scripts/run_e8_multitask_exp_coldstart.sh").read_text(encoding="utf-8")
    assert "E8_COLDSTART_EXECUTION_IDENTITY_PATH" in runner
    assert "model_snapshot_identity" in runner
    assert "runtime_fingerprint" in runner
    assert "workload_matches_execution_identity" in runner
    assert "--source-file src/drpo/e8_experiment_config.py" in runner
    assert "--source-file scripts/preflight_e8_multitask_config.py" in runner


def test_e8_bootstrap_refreshes_authoritative_ref_even_when_complete() -> None:
    bootstrap = Path("scripts/bootstrap_e8_multitask_exp_coldstart.sh").read_text(encoding="utf-8")
    fetch_index = bootstrap.index('CURRENT_STAGE="fetch_authoritative_ref"')
    stale_index = bootstrap.index('completed bootstrap is stale: authoritative')
    assert fetch_index < stale_index
    assert 'if [[ "${BOOTSTRAP_WAS_COMPLETE}" -eq 1 ]]; then\n  TARGET_COMMIT=' not in bootstrap
'''
path.write_text(text, encoding='utf-8')
PY

ruff format src/drpo/e8_experiment_config.py scripts/preflight_e8_multitask_config.py tests/test_e8_multitask_p0.py
python -m py_compile src/drpo/e8_experiment_config.py scripts/preflight_e8_multitask_config.py src/drpo/e8_multitask_exp_tuning.py tests/test_e8_multitask_p0.py
bash -n scripts/run_e8_multitask_exp_coldstart.sh
bash -n scripts/bootstrap_e8_multitask_exp_coldstart.sh
bash -n scripts/run_e8_multitask_exp_lambda_completion.sh
python -m pytest -q tests/test_e8_multitask_p0.py
ruff check src/drpo/e8_experiment_config.py scripts/preflight_e8_multitask_config.py tests/test_e8_multitask_p0.py
ruff format --check src/drpo/e8_experiment_config.py scripts/preflight_e8_multitask_config.py tests/test_e8_multitask_p0.py
git diff --check

git rm .github/workflows/e8-third-audit-identity-fix-once.yml scripts/.tmp_e8_third_audit_identity_fix.sh
git add scripts/run_e8_multitask_exp_coldstart.sh src/drpo/e8_multitask_exp_tuning.py tests/test_e8_multitask_p0.py
git diff --cached --check
git config user.name github-actions[bot]
git config user.email 41898282+github-actions[bot]@users.noreply.github.com
git commit -m 'fix(e8): bind execution identity end to end'
git push origin HEAD:dev/e8-config-driven-sweep-01
