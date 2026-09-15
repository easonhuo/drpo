from __future__ import annotations

import ast
from pathlib import Path

MAIN = Path("src/drpo/e8_multitask_exp_tuning.py")
RUNTIME = Path("src/drpo/e8_multitask_runtime.py")
TESTS = Path("tests/test_e8_multitask_p0.py")


def replace_once(source: str, old: str, new: str, label: str) -> str:
    count = source.count(old)
    if count != 1:
        raise RuntimeError(f"{label}: expected exactly one match, found {count}")
    return source.replace(old, new, 1)


def remove_function(source: str, name: str) -> str:
    tree = ast.parse(source)
    node = next(
        (
            item
            for item in tree.body
            if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)) and item.name == name
        ),
        None,
    )
    if node is None or node.end_lineno is None:
        raise RuntimeError(f"Top-level function not found: {name}")
    lines = source.splitlines(keepends=True)
    start = node.lineno - 1
    end = node.end_lineno
    while end < len(lines) and lines[end].strip() == "":
        end += 1
    del lines[start:end]
    return "".join(lines)


runtime = RUNTIME.read_text(encoding="utf-8")

runtime = replace_once(
    runtime,
    "    artifact_path: Path | None,\n    read_json_fn: Callable[[Path], dict[str, Any]] = read_json_object,\n) -> bool:\n",
    "    artifact_path: Path | None,\n) -> bool:\n",
    "successful_attempt signature",
)
runtime = replace_once(
    runtime,
    '        provenance = read_json_fn(workload_root / "source_provenance.json")\n        prepare = read_json_fn(workload_root / "prepare_manifest.json")\n',
    '        provenance = read_json_object(workload_root / "source_provenance.json")\n        prepare = read_json_object(workload_root / "prepare_manifest.json")\n',
    "successful_attempt reader",
)
runtime = replace_once(
    runtime,
    "    sha256_fn: Callable[[Path], str],\n    read_json_fn: Callable[[Path], dict[str, Any]] = read_json_object,\n) -> tuple[dict[str, dict[str, Any]], dict[str, str]]:\n",
    "    sha256_fn: Callable[[Path], str],\n) -> tuple[dict[str, dict[str, Any]], dict[str, str]]:\n",
    "reusable signature",
)
runtime = replace_once(
    runtime,
    "            value = read_json_fn(manifest_path)\n",
    "            value = read_json_object(manifest_path)\n",
    "reusable reader",
)
runtime = replace_once(
    runtime,
    "    expected_cells: int,\n    load_prepared_fn: Callable[[Path, Mapping[str, Any]], Any],\n    require_calibration_fn: Callable[..., None],\n    require_liveness_fn: Callable[..., None],\n    reusable_cell_manifests_fn: Callable[\n        [Mapping[str, Any], Path],\n        tuple[dict[str, dict[str, Any]], dict[str, str]],\n    ],\n    read_json_fn: Callable[[Path], dict[str, Any]] = read_json_object,\n) -> dict[str, Any]:\n",
    "    expected_cells: int,\n    reusable: Mapping[str, Mapping[str, Any]],\n    rejected: Mapping[str, str],\n    load_prepared_fn: Callable[[Path, Mapping[str, Any]], Any],\n    require_calibration_fn: Callable[..., None],\n    require_liveness_fn: Callable[..., None],\n) -> dict[str, Any]:\n",
    "recovery_stage_plan signature",
)
runtime = replace_once(
    runtime,
    "    reusable, rejected = reusable_cell_manifests_fn(config, output_root)\n",
    "",
    "recovery_stage_plan callback removal",
)
for label, old, new in (
    (
        "aggregate reader",
        "int(read_json_fn(aggregate_path).get(\"cell_count\", 0))",
        "int(read_json_object(aggregate_path).get(\"cell_count\", 0))",
    ),
    (
        "audit reader",
        "read_json_fn(audit_path).get(\"all_training_and_evaluation_complete\")",
        "read_json_object(audit_path).get(\"all_training_and_evaluation_complete\")",
    ),
    (
        "complete reader",
        "read_json_fn(complete_path).get(\"complete\")",
        "read_json_object(complete_path).get(\"complete\")",
    ),
):
    runtime = replace_once(runtime, old, new, label)
runtime = replace_once(
    runtime,
    "    source_commit: str,\n    reusable: Mapping[str, Mapping[str, Any]],\n",
    "    source_commit: str,\n    schema_version: int,\n    reusable: Mapping[str, Mapping[str, Any]],\n",
    "checkpoint schema parameter",
)
runtime = replace_once(
    runtime,
    '    payload = {\n        "schema_version": 1,\n        "experiment_id": experiment_id_value,\n',
    '    payload = {\n        "schema_version": schema_version,\n        "experiment_id": experiment_id_value,\n',
    "checkpoint schema payload",
)
RUNTIME.write_text(runtime, encoding="utf-8")

main = MAIN.read_text(encoding="utf-8")
main = replace_once(
    main,
    "        artifact_path=artifact_path,\n        read_json_fn=_read_json_object,\n",
    "        artifact_path=artifact_path,\n",
    "main successful_attempt callback",
)
main = replace_once(
    main,
    "        engineering_self_test=_is_engineering_self_test(config),\n        sha256_fn=sha256_file,\n        read_json_fn=_read_json_object,\n",
    "        engineering_self_test=_is_engineering_self_test(config),\n        sha256_fn=sha256_file,\n",
    "main reusable callback",
)
main = replace_once(
    main,
    "    effective = _effective_recovery_config(config, output_root)\n    return e8_runtime.recovery_stage_plan(\n",
    "    effective = _effective_recovery_config(config, output_root)\n    reusable, rejected = _reusable_cell_manifests(effective, output_root)\n    return e8_runtime.recovery_stage_plan(\n",
    "main stage reusable materialization",
)
main = replace_once(
    main,
    "        expected_cells=len(build_cells(effective)),\n        load_prepared_fn=_load_prepared,\n        require_calibration_fn=_require_calibration_gate,\n        require_liveness_fn=_require_liveness_gate,\n        reusable_cell_manifests_fn=_reusable_cell_manifests,\n        read_json_fn=_read_json_object,\n",
    "        expected_cells=len(build_cells(effective)),\n        reusable=reusable,\n        rejected=rejected,\n        load_prepared_fn=_load_prepared,\n        require_calibration_fn=_require_calibration_gate,\n        require_liveness_fn=_require_liveness_gate,\n",
    "main stage callback cleanup",
)
main = remove_function(main, "_hardlink_file")
main = remove_function(main, "_replace_path_prefix")
if main.count("_hardlink_file(source, destination)") != 1:
    raise RuntimeError("Expected exactly one _hardlink_file call after wrapper removal")
main = main.replace(
    "_hardlink_file(source, destination)",
    "e8_runtime.hardlink_file(source, destination)",
    1,
)
if main.count("_replace_path_prefix(") != 2:
    raise RuntimeError("Expected exactly two _replace_path_prefix calls after wrapper removal")
main = main.replace("_replace_path_prefix(", "e8_runtime.replace_path_prefix(")
main = replace_once(
    main,
    "        source_commit=source_commit,\n        reusable=reusable,\n",
    "        source_commit=source_commit,\n        schema_version=RECOVERY_SNAPSHOT_SCHEMA_VERSION,\n        reusable=reusable,\n",
    "main checkpoint schema forwarding",
)
MAIN.write_text(main, encoding="utf-8")

tests = TESTS.read_text(encoding="utf-8")
marker = "def test_recovery_checkpoint_snapshot_uses_caller_schema_version"
if marker in tests:
    raise RuntimeError("Schema-version regression test already exists")

tests += r'''\n\ndef test_recovery_checkpoint_snapshot_uses_caller_schema_version(tmp_path: Path) -> None:\n    from drpo import e8_multitask_exp_tuning as exp_tuning\n    from drpo import e8_multitask_runtime as runtime\n\n    output_root = tmp_path / "run"\n    cell_root = output_root / "cells" / "cell-a"\n    cell_root.mkdir(parents=True)\n    exp_tuning.atomic_json(cell_root / "cell_manifest.json", {"complete": True})\n    snapshot_root = tmp_path / "snapshot"\n\n    payload = runtime.recovery_checkpoint_snapshot(\n        output_root,\n        snapshot_root,\n        source_commit="a" * 40,\n        schema_version=7,\n        reusable={\n            "cell-a": {\n                "canonical_output": str(cell_root / "summary.json"),\n                "terminal_adapter": str(cell_root / "terminal_adapter"),\n            }\n        },\n        rejected={},\n        experiment_id_value="DEV-SCHEMA-AUTHORITY",\n        config_hash="b" * 64,\n        expected_cells=1,\n        scientific_status="pilot",\n        sha256_fn=exp_tuning.sha256_file,\n        write_json=exp_tuning.atomic_json,\n    )\n\n    assert payload["schema_version"] == 7\n    stored = exp_tuning._read_json_object(snapshot_root / "RECOVERY_SNAPSHOT.json")\n    assert stored["schema_version"] == 7\n'''
TESTS.write_text(tests, encoding="utf-8")
