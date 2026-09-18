"""Method-agnostic recovery primitives for E8 multitask experiments."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import tarfile
import time
import zipfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol


class CellLike(Protocol):
    @property
    def key(self) -> str: ...

    task: str
    method: str
    seed: int
    stage: str


@dataclass(frozen=True)
class MethodAuditResult:
    passed: bool
    failure_bucket: str = "terminal_contract_failures"
    failures: tuple[str, ...] = ()


def stable_json_hash(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def recovery_identity(
    cell: CellLike,
    *,
    experiment_id: str,
    config_hash: str,
    cell_identity_fields: Mapping[str, Any],
    common_identity_fields: Mapping[str, Any],
    schema_version: int = 1,
) -> dict[str, Any]:
    cell_identity: dict[str, Any] = {
        "task": cell.task,
        "method": cell.method,
    }
    collisions = sorted(set(cell_identity).intersection(cell_identity_fields))
    if collisions:
        raise ValueError(
            f"Method cell identity attempted to overwrite: {collisions}"
        )
    cell_identity.update(cell_identity_fields)
    for reserved, expected in (
        ("seed", int(cell.seed)),
        ("stage", cell.stage),
    ):
        if reserved in cell_identity:
            raise ValueError(
                f"Method cell identity attempted to overwrite: ['{reserved}']"
            )
        cell_identity[reserved] = expected

    identity: dict[str, Any] = {
        "schema_version": int(schema_version),
        "experiment_id": experiment_id,
        "config_hash": config_hash,
        "cell": cell_identity,
    }
    collisions = sorted(set(identity).intersection(common_identity_fields))
    if collisions:
        raise ValueError(
            f"Common recovery identity attempted to overwrite: {collisions}"
        )
    identity.update(common_identity_fields)
    identity["identity_hash"] = stable_json_hash(identity)
    return identity

# Recovery/runtime lifecycle ownership

def read_json_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"Expected one JSON object: {path}")
    return value


def successful_attempt_matches_current_identity(
    workload_root: Path,
    *,
    source_commit: str,
    experiment_id_value: str,
    config_hash: str,
    artifact_path: Path | None,
) -> bool:
    """Return whether live and packaged completed evidence matches this invocation."""

    try:
        provenance = read_json_object(workload_root / "source_provenance.json")
        prepare = read_json_object(workload_root / "prepare_manifest.json")
        if not (
            provenance.get("source_commit") == source_commit
            and prepare.get("experiment_id") == experiment_id_value
            and prepare.get("config_hash") == config_hash
        ):
            return False
        if artifact_path is None:
            return True
        prefix = f"results/{experiment_id_value}"
        with zipfile.ZipFile(artifact_path) as archive:
            artifact_manifest = json.loads(archive.read("ARTIFACT_MANIFEST.json"))
            base_commit = archive.read("BASE_COMMIT.txt").decode("utf-8").strip()
            run_manifest = json.loads(archive.read(f"{prefix}/run_manifest.json"))
            packaged_provenance = json.loads(
                archive.read(f"{prefix}/workload/source_provenance.json")
            )
            packaged_prepare = json.loads(
                archive.read(f"{prefix}/workload/prepare_manifest.json")
            )
    except (
        OSError,
        ValueError,
        TypeError,
        KeyError,
        UnicodeDecodeError,
        zipfile.BadZipFile,
    ):
        return False
    if not all(
        isinstance(value, dict)
        for value in (
            artifact_manifest,
            run_manifest,
            packaged_provenance,
            packaged_prepare,
        )
    ):
        return False
    return (
        artifact_manifest.get("package_kind") == "experiment-raw-complete"
        and artifact_manifest.get("experiment_id") == experiment_id_value
        and artifact_manifest.get("base_commit") == source_commit
        and base_commit == source_commit
        and run_manifest.get("experiment_id") == experiment_id_value
        and run_manifest.get("base_commit") == source_commit
        and packaged_provenance.get("source_commit") == source_commit
        and packaged_prepare.get("experiment_id") == experiment_id_value
        and packaged_prepare.get("config_hash") == config_hash
    )


def effective_recovery_config(
    config: Mapping[str, Any],
    output_root: Path,
    *,
    load_config_fn: Callable[[Path], dict[str, Any]],
    is_engineering_self_test_fn: Callable[[Mapping[str, Any]], bool],
) -> dict[str, Any]:
    engineering_config = output_root / "engineering_self_test_config.yaml"
    if engineering_config.is_file():
        recovered = load_config_fn(engineering_config)
        if not is_engineering_self_test_fn(recovered):
            raise RuntimeError("Recovery engineering config is not a placeholder config")
        return recovered
    return dict(config)


def reusable_cell_manifests(
    output_root: Path,
    *,
    cells: Sequence[CellLike],
    experiment_id_value: str,
    config_hash: str,
    engineering_self_test: bool,
    sha256_fn: Callable[[Path], str],
) -> tuple[dict[str, dict[str, Any]], dict[str, str]]:
    reusable: dict[str, dict[str, Any]] = {}
    rejected: dict[str, str] = {}
    for cell in cells:
        manifest_path = output_root / "cells" / cell.key / "cell_manifest.json"
        if not manifest_path.is_file():
            rejected[cell.key] = "missing_cell_manifest"
            continue
        try:
            value = read_json_object(manifest_path)
            if (
                value.get("experiment_id") != experiment_id_value
                or value.get("config_hash") != config_hash
                or value.get("complete") is not True
                or value.get("evaluation_status") != "complete"
                or value.get("nan_inf_failure") is not False
            ):
                raise RuntimeError("identity_or_completion_fields_mismatch")
            if engineering_self_test:
                if value.get("engineering_placeholder_backend") is not True:
                    raise RuntimeError("placeholder_backend_marker_missing")
            else:
                if (
                    not isinstance(value.get("identity_hash"), str)
                    or len(str(value["identity_hash"])) != 64
                    or value.get("canonical_dispatch_verified") is not True
                ):
                    raise RuntimeError("canonical_identity_or_dispatch_marker_missing")
                summary = Path(str(value.get("canonical_summary", "")))
                expected_summary_hash = str(value.get("canonical_summary_sha256", ""))
                if (
                    not summary.is_file()
                    or len(expected_summary_hash) != 64
                    or sha256_fn(summary) != expected_summary_hash
                ):
                    raise RuntimeError("canonical_summary_missing_or_corrupt")
                for field in ("best_adapter", "terminal_adapter"):
                    adapter = Path(str(value.get(field, "")))
                    if not (adapter / "adapter_config.json").is_file() or not any(
                        (adapter / name).is_file()
                        for name in ("adapter_model.safetensors", "adapter_model.bin")
                    ):
                        raise RuntimeError(f"{field}_missing_or_incomplete")
            reusable[cell.key] = value
        except (OSError, ValueError, TypeError, RuntimeError, json.JSONDecodeError) as exc:
            rejected[cell.key] = f"{type(exc).__name__}: {exc}"
    return reusable, rejected


def recovery_stage_plan(
    config: Mapping[str, Any],
    output_root: Path,
    *,
    base_model_path: str,
    schema_version: int,
    experiment_id_value: str,
    config_hash: str,
    expected_cells: int,
    reusable: Mapping[str, Mapping[str, Any]],
    rejected: Mapping[str, str],
    load_prepared_fn: Callable[[Path, Mapping[str, Any]], Any],
    require_calibration_fn: Callable[..., None],
    require_liveness_fn: Callable[..., None],
) -> dict[str, Any]:
    prepare_error: str | None = None
    calibration_error: str | None = None
    liveness_error: str | None = None
    try:
        load_prepared_fn(output_root, config)
        prepare_complete = True
    except Exception as exc:  # noqa: BLE001 - fail-closed reason capture
        prepare_complete = False
        prepare_error = f"{type(exc).__name__}: {exc}"
    if prepare_complete:
        try:
            require_calibration_fn(config, output_root, base_model_path=base_model_path)
            calibration_complete = True
        except Exception as exc:  # noqa: BLE001 - fail-closed reason capture
            calibration_complete = False
            calibration_error = f"{type(exc).__name__}: {exc}"
    else:
        calibration_complete = False
        calibration_error = "prepare_incomplete"
    if calibration_complete:
        try:
            require_liveness_fn(config, output_root, base_model_path=base_model_path)
            liveness_complete = True
        except Exception as exc:  # noqa: BLE001 - fail-closed reason capture
            liveness_complete = False
            liveness_error = f"{type(exc).__name__}: {exc}"
    else:
        liveness_complete = False
        liveness_error = "calibration_incomplete"
    cells_complete = len(reusable) == expected_cells
    aggregate_path = output_root / "aggregate" / "aggregate_summary.json"
    aggregate_complete = False
    if aggregate_path.is_file():
        try:
            aggregate_complete = int(read_json_object(aggregate_path).get("cell_count", 0)) == (
                expected_cells
            )
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            aggregate_complete = False
    audit_path = output_root / "terminal_audit.json"
    audit_complete = False
    if audit_path.is_file():
        try:
            audit_complete = bool(
                read_json_object(audit_path).get("all_training_and_evaluation_complete")
            )
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            audit_complete = False
    finalized = False
    complete_path = output_root / "RUN_COMPLETE.json"
    if complete_path.is_file():
        try:
            finalized = bool(read_json_object(complete_path).get("complete")) and audit_complete
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            finalized = False
    if not prepare_complete:
        next_stage = "prepare"
    elif not calibration_complete:
        next_stage = "calibrate"
    elif not liveness_complete:
        next_stage = "liveness"
    elif not cells_complete:
        next_stage = "run_queue"
    elif not aggregate_complete:
        next_stage = "aggregate"
    elif not audit_complete:
        next_stage = "audit"
    elif not finalized:
        next_stage = "finalize"
    else:
        next_stage = "delivery_preflight"
    return {
        "schema_version": schema_version,
        "experiment_id": experiment_id_value,
        "config_hash": config_hash,
        "output_root": str(output_root.resolve()),
        "prepare_complete": prepare_complete,
        "prepare_error": prepare_error,
        "calibration_complete": calibration_complete,
        "calibration_error": calibration_error,
        "liveness_complete": liveness_complete,
        "liveness_error": liveness_error,
        "expected_cells": expected_cells,
        "reusable_completed_cells": len(reusable),
        "reusable_cell_keys": sorted(reusable),
        "rejected_cells": rejected,
        "cells_complete": cells_complete,
        "aggregate_complete": aggregate_complete,
        "audit_complete": audit_complete,
        "finalized": finalized,
        "next_stage": next_stage,
        "intra_cell_resume_supported": False,
        "intra_cell_resume_reason": (
            "locked canonical kernels do not persist complete optimizer, scheduler, RNG, "
            "and dataloader state"
        ),
    }


def hardlink_file(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, destination)
    except OSError as exc:
        raise RuntimeError(
            "Recovery requires source and destination on one hard-link-capable persistent "
            f"filesystem; could not link {source} -> {destination}: {exc}"
        ) from exc


def replace_path_prefix(value: Any, source: str, destination: str) -> Any:
    if isinstance(value, str) and (value == source or value.startswith(source + os.sep)):
        return destination + value[len(source) :]
    if isinstance(value, list):
        return [replace_path_prefix(item, source, destination) for item in value]
    if isinstance(value, dict):
        return {key: replace_path_prefix(item, source, destination) for key, item in value.items()}
    return value


def recovery_checkpoint_snapshot(
    output_root: Path,
    snapshot_root: Path,
    *,
    source_commit: str,
    schema_version: int,
    reusable: Mapping[str, Mapping[str, Any]],
    rejected: Mapping[str, str],
    experiment_id_value: str,
    config_hash: str,
    expected_cells: int,
    scientific_status: str,
    sha256_fn: Callable[[Path], str],
    write_json: Callable[[Path, Any], None],
) -> dict[str, Any]:
    temporary = snapshot_root.with_name(
        f".{snapshot_root.name}.tmp-{os.getpid()}-{time.time_ns()}"
    )
    if temporary.exists():
        shutil.rmtree(temporary)
    (temporary / "logs").mkdir(parents=True)
    (temporary / "cell_manifests").mkdir(parents=True)
    cells: list[dict[str, Any]] = []
    for key, manifest in sorted(reusable.items()):
        source = output_root / "cells" / key / "cell_manifest.json"
        destination = temporary / "cell_manifests" / f"{key}.json"
        shutil.copy2(source, destination)
        cells.append(
            {
                "cell_key": key,
                "manifest_sha256": sha256_fn(source),
                "manifest_path": str(source.resolve()),
                "canonical_output": manifest.get("canonical_output"),
                "terminal_adapter": manifest.get("terminal_adapter"),
            }
        )
    payload = {
        "schema_version": schema_version,
        "experiment_id": experiment_id_value,
        "base_commit": source_commit,
        "config_hash": config_hash,
        "output_root": str(output_root.resolve()),
        "expected_cells": expected_cells,
        "completed_cells": len(cells),
        "cells": cells,
        "rejected_cells": dict(rejected),
        "recovery_semantics": "reuse complete identity-checked cells; rerun incomplete cells",
        "intra_cell_resume_supported": False,
        "scientific_status": scientific_status,
    }
    write_json(temporary / "RECOVERY_SNAPSHOT.json", payload)
    write_json(
        temporary / "run_manifest.json",
        {
            "schema_version": 1,
            "experiment_id": experiment_id_value,
            "base_commit": source_commit,
            "run_id": output_root.parent.name,
            "execution_state": "checkpoint",
            "artifact_state": "checkpoint",
            "completed_cells": len(cells),
            "expected_cells": expected_cells,
            "scientific_status": payload["scientific_status"],
        },
    )
    (temporary / "logs" / "recovery_checkpoint.log").write_text(
        f"completed_cells={len(cells)} expected_cells={expected_cells}\n",
        encoding="utf-8",
    )
    if snapshot_root.exists():
        shutil.rmtree(snapshot_root)
    os.replace(temporary, snapshot_root)
    return payload




def import_recovery(
    config: Mapping[str, Any],
    output_root: Path,
    *,
    source_output_root: Path,
    base_model_path: str,
    source_commit: str,
    schema_version: int,
    transient_top_level: set[str],
    transient_files: set[str],
    effective_config_fn: Callable[[Mapping[str, Any], Path], dict[str, Any]],
    recovery_stage_plan_fn: Callable[..., Mapping[str, Any]],
    experiment_id_fn: Callable[[Mapping[str, Any]], str],
    sha256_fn: Callable[[Path], str],
    write_json: Callable[[Path, Any], None],
) -> dict[str, Any]:
    """Import identity-checked reusable cells from a prior output tree."""

    source_output_root = source_output_root.resolve()
    output_root = output_root.resolve()
    if source_output_root == output_root:
        raise ValueError("Recovery source and destination must differ")
    if not source_output_root.is_dir():
        raise FileNotFoundError(f"Recovery source does not exist: {source_output_root}")
    if output_root.exists() and any(output_root.iterdir()):
        raise RuntimeError("Recovery destination must be new and empty")
    if len(source_commit) != 40 or any(
        character not in "0123456789abcdef" for character in source_commit
    ):
        raise ValueError("Recovery import requires one full lowercase source commit")

    provenance = read_json_object(source_output_root / "source_provenance.json")
    if provenance.get("source_commit") != source_commit:
        raise RuntimeError(
            "Recovery source commit does not match the reviewed execution commit"
        )
    output_root.mkdir(parents=True, exist_ok=True)
    effective = effective_config_fn(config, source_output_root)
    source_plan = recovery_stage_plan_fn(
        effective,
        source_output_root,
        base_model_path=base_model_path,
    )
    reusable = (
        set(source_plan["reusable_cell_keys"])
        if source_plan["prepare_complete"]
        else set()
    )
    source_text = str(source_output_root)
    destination_text = str(output_root)
    linked_files = 0
    linked_bytes = 0
    source_cell_hashes = {
        key: sha256_fn(source_output_root / "cells" / key / "cell_manifest.json")
        for key in reusable
    }

    if source_plan["prepare_complete"]:
        recovery_label = source_output_root.parent.name

        def mapped_relative(relative: Path) -> Path | None:
            if relative.parts[0] in transient_top_level:
                return None
            if len(relative.parts) == 1 and relative.name in transient_files:
                return None
            if relative.parts[0] == "cells" and (
                len(relative.parts) < 2 or relative.parts[1] not in reusable
            ):
                return None
            if (
                relative.parts[0] == "liveness"
                and not source_plan["liveness_complete"]
            ):
                return None
            if relative.parts[0] == "logs":
                return (
                    Path("logs")
                    / f"recovered_{recovery_label}"
                    / Path(*relative.parts[1:])
                )
            return relative

        for source in sorted(
            path for path in source_output_root.rglob("*") if path.is_dir()
        ):
            if source.is_symlink():
                raise RuntimeError(f"Recovery refuses symbolic links: {source}")
            relative = mapped_relative(source.relative_to(source_output_root))
            if relative is not None:
                (output_root / relative).mkdir(parents=True, exist_ok=True)
        for source in sorted(source_output_root.rglob("*")):
            if source.is_symlink():
                raise RuntimeError(f"Recovery refuses symbolic links: {source}")
            if not source.is_file():
                continue
            relative = mapped_relative(source.relative_to(source_output_root))
            if relative is None:
                continue
            destination = output_root / relative
            hardlink_file(source, destination)
            linked_files += 1
            linked_bytes += source.stat().st_size

        for path in (
            output_root / "prepare_manifest.json",
            output_root / "split_manifest.json",
            output_root / "source_provenance.json",
        ):
            if not path.is_file():
                continue
            try:
                value = read_json_object(path)
            except (OSError, ValueError, TypeError, json.JSONDecodeError):
                continue
            updated = replace_path_prefix(value, source_text, destination_text)
            if updated != value:
                write_json(path, updated)

        for key, source_hash in sorted(source_cell_hashes.items()):
            manifest_path = output_root / "cells" / key / "cell_manifest.json"
            value = read_json_object(manifest_path)
            value = replace_path_prefix(value, source_text, destination_text)
            value["recovery_provenance"] = {
                "source_output_root": source_text,
                "source_manifest_sha256": source_hash,
                "import_mode": "identity_checked_hardlink",
                "scientific_variables_changed": False,
            }
            write_json(manifest_path, value)

    imported_cell_manifests = [
        {
            "cell_key": key,
            "source_manifest_sha256": source_cell_hashes[key],
            "imported_manifest_sha256": sha256_fn(
                output_root / "cells" / key / "cell_manifest.json"
            ),
        }
        for key in sorted(reusable)
    ]
    import_manifest = {
        "schema_version": int(schema_version),
        "experiment_id": experiment_id_fn(effective),
        "source_commit": source_commit,
        "source_output_root": source_text,
        "destination_output_root": destination_text,
        "source_plan": source_plan,
        "imported_reusable_cells": sorted(reusable),
        "imported_cell_manifests": imported_cell_manifests,
        "linked_files": linked_files,
        "linked_bytes": linked_bytes,
        "copy_mode": "hardlink_read_only_then_copy_on_atomic_json_rewrite",
        "incomplete_cells_imported": False,
        "scientific_variables_changed": False,
        "complete": True,
    }
    write_json(output_root / "recovery" / "IMPORT_MANIFEST.json", import_manifest)
    return import_manifest


def publish_recovery_checkpoint(
    *,
    payload: Mapping[str, Any],
    snapshot_root: Path,
    package_output: Path,
    repo_root: Path,
    experiment_id_value: str,
    source_commit: str,
    require_origin_main_match: bool,
    mirror_value: str,
    sha256_fn: Callable[[Path], str],
    write_json: Callable[[Path, Any], None],
) -> dict[str, Any]:
    """Package and optionally mirror an already-materialized recovery snapshot."""

    command = [
        sys.executable,
        str(repo_root / "scripts" / "package_experiment_hardened.py"),
        "--repo-root",
        str(repo_root),
        "--experiment-id",
        experiment_id_value,
        "--package-kind",
        "experiment-checkpoint",
        "--result-dir",
        str(snapshot_root),
        "--output",
        str(package_output),
        "--base-commit",
        source_commit,
        "--no-repository-changes",
        "--large-file-persistence",
        "persistent_local",
        "--source-file",
        "scripts/run_e8_multitask_exp_coldstart.sh",
        "--source-file",
        "src/drpo/e8_multitask_exp_tuning.py",
    ]
    if require_origin_main_match:
        command.append("--require-origin-main-match")
    completed = subprocess.run(
        command,
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "Recovery checkpoint packaging failed: "
            + (completed.stderr.strip() or completed.stdout.strip())
        )

    mirror_value = mirror_value.strip()
    mirror_path: Path | None = None
    if mirror_value:
        mirror_root = Path(mirror_value).resolve()
        mirror_root.mkdir(parents=True, exist_ok=True)
        mirror_path = mirror_root / package_output.name
        temporary = mirror_path.with_name(f".{mirror_path.name}.tmp-{os.getpid()}")
        shutil.copy2(package_output, temporary)
        if sha256_fn(temporary) != sha256_fn(package_output):
            temporary.unlink(missing_ok=True)
            raise RuntimeError("Recovery mirror copy failed checksum verification")
        os.replace(temporary, mirror_path)

    status = {
        **dict(payload),
        "package": str(package_output.resolve()),
        "package_sha256": sha256_fn(package_output),
        "mirror": str(mirror_path) if mirror_path else None,
        "mirror_configured": mirror_path is not None,
        "complete": True,
    }
    write_json(package_output.parent / "RECOVERY_CHECKPOINT_STATUS.json", status)
    return status


def terminal_audit(
    output_root: Path,
    *,
    cells: Sequence[CellLike],
    experiment_id_value: str,
    expected_terminal_step: int,
    engineering_self_test: bool,
    dense_profile: bool,
    coldstart_profile: bool,
    method_matrix: bool,
    execution_class: str,
    excluded_tasks: Mapping[str, Any],
    seed_batch_order: Sequence[int],
    coldstart_single_seed_response_shape: bool,
    compatibility_failure_buckets: Sequence[str],
    method_audit_fn: Callable[
        [CellLike, Mapping[str, Any]], MethodAuditResult
    ],
    audited_status_fn: Callable[[bool], str],
    write_json: Callable[[Path, Any], None],
) -> dict[str, Any]:
    """Run method-agnostic terminal checks with method evidence supplied by a hook."""

    provenance_path = output_root / "source_provenance.json"
    if not provenance_path.is_file():
        raise RuntimeError(
            "source_provenance.json is required before terminal audit"
        )
    provenance = read_json_object(provenance_path)
    base_commit = str(provenance.get("source_commit", ""))
    if len(base_commit) != 40 or any(
        char not in "0123456789abcdef" for char in base_commit
    ):
        raise RuntimeError(
            "source_provenance.json must contain one full lowercase Git SHA"
        )

    missing: list[str] = []
    incomplete: list[str] = []
    nan_inf: list[str] = []
    terminal_contract_failures: list[str] = []
    method_failure_buckets = {
        str(bucket): [] for bucket in compatibility_failure_buckets
    }
    for cell in cells:
        cell_root = output_root / "cells" / cell.key
        path = cell_root / "cell_manifest.json"
        if not path.is_file():
            failure_path = cell_root / "failure.json"
            if failure_path.is_file():
                failure = read_json_object(failure_path)
                incomplete.append(cell.key)
                if failure.get("nan_inf_failure"):
                    nan_inf.append(cell.key)
            else:
                missing.append(cell.key)
            continue

        value = read_json_object(path)
        if not value.get("complete") or value.get("evaluation_status") != "complete":
            incomplete.append(cell.key)
        if value.get("nan_inf_failure"):
            nan_inf.append(cell.key)
        if not engineering_self_test:
            if (
                int(value.get("terminal_step", -1)) != expected_terminal_step
                or value.get("stop_reason") != "max_steps"
                or value.get("test_partition_accessed") is not False
            ):
                terminal_contract_failures.append(cell.key)
            method_audit = method_audit_fn(cell, value)
            if not method_audit.passed:
                bucket = str(method_audit.failure_bucket)
                if bucket in method_failure_buckets:
                    method_failure_buckets[bucket].append(cell.key)
                else:
                    terminal_contract_failures.append(cell.key)

    inherited_complete = True
    aggregate_complete = True
    reproduction_gate_status: str | None = None
    if dense_profile:
        snapshot_path = output_root / "inherited" / "parent_snapshot.json"
        aggregate_path = output_root / "aggregate" / "aggregate_summary.json"
        inherited_complete = snapshot_path.is_file() and bool(
            read_json_object(snapshot_path).get("complete")
        )
        aggregate_complete = aggregate_path.is_file() and int(
            read_json_object(aggregate_path).get("cell_count", 0)
        ) == len(cells)
    elif coldstart_profile:
        aggregate_path = output_root / "aggregate" / "aggregate_summary.json"
        protocol_path = (
            output_root / "aggregate" / "countdown_protocol_diagnostic.json"
        )
        if protocol_path.is_file():
            reproduction_gate_status = str(
                read_json_object(protocol_path).get("status")
            )
        expected_protocol_status = (
            "NOT_RUN_ENGINEERING"
            if engineering_self_test
            else (
                "NOT_RUN"
                if not any(cell.task == "countdown" for cell in cells)
                else "PASS"
            )
        )
        aggregate_complete = (
            aggregate_path.is_file()
            and int(read_json_object(aggregate_path).get("cell_count", 0))
            == len(cells)
            and reproduction_gate_status == expected_protocol_status
        )

    seed_batch_protocol_complete: bool | None = None
    if method_matrix:
        scheduler_path = output_root / "scheduler" / "dynamic_run.json"
        scheduler = (
            read_json_object(scheduler_path)
            if scheduler_path.is_file()
            else {}
        )
        order = [int(value) for value in seed_batch_order]
        expected = {
            str(seed): sum(cell.seed == seed for cell in cells)
            for seed in order
        }
        seed_batch_protocol_complete = (
            scheduler.get("seed_batch_barriers") is True
            and scheduler.get("seed_batch_order") == order
            and scheduler.get("seed_batch_expected_cells") == expected
            and scheduler.get("seed_batch_completed_cells") == expected
            and scheduler.get("complete") is True
        )

    all_complete = (
        not missing
        and not incomplete
        and not nan_inf
        and not terminal_contract_failures
        and not any(method_failure_buckets.values())
        and inherited_complete
        and aggregate_complete
        and seed_batch_protocol_complete is not False
    )
    audit = {
        "schema_version": 1,
        "experiment_id": experiment_id_value,
        "base_commit": base_commit,
        "expected_cells": len(cells),
        "missing_cells": sorted(set(missing)),
        "incomplete_cells": sorted(set(incomplete)),
        "nan_inf_cells": sorted(set(nan_inf)),
        "terminal_contract_failures": sorted(
            set(terminal_contract_failures)
        ),
        **{
            bucket: sorted(set(values))
            for bucket, values in method_failure_buckets.items()
        },
        "seed_batch_protocol_complete": seed_batch_protocol_complete,
        "all_training_and_evaluation_complete": all_complete,
        "execution_class": execution_class,
        "test_partition_accessed": False,
        "task_performance_event": (
            "not_adjudicated_without_registered_collapse_threshold"
        ),
        "structure_event": "greedy_and_sampled_valid_rate_diagnostic_only",
        "nan_inf_event_count": len(set(nan_inf)),
        "inherited_parent_inputs_complete": inherited_complete,
        "aggregate_complete": aggregate_complete,
        "countdown_protocol_diagnostic_status": reproduction_gate_status,
        "countdown_result_gate": False if coldstart_profile else None,
        "transfer_exp_single_seed_response_shape_localization": (
            coldstart_single_seed_response_shape
        ),
        "excluded_tasks": dict(excluded_tasks),
        "single_seed_shape_discovery": (
            dense_profile or coldstart_single_seed_response_shape
        ),
        "fresh_seed_confirmation_required": (
            dense_profile or coldstart_single_seed_response_shape
        ),
        "fixed_horizon_is_convergence": False,
        "scientific_status": audited_status_fn(all_complete),
        "engineering_placeholder_backend": engineering_self_test,
    }
    write_json(output_root / "terminal_audit.json", audit)
    return audit


def compact_logs(
    output_root: Path,
    *,
    experiment_id_value: str,
    sha256_fn: Callable[[Path], str],
    write_json: Callable[[Path, Any], None],
) -> dict[str, Any]:
    logs_root = output_root / "logs"
    archive_root = output_root / "persistent_raw_archives"
    archive = archive_root / "cell_and_stage_logs.tar.gz"
    index_path = logs_root / "LOG_ARCHIVE_INDEX.json"
    prepared_path = logs_root / "LOG_ARCHIVE_PREPARED.json"
    if index_path.is_file() and archive.is_file():
        value = read_json_object(index_path)
        if value.get("archive_sha256") == sha256_fn(archive) and value.get("complete"):
            return value
    if prepared_path.is_file():
        prepared = read_json_object(prepared_path)
        if (
            not archive.is_file()
            or prepared.get("archive_sha256") != sha256_fn(archive)
            or prepared.get("prepared") is not True
        ):
            raise RuntimeError("Prepared log archive transaction is missing or corrupt")
        rows = list(prepared.get("members", ()))
        if not rows or not all(isinstance(row, dict) for row in rows):
            raise RuntimeError("Prepared log archive inventory is empty or invalid")
    else:
        log_files = [
            path
            for path in sorted(logs_root.rglob("*"))
            if path.is_file()
            and path not in {index_path, prepared_path}
            and "tails" not in path.relative_to(logs_root).parts
            and not path.is_symlink()
        ]
        if not log_files:
            raise RuntimeError("No logs are available for transactional compaction")
        archive_root.mkdir(parents=True, exist_ok=True)
        temporary = archive.with_name(f".{archive.name}.tmp-{os.getpid()}")
        rows = []
        with tarfile.open(temporary, "w:gz") as handle:
            for path in log_files:
                relative = path.relative_to(logs_root)
                rows.append(
                    {
                        "path": relative.as_posix(),
                        "size_bytes": path.stat().st_size,
                        "sha256": sha256_fn(path),
                    }
                )
                handle.add(path, arcname=relative.as_posix(), recursive=False)
        os.replace(temporary, archive)
        with tarfile.open(archive, "r:gz") as handle:
            members = {member.name: member for member in handle.getmembers()}
            for row in rows:
                name = str(row["path"])
                member = members.get(name)
                extracted = handle.extractfile(member) if member is not None else None
                if member is None or not member.isfile() or extracted is None:
                    raise RuntimeError(f"Log archive member is missing or invalid: {name}")
                digest = hashlib.sha256()
                size = 0
                while chunk := extracted.read(1024 * 1024):
                    digest.update(chunk)
                    size += len(chunk)
                if size != row["size_bytes"] or digest.hexdigest() != row["sha256"]:
                    raise RuntimeError(f"Log archive member verification failed: {name}")
        tails_root = logs_root / "tails"
        for path in log_files:
            relative = path.relative_to(logs_root)
            tail = tails_root / relative
            tail.parent.mkdir(parents=True, exist_ok=True)
            with path.open("rb") as source:
                size = path.stat().st_size
                if size > 65536:
                    source.seek(-65536, os.SEEK_END)
                tail.write_bytes(source.read())
        prepared = {
            "schema_version": 1,
            "experiment_id": experiment_id_value,
            "archive": str(archive.resolve()),
            "archive_sha256": sha256_fn(archive),
            "members": rows,
            "prepared": True,
        }
        write_json(prepared_path, prepared)
    for row in rows:
        relative = Path(str(row.get("path", "")))
        if (
            not relative.parts
            or relative.is_absolute()
            or ".." in relative.parts
            or "tails" in relative.parts
        ):
            raise RuntimeError(f"Unsafe prepared log path: {relative}")
        path = logs_root / relative
        tail = logs_root / "tails" / relative
        if not tail.is_file():
            raise RuntimeError(f"Prepared log tail is missing: {tail}")
        if path.is_file():
            if path.stat().st_size != int(row.get("size_bytes", -1)) or sha256_fn(
                path
            ) != row.get("sha256"):
                raise RuntimeError(f"Log changed during compaction transaction: {path}")
            path.unlink()
    value = {
        "schema_version": 1,
        "experiment_id": experiment_id_value,
        "archive": str(archive.resolve()),
        "archive_size_bytes": archive.stat().st_size,
        "archive_sha256": sha256_fn(archive),
        "members": rows,
        "tail_bytes_per_log": 65536,
        "raw_logs_persist_locally": True,
        "transactionally_resumable": True,
        "complete": True,
    }
    write_json(index_path, value)
    prepared_path.unlink(missing_ok=True)
    return value

