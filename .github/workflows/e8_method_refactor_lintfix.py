from __future__ import annotations

import ast
import textwrap
from pathlib import Path


def replace_once(text: str, old: str, new: str, *, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{label}: expected one anchor, found {count}")
    return text.replace(old, new, 1)


def replace_top_level_function(path: Path, name: str, replacement: str) -> None:
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    node = next(
        (
            item
            for item in tree.body
            if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef))
            and item.name == name
        ),
        None,
    )
    if node is None or node.end_lineno is None:
        raise RuntimeError(f"cannot find top-level function {name} in {path}")
    start = min([node.lineno, *(d.lineno for d in node.decorator_list)])
    lines = source.splitlines(keepends=True)
    new = textwrap.dedent(replacement).strip("\n") + "\n"
    path.write_text(
        "".join(lines[: start - 1]) + new + "".join(lines[node.end_lineno :]),
        encoding="utf-8",
    )


def remove_top_level_function(path: Path, name: str) -> None:
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    node = next(
        (
            item
            for item in tree.body
            if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef))
            and item.name == name
        ),
        None,
    )
    if node is None or node.end_lineno is None:
        raise RuntimeError(f"cannot find top-level function {name} in {path}")
    start = min([node.lineno, *(d.lineno for d in node.decorator_list)])
    lines = source.splitlines(keepends=True)
    path.write_text(
        "".join(lines[: start - 1]) + "".join(lines[node.end_lineno :]),
        encoding="utf-8",
    )


def remove_methodspec_projection_surface(path: Path) -> None:
    names = {"group_projection", "group_order", "plot_columns", "plot_projection"}
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    ranges: list[tuple[int, int]] = []
    method_spec = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "MethodSpec"
    )
    for item in method_spec.body:
        if isinstance(item, ast.AnnAssign) and isinstance(item.target, ast.Name):
            if item.target.id in names:
                ranges.append((item.lineno, item.end_lineno or item.lineno))
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if not isinstance(node.func, ast.Name) or node.func.id != "MethodSpec":
            continue
        for keyword in node.keywords:
            if keyword.arg in names:
                ranges.append((keyword.lineno, keyword.end_lineno or keyword.lineno))
    lines = source.splitlines(keepends=True)
    for start, end in sorted(ranges, reverse=True):
        del lines[start - 1 : end]
    path.write_text("".join(lines), encoding="utf-8")


exp_path = Path("src/drpo/e8_multitask_exp_tuning.py")
results_path = Path("src/drpo/e8_multitask_results.py")
runtime_path = Path("src/drpo/e8_multitask_runtime.py")
orchestration_path = Path("src/drpo/e8_multitask_orchestration.py")
contract_path = Path("tests/test_e8_method_integration_contract.sh")
test_path = Path("tests/test_e8_multitask_p0.py")

# Existing lint-only repairs.
text = exp_path.read_text(encoding="utf-8")
text = replace_once(text, "import queue\n", "", label="unused queue import")
text = replace_once(
    text,
    "    except Exception as exc:  # The plan records the exact fail-closed reason.\n",
    "    except Exception as exc:  # noqa: BLE001 - fail-closed reason capture\n",
    label="prepare recovery catch",
)
text = replace_once(
    text,
    "            _require_calibration_gate(config, output_root, base_model_path=base_model_path)\n"
    "            calibration_complete = True\n"
    "        except Exception as exc:\n",
    "            _require_calibration_gate(config, output_root, base_model_path=base_model_path)\n"
    "            calibration_complete = True\n"
    "        except Exception as exc:  # noqa: BLE001 - fail-closed reason capture\n",
    label="calibration recovery catch",
)
text = replace_once(
    text,
    "            _require_liveness_gate(config, output_root, base_model_path=base_model_path)\n"
    "            liveness_complete = True\n"
    "        except Exception as exc:\n",
    "            _require_liveness_gate(config, output_root, base_model_path=base_model_path)\n"
    "            liveness_complete = True\n"
    "        except Exception as exc:  # noqa: BLE001 - fail-closed reason capture\n",
    label="liveness recovery catch",
)
text = replace_once(
    text,
    "        def aggregate_group(group: Sequence[Mapping[str, Any]]) -> dict[str, Any]:\n"
    "            first = group[0]\n"
    "            return {\n"
    "                \"task\": task,\n",
    "        def aggregate_group(\n"
    "            group: Sequence[Mapping[str, Any]], task_name: str = task\n"
    "        ) -> dict[str, Any]:\n"
    "            first = group[0]\n"
    "            return {\n"
    "                \"task\": task_name,\n",
    label="legacy exponential aggregate closure",
)
exp_path.write_text(text, encoding="utf-8")

text = orchestration_path.read_text(encoding="utf-8")
text = replace_once(
    text,
    "len(set(int(value) for value in gpu_ids))",
    "len({int(value) for value in gpu_ids})",
    label="gpu id set comprehension",
)
text = replace_once(
    text,
    "            except Exception as exc:  # pragma: no cover - caller-specific failures.\n",
    "            except Exception as exc:  # noqa: BLE001  # pragma: no cover\n",
    label="run-cell callback boundary",
)
text = replace_once(
    text,
    "                except Exception as exc:  # Keep failure evidence in scheduler output.\n",
    "                except Exception as exc:  # noqa: BLE001 - callback failure evidence\n",
    label="after-success callback boundary",
)
orchestration_path.write_text(text, encoding="utf-8")

# Results: keep one production result projection/group identity implementation.
results_path.write_text(
    textwrap.dedent(
        r'''
        """Method-agnostic result projection helpers for E8 multitask experiments."""

        from __future__ import annotations

        import csv
        import json
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
        class MethodResultProjection:
            parameters: Mapping[str, Any]
            compatibility_columns: Mapping[str, Any]


        @dataclass(frozen=True)
        class ResultRecord:
            public: Mapping[str, Any]
            method_parameters: Mapping[str, Any]

            def public_row(self) -> dict[str, Any]:
                return dict(self.public)


        _RESERVED_COMMON_COLUMNS = frozenset(
            {"source", "task", "method", "seed", "stage", "cell_key"}
        )


        def _checked_merge(
            target: dict[str, Any], values: Mapping[str, Any], *, label: str
        ) -> None:
            collisions = sorted(set(target).intersection(values))
            if collisions:
                raise ValueError(
                    f"{label} attempted to overwrite result columns: {collisions}"
                )
            target.update(values)


        def common_result_record(
            cell: CellLike,
            value: Mapping[str, Any],
            *,
            source: str,
            project_method: Callable[[CellLike], MethodResultProjection],
            project_metrics: Callable[[Mapping[str, Any]], Mapping[str, Any]],
        ) -> ResultRecord:
            projection = project_method(cell)
            compatibility = dict(projection.compatibility_columns)
            reserved = sorted(_RESERVED_COMMON_COLUMNS.intersection(compatibility))
            if reserved:
                raise ValueError(
                    f"Method compatibility projection uses reserved columns: {reserved}"
                )
            public: dict[str, Any] = {
                "source": source,
                "task": cell.task,
                "method": cell.method,
            }
            _checked_merge(
                public,
                compatibility,
                label="Method compatibility projection",
            )
            public.update(
                {
                    "seed": int(cell.seed),
                    "stage": cell.stage,
                    "cell_key": cell.key,
                }
            )
            _checked_merge(
                public,
                dict(project_metrics(value)),
                label="Metric projection",
            )
            return ResultRecord(
                public=public,
                method_parameters=dict(projection.parameters),
            )


        def parameter_identity(parameters: Mapping[str, Any]) -> str:
            return json.dumps(
                dict(parameters),
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
            )


        def parameter_sort_key(parameters: Mapping[str, Any]) -> tuple[Any, ...]:
            def sortable(value: Any) -> tuple[int, Any]:
                if value is None:
                    return (0, "")
                if isinstance(value, bool):
                    return (1, int(value))
                if isinstance(value, (int, float)):
                    return (2, float(value))
                return (3, str(value))

            return tuple(
                (str(key), *sortable(value))
                for key, value in sorted(
                    parameters.items(), key=lambda item: str(item[0])
                )
            )


        def write_csv(
            path: Path, rows: Sequence[Mapping[str, Any] | ResultRecord]
        ) -> None:
            if not rows:
                raise ValueError(f"Cannot write empty CSV: {path}")
            materialized = [
                row.public_row() if isinstance(row, ResultRecord) else dict(row)
                for row in rows
            ]
            path.parent.mkdir(parents=True, exist_ok=True)
            fieldnames = list(materialized[0])
            for row in materialized[1:]:
                for key in row:
                    if key not in fieldnames:
                        fieldnames.append(key)
            with path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(materialized)
        '''
    ).lstrip(),
    encoding="utf-8",
)

# Runtime: remove the parallel test-only audit/inventory framework. Production
# cmd_audit remains the terminal gate and consumes method-owned audit hooks.
runtime_path.write_text(
    textwrap.dedent(
        r'''
        """Method-agnostic recovery primitives for E8 multitask experiments."""

        from __future__ import annotations

        import hashlib
        import json
        from collections.abc import Mapping
        from dataclasses import dataclass
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
            evidence: Mapping[str, Any]
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
            collisions = sorted(
                set(cell_identity).intersection(cell_identity_fields)
            )
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
        '''
    ).lstrip(),
    encoding="utf-8",
)

# Narrow MethodSpec: one opaque parameter mapping is the aggregation/plot source.
text = exp_path.read_text(encoding="utf-8")
text = replace_once(
    text,
    "    paper_grid_source: Callable[[Mapping[str, Any], Cell], Path]\n",
    "    paper_grid_paths: Callable[[Mapping[str, Any], Mapping[str, Any], Cell], tuple[Path, Path]] | None\n",
    label="MethodSpec paper grid hook",
)
text = replace_once(
    text,
    "    paper_cell_parameters: Callable[[Cell], tuple[str, float, float]]\n",
    "    paper_cell_parameters: Callable[[Cell], tuple[str, float, float]] | None\n",
    label="MethodSpec optional paper parameters",
)
text = replace_once(
    text,
    "            parameters=lambda cell: {\n"
    "                \"beta\": cell.beta,\n"
    "                \"dpo_initialization\": cell.dpo_initialization,\n"
    "            },\n",
    "            parameters=lambda cell: {\"beta\": cell.beta},\n",
    label="DPO sweep parameter identity",
)
text = text.replace(
    "            paper_grid_source=_paper_grid_source_exponential,\n",
    "            paper_grid_paths=_paper_grid_paths_exponential,\n",
)
if text.count("paper_grid_paths=_paper_grid_paths_exponential") != 3:
    raise RuntimeError("expected three exponential paper-grid registrations")
text = replace_once(
    text,
    "            paper_grid_source=lambda config, cell: _canonical_asymre_grid_path(),\n",
    "            paper_grid_paths=lambda config, record, cell: (\n"
    "                _canonical_asymre_grid_path(), _canonical_asymre_grid_path()\n"
    "            ),\n",
    label="AsymRE paper grid registration",
)
text = replace_once(
    text,
    "            paper_grid_source=lambda config, cell: _canonical_topr_grid_path(),\n",
    "            paper_grid_paths=lambda config, record, cell: (\n"
    "                _canonical_topr_grid_path(), _canonical_topr_grid_path()\n"
    "            ),\n",
    label="TOPR paper grid registration",
)
text = replace_once(
    text,
    "            paper_grid_source=_unsupported_paper_grid,\n",
    "            paper_grid_paths=None,\n",
    label="DPO paper grid registration",
)
text = replace_once(
    text,
    "            paper_cell_parameters=_unsupported_paper_params,\n",
    "            paper_cell_parameters=None,\n",
    label="DPO paper parameter registration",
)
exp_path.write_text(text, encoding="utf-8")

replace_top_level_function(
    exp_path,
    "_paper_grid_source_exponential",
    r'''
    def _paper_grid_paths_exponential(
        config: Mapping[str, Any],
        record: Mapping[str, Any],
        cell: Cell,
    ) -> tuple[Path, Path]:
        grid_name = (
            "round1_grid"
            if cell.task != "countdown"
            else _paper_grid_name(
                0.0 if cell.lambda_value is None else float(cell.lambda_value)
            )
        )
        return Path(str(record[grid_name])), _canonical_paths(config)[grid_name]
    ''',
)
remove_top_level_function(exp_path, "_unsupported_paper_grid")
remove_top_level_function(exp_path, "_unsupported_paper_params")

replace_top_level_function(
    exp_path,
    "_paper_grid_for_cell",
    r'''
    def _paper_grid_for_cell(
        config: Mapping[str, Any],
        record: Mapping[str, Any],
        cell: Cell,
    ) -> tuple[Path, Path]:
        spec = _method_spec(cell.method)
        if spec.paper_grid_paths is None:
            raise RuntimeError(
                f"{cell.method} does not use the canonical paper grid runtime"
            )
        return spec.paper_grid_paths(config, record, cell)
    ''',
)

text = exp_path.read_text(encoding="utf-8")
text = replace_once(
    text,
    "    grid_path = _paper_grid_for_cell(record, cell)\n"
    "    method_spec = _method_spec(cell.method)\n"
    "    grid_source_path = method_spec.paper_grid_source(config, cell)\n",
    "    method_spec = _method_spec(cell.method)\n"
    "    grid_path, grid_source_path = _paper_grid_for_cell(config, record, cell)\n"
    "    if method_spec.paper_cell_parameters is None:\n"
    "        raise RuntimeError(\n"
    "            f\"{cell.method} does not use canonical paper cell parameters\"\n"
    "        )\n",
    label="canonical train grid dispatch",
)
exp_path.write_text(text, encoding="utf-8")

# Liveness infers the registered method from representative_family and the one
# method parameter from representative_* metadata. No scientific method names.
replace_top_level_function(
    exp_path,
    "_canonical_cold_liveness_cell",
    r'''
    def _canonical_cold_liveness_cell(grid_path: Path) -> Cell:
        """Derive one liveness cell through the registered method contract."""

        grid = yaml.safe_load(grid_path.read_text(encoding="utf-8"))
        if not isinstance(grid, dict):
            raise TypeError("Canonical liveness grid root must be a mapping")
        liveness = grid["execution"]["liveness"]
        seed_offsets = grid["sweep"]["seed_offsets"]
        method = str(liveness.get("representative_family", METHOD_EXPONENTIAL))
        parameter_keys = [
            str(key)
            for key in liveness
            if str(key).startswith("representative_")
            and str(key) not in {"representative_family", "representative_alpha"}
        ]
        if len(parameter_keys) != 1:
            raise RuntimeError(
                "Canonical liveness grid must expose exactly one representative method parameter"
            )
        value = float(liveness[parameter_keys[0]])
        return _method_spec(method).build_cell(
            task="countdown",
            method=method,
            seed=int(seed_offsets[0]),
            stage="liveness",
            value=value,
            lambda_only=False,
            dpo_initialization=None,
        )
    ''',
)

remove_methodspec_projection_surface(exp_path)

# Reuse parameters directly for grouping and plotting.
text = exp_path.read_text(encoding="utf-8")
text = replace_once(
    text,
    "        key=lambda item: spec.group_order(item[0]),\n",
    "        key=lambda item: e8_results.parameter_sort_key(item[0]),\n",
    label="generic grouped-curve ordering",
)
text = replace_once(
    text,
    "                **dict(spec.group_projection(parameters)),\n",
    "                **parameters,\n",
    label="generic grouped-curve projection",
)
text = replace_once(
    text,
    "            projection = dict(spec.plot_projection(cells_by_key[cell_key]))\n"
    "            parameter_columns = {\n"
    "                column: projection.get(column)\n"
    "                for column in spec.plot_columns\n"
    "                if column != \"dpo_initialization\"\n"
    "            }\n",
    "            projection = dict(spec.parameters(cells_by_key[cell_key]))\n"
    "            parameter_columns = {\n"
    "                name: value\n"
    "                for name, value in projection.items()\n"
    "                if name != \"dpo_initialization\"\n"
    "            }\n",
    label="single-method plot parameters",
)
exp_path.write_text(text, encoding="utf-8")

# Matrix plot keeps the frozen legacy columns and appends any future parameters.
text = exp_path.read_text(encoding="utf-8")
old = '''        methods = _coldstart_methods(config)\n        run_id, source_commit = _coldstart_run_provenance(output_root)\n        plot_rows = [\n            {\n                "task": row["task"],\n                "method": row["method"],\n                "delta_v": row.get("delta_v"),\n                "beta": row.get("beta"),\n                "dpo_initialization": row.get("dpo_initialization"),\n                "seed": row["seed"],\n                "stage": row["stage"],\n                "experiment_id": experiment_id(config),\n                "run_id": run_id,\n                "source_commit": source_commit,\n                **_coldstart_plot_metrics(row),\n            }\n            for row in rows\n        ]\n        _write_csv(output_root / "aggregate" / "plot_curve_points.csv", plot_rows)\n\n        configured_cells = build_cells(config)\n        cells_by_key = {cell.key: cell for cell in configured_cells}\n        if len(cells_by_key) != len(configured_cells):\n            raise RuntimeError("Baseline matrix contains duplicate cell keys")\n'''
new = '''        methods = _coldstart_methods(config)\n        run_id, source_commit = _coldstart_run_provenance(output_root)\n        configured_cells = build_cells(config)\n        cells_by_key = {cell.key: cell for cell in configured_cells}\n        if len(cells_by_key) != len(configured_cells):\n            raise RuntimeError("Baseline matrix contains duplicate cell keys")\n        legacy_parameter_columns = ("delta_v", "beta", "dpo_initialization")\n        extra_parameter_names = tuple(\n            dict.fromkeys(\n                name\n                for cell in configured_cells\n                if cell.method in methods\n                for name in _method_spec(cell.method).parameters(cell)\n                if name not in legacy_parameter_columns\n            )\n        )\n        plot_rows: list[dict[str, Any]] = []\n        for row in rows:\n            cell = cells_by_key[str(row["cell_key"])]\n            parameters = dict(_method_spec(cell.method).parameters(cell))\n            plot_rows.append(\n                {\n                    "task": row["task"],\n                    "method": row["method"],\n                    "delta_v": row.get("delta_v"),\n                    "beta": row.get("beta"),\n                    "dpo_initialization": row.get("dpo_initialization"),\n                    **{name: parameters.get(name) for name in extra_parameter_names},\n                    "seed": row["seed"],\n                    "stage": row["stage"],\n                    "experiment_id": experiment_id(config),\n                    "run_id": run_id,\n                    "source_commit": source_commit,\n                    **_coldstart_plot_metrics(row),\n                }\n            )\n        _write_csv(output_root / "aggregate" / "plot_curve_points.csv", plot_rows)\n'''
text = replace_once(text, old, new, label="matrix generic plot parameters")
exp_path.write_text(text, encoding="utf-8")

# Per-task plot follows the same compatibility rule.
text = exp_path.read_text(encoding="utf-8")
old = '''    _write_csv(all_cells_path, rows)\n    plot_rows = [\n        {\n            "experiment_id": experiment_id(config),\n            "run_id": run_id,\n            "source_commit": source_commit,\n            "task": row["task"],\n            "method": row["method"],\n            "delta_v": row.get("delta_v"),\n            "beta": row.get("beta"),\n            "dpo_initialization": row.get("dpo_initialization"),\n            "lambda": row["lambda"],\n            "rho": row["rho"],\n            "seed": row["seed"],\n            "stage": row["stage"],\n            **_coldstart_plot_metrics(row),\n        }\n        for row in rows\n    ]\n'''
new = '''    _write_csv(all_cells_path, rows)\n    cells_by_key = {\n        cell.key: cell for cell in build_cells(config) if cell.task == task\n    }\n    legacy_parameter_columns = (\n        "delta_v", "beta", "dpo_initialization", "lambda", "rho"\n    )\n    extra_parameter_names = tuple(\n        dict.fromkeys(\n            name\n            for cell in cells_by_key.values()\n            for name in _method_spec(cell.method).parameters(cell)\n            if name not in legacy_parameter_columns\n        )\n    )\n    plot_rows: list[dict[str, Any]] = []\n    for row in rows:\n        cell = cells_by_key[str(row["cell_key"])]\n        parameters = dict(_method_spec(cell.method).parameters(cell))\n        plot_rows.append(\n            {\n                "experiment_id": experiment_id(config),\n                "run_id": run_id,\n                "source_commit": source_commit,\n                "task": row["task"],\n                "method": row["method"],\n                "delta_v": row.get("delta_v"),\n                "beta": row.get("beta"),\n                "dpo_initialization": row.get("dpo_initialization"),\n                "lambda": row["lambda"],\n                "rho": row["rho"],\n                **{name: parameters.get(name) for name in extra_parameter_names},\n                "seed": row["seed"],\n                "stage": row["stage"],\n                **_coldstart_plot_metrics(row),\n            }\n        )\n'''
text = replace_once(text, old, new, label="task-result generic plot parameters")
exp_path.write_text(text, encoding="utf-8")

# Characterization test generated by the first helper follows the narrowed seam.
text = test_path.read_text(encoding="utf-8")
text = text.replace(
    'assert "method_spec.paper_grid_source(config, cell)" in source',
    'assert "_paper_grid_for_cell(config, record, cell)" in source',
)
test_path.write_text(text, encoding="utf-8")

# One real dummy MethodSpec acceptance path; no parallel test-only runtime layer.
contract_path.write_text(
    textwrap.dedent(
        r'''
        #!/usr/bin/env bash
        set -euo pipefail
        ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
        export PYTHONPATH="${ROOT_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}"
        python3 -m py_compile \
          "${ROOT_DIR}/src/drpo/e8_multitask_orchestration.py" \
          "${ROOT_DIR}/src/drpo/e8_multitask_results.py" \
          "${ROOT_DIR}/src/drpo/e8_multitask_runtime.py"
        python3 - <<'PY'
        from __future__ import annotations

        import inspect
        import json
        import tempfile
        import threading
        import time
        from pathlib import Path

        import yaml

        from drpo import e8_multitask_exp_tuning as exp_tuning
        from drpo.e8_multitask_orchestration import (
            SchedulerCallbacks,
            execution_geometry,
            nominal_batches,
            plan_rows,
            run_dynamic_queue,
        )
        from drpo.e8_multitask_runtime import MethodAuditResult, recovery_identity

        repo_root = Path(__import__("os").environ["PYTHONPATH"].split(":", 1)[0]).parent
        for relative in (
            "src/drpo/e8_multitask_orchestration.py",
            "src/drpo/e8_multitask_results.py",
            "src/drpo/e8_multitask_runtime.py",
        ):
            source = (repo_root / relative).read_text(encoding="utf-8").lower()
            for forbidden in ("asymre", "topr", "canonical_dpo"):
                assert forbidden not in source, (relative, forbidden)

        dummy_name = "dummy_contract_method"

        def build_dummy(*, task, method, seed, stage, value, lambda_only, dpo_initialization):
            del lambda_only, dpo_initialization
            return exp_tuning.Cell(
                task=task,
                method=method,
                rho=None,
                seed=seed,
                stage=stage,
                method_parameters={"temperature": float(value)},
            )

        def parameters(cell):
            return dict(cell.method_parameters or {})

        def compatibility(cell):
            return {
                "delta_v": None,
                "beta": None,
                "dpo_initialization": None,
                "rho": None,
                "lambda": None,
                "temperature": parameters(cell)["temperature"],
            }

        def cell_key(cell):
            tag = f"{parameters(cell)['temperature']:.3f}".replace(".", "p")
            return f"{cell.task}__{cell.method}_{tag}__seed{cell.seed}"

        spec = exp_tuning.MethodSpec(
            name=dummy_name,
            build_cell=build_dummy,
            cell_key=cell_key,
            parameters=parameters,
            compatibility_columns=compatibility,
            cell_initialization=lambda config: None,
            initialization_identity=lambda config: {},
            train_cold=lambda cell, **kwargs: {"complete": True},
            liveness_task=lambda config: "countdown",
            liveness_runner=lambda **kwargs: {"complete": True},
            canonical_liveness_grid=None,
            paper_grid_paths=None,
            paper_cell_parameters=None,
            paper_formula="dummy_contract_only",
            audit_record=lambda cell, record: MethodAuditResult(
                True, {"temperature": parameters(cell)["temperature"]}
            ),
            audit_failure_bucket="terminal_contract_failures",
            scientific_kernel="dummy_contract_only",
            single_aggregate_metadata=lambda config: {},
            matrix_aggregate_metadata=lambda config: {},
        )
        exp_tuning._register_method_spec(spec)
        try:
            cells = tuple(
                exp_tuning._coldstart_method_cell(
                    task,
                    dummy_name,
                    seed,
                    "task_transfer",
                    value,
                    lambda_only=False,
                )
                for seed in (4000, 5000)
                for task in ("a", "b")
                for value in (0.1, 0.2)
            )
            batches = nominal_batches(
                cells,
                slot_count=4,
                seed_barrier=True,
                seed_order=(4000, 5000),
            )
            rows = plan_rows(
                batches,
                gpu_ids=(0, 1),
                project_cell=lambda cell: exp_tuning._method_spec(
                    cell.method
                ).compatibility_columns(cell),
            )
            assert len(rows) == 8
            assert {row["temperature"] for row in rows} == {0.1, 0.2}
            geometry = execution_geometry(
                cells,
                gpu_ids=(0, 1),
                slots_per_gpu=2,
                max_concurrent_cells=4,
                seed_barrier=True,
                seed_order=(4000, 5000),
            )
            assert geometry.expected_cells_by_seed == {4000: 4, 5000: 4}

            lock = threading.Lock()
            state = {"seed4000_hooks": 0, "released_early": False}

            def run_cell(cell, slot, gpu_id):
                del slot, gpu_id
                with lock:
                    if cell.seed == 5000 and state["seed4000_hooks"] < 4:
                        state["released_early"] = True
                time.sleep(0.001)
                return {"returncode": 0}

            def after_success(cell, row):
                del row
                if cell.seed == 4000:
                    with lock:
                        state["seed4000_hooks"] += 1

            run = run_dynamic_queue(
                cells,
                gpu_ids=(0, 1),
                slots_per_gpu=2,
                max_concurrent_cells=4,
                seed_barrier=True,
                seed_order=(4000, 5000),
                callbacks=SchedulerCallbacks(
                    run_cell=run_cell,
                    after_success=after_success,
                ),
            )
            assert len(run) == 8
            assert not state["released_early"]
            assert state["seed4000_hooks"] == 4

            aggregate_rows = [
                {
                    "cell_key": cell.key,
                    "task": cell.task,
                    "method": cell.method,
                    "seed": cell.seed,
                    "late_window_pass8_mean": parameters(cell)["temperature"],
                    "late_window_greedy_mean": 0.0,
                    "terminal_pass8": parameters(cell)["temperature"],
                    "terminal_greedy_valid_rate": 1.0,
                    "nan_inf_failure": False,
                }
                for cell in cells
            ]
            curve = exp_tuning._coldstart_method_grouped_curve(
                task="a",
                method=dummy_name,
                method_rows=[row for row in aggregate_rows if row["task"] == "a"],
                cells_by_key={cell.key: cell for cell in cells},
            )
            assert [row["temperature"] for row in curve] == [0.1, 0.2]

            identity = recovery_identity(
                cells[0],
                experiment_id="DEV-DUMMY-METHOD-SPEC",
                config_hash="cfg",
                cell_identity_fields=compatibility(cells[0]),
                common_identity_fields={"bank_hash": "bank"},
            )
            assert len(identity["identity_hash"]) == 64

            with tempfile.TemporaryDirectory() as temporary:
                grid_path = Path(temporary) / "liveness.yaml"
                grid_path.write_text(
                    yaml.safe_dump(
                        {
                            "execution": {
                                "liveness": {
                                    "representative_family": dummy_name,
                                    "representative_temperature": 0.1,
                                }
                            },
                            "sweep": {"seed_offsets": [4000]},
                        }
                    ),
                    encoding="utf-8",
                )
                live_cell = exp_tuning._canonical_cold_liveness_cell(grid_path)
                assert parameters(live_cell) == {"temperature": 0.1}

            assert spec.audit_record(cells[0], {}).passed
            audit_source = inspect.getsource(exp_tuning.cmd_audit)
            assert "_method_spec(cell.method).audit_record" in audit_source
            matrix_source = inspect.getsource(
                exp_tuning._aggregate_coldstart_matrix_unranked
            )
            for forbidden in ("METHOD_ASYMRE", "METHOD_TOPR", "METHOD_DPO"):
                assert forbidden not in matrix_source
            method_spec_fields = set(exp_tuning.MethodSpec.__dataclass_fields__)
            assert not {
                "group_projection",
                "group_order",
                "plot_columns",
                "plot_projection",
            }.intersection(method_spec_fields)
        finally:
            removed = exp_tuning._METHOD_SPECS.pop(dummy_name)
            assert removed is spec

        print(
            json.dumps(
                {
                    "dummy_cells": len(cells),
                    "generic_liveness_cell": "pass",
                    "matrix_parameter_extensibility": "pass",
                    "method_name_leakage": "pass",
                    "methodspec_surface_reduced": "pass",
                    "production_audit_hook": "pass",
                    "production_grouping": "pass",
                    "recovery_identity": "pass",
                    "seed_barrier": "pass",
                },
                sort_keys=True,
            )
        )
        PY
        '''
    ).lstrip(),
    encoding="utf-8",
)

# Guard against the actual redundant interfaces, while preserving the frozen
# artifact key named "paper_grid_source".
combined = "\n".join(
    path.read_text(encoding="utf-8")
    for path in (exp_path, results_path, runtime_path, contract_path, test_path)
)
for forbidden in (
    ".group_projection",
    ".group_order",
    ".plot_columns",
    ".plot_projection",
    "_unsupported_paper_grid",
    "_unsupported_paper_params",
):
    if forbidden in combined:
        raise RuntimeError(f"stale refactor surface remains: {forbidden}")
