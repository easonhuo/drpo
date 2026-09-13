#!/usr/bin/env bash
set -euo pipefail

cat > /tmp/trim_e8.py <<'PY'
from __future__ import annotations

import ast
import textwrap
from pathlib import Path

ROOT = Path.cwd()
EXP = ROOT / "src/drpo/e8_multitask_exp_tuning.py"


def _span(source: str, name: str) -> tuple[int, int]:
    tree = ast.parse(source)
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and node.name == name:
            starts = [node.lineno, *(d.lineno for d in node.decorator_list)]
            if node.end_lineno is None:
                raise RuntimeError(f"No end line for {name}")
            return min(starts), node.end_lineno
    raise RuntimeError(f"Cannot find top-level definition {name}")


def replace_top_level(source: str, name: str, replacement: str) -> str:
    start, end = _span(source, name)
    lines = source.splitlines(keepends=True)
    replacement = textwrap.dedent(replacement).strip("\n")
    middle = (replacement + "\n\n") if replacement else ""
    return "".join(lines[: start - 1]) + middle + "".join(lines[end:])


def insert_after_top_level(source: str, name: str, addition: str) -> str:
    _, end = _span(source, name)
    lines = source.splitlines(keepends=True)
    addition = textwrap.dedent(addition).strip("\n") + "\n\n"
    return "".join(lines[:end]) + "\n" + addition + "".join(lines[end:])


def replace_once(source: str, old: str, new: str, label: str) -> str:
    count = source.count(old)
    if count != 1:
        raise RuntimeError(f"{label}: expected one match, found {count}")
    return source.replace(old, new, 1)


(ROOT / "src/drpo/e8_multitask_results.py").write_text(
    '''"""Method-agnostic result projection helpers for E8 multitask experiments."""

from __future__ import annotations

import csv
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Protocol


class CellLike(Protocol):
    @property
    def key(self) -> str: ...

    task: str
    method: str
    seed: int
    stage: str


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


def common_result_row(
    cell: CellLike,
    *,
    source: str,
    method_columns: Mapping[str, Any],
    metrics: Mapping[str, Any],
) -> dict[str, Any]:
    compatibility = dict(method_columns)
    reserved = sorted(_RESERVED_COMMON_COLUMNS.intersection(compatibility))
    if reserved:
        raise ValueError(
            f"Method projection uses reserved result columns: {reserved}"
        )
    public: dict[str, Any] = {
        "source": source,
        "task": cell.task,
        "method": cell.method,
    }
    _checked_merge(public, compatibility, label="Method projection")
    public.update(
        {
            "seed": int(cell.seed),
            "stage": cell.stage,
            "cell_key": cell.key,
        }
    )
    _checked_merge(public, dict(metrics), label="Metric projection")
    return public


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


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Cannot write empty CSV: {path}")
    materialized = [dict(row) for row in rows]
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
''',
    encoding="utf-8",
)

(ROOT / "src/drpo/e8_multitask_runtime.py").write_text(
    '''"""Method-agnostic recovery primitives for E8 multitask experiments."""

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
''',
    encoding="utf-8",
)

source = EXP.read_text(encoding="utf-8")
source = replace_top_level(source, "_default_method_audit", "")
source = replace_top_level(source, "_empty_metadata", "")
source = replace_top_level(source, "_run_default_canonical_liveness", "")

source = replace_top_level(
    source,
    "MethodSpec",
    r'''
    def _default_method_audit(
        cell: Cell, record: Mapping[str, Any]
    ) -> e8_runtime.MethodAuditResult:
        del cell, record
        return e8_runtime.MethodAuditResult(True)


    def _empty_metadata(config: Mapping[str, Any]) -> Mapping[str, Any]:
        del config
        return {}


    @dataclass(frozen=True)
    class PaperRuntimeSpec:
        """Optional capability for methods using the canonical paper runtime."""

        liveness_grid: Callable[[Mapping[str, Any], Mapping[str, Any]], Path]
        liveness_parameter: str
        grid_paths: Callable[
            [Mapping[str, Any], Mapping[str, Any], Cell], tuple[Path, Path]
        ]
        cell_parameters: Callable[[Cell], tuple[str, float, float]]
        formula: str


    @dataclass(frozen=True)
    class MethodSpec:
        """One scientific-method adapter; generic infrastructure never branches on names."""

        name: str
        build_cell: Callable[..., Cell]
        cell_key: Callable[[Cell], str]
        parameters: Callable[[Cell], Mapping[str, Any]]
        cell_initialization: Callable[[Mapping[str, Any]], str | None]
        initialization_identity: Callable[[Mapping[str, Any]], Mapping[str, Any]]
        train_cold: Callable[..., dict[str, Any]]
        liveness_task: Callable[[Mapping[str, Any]], str]
        liveness_runner: Callable[..., dict[str, Any]]
        scientific_kernel: str
        paper_runtime: PaperRuntimeSpec | None = None
        audit_record: Callable[
            [Cell, Mapping[str, Any]], e8_runtime.MethodAuditResult
        ] = _default_method_audit
        single_aggregate_metadata: Callable[
            [Mapping[str, Any]], Mapping[str, Any]
        ] = _empty_metadata
        matrix_aggregate_metadata: Callable[
            [Mapping[str, Any]], Mapping[str, Any]
        ] = _empty_metadata
    ''',
)

source = insert_after_top_level(
    source,
    "_legacy_compatibility_columns",
    r'''
    def _method_output_columns(cell: Cell) -> dict[str, Any]:
        """Merge frozen legacy columns with the method's single parameter source."""

        columns = dict(_legacy_compatibility_columns(cell))
        for name, value in _method_spec(cell.method).parameters(cell).items():
            if name in columns and columns[name] != value:
                raise ValueError(
                    f"Method parameter {name!r} conflicts with its legacy cell value"
                )
            columns[name] = value
        return columns
    ''',
)

source = replace_top_level(
    source,
    "_dpo_method_audit",
    r'''
    def _dpo_method_audit(
        cell: Cell, record: Mapping[str, Any]
    ) -> e8_runtime.MethodAuditResult:
        del cell
        failures: list[str] = []
        if record.get("reference_initial_state_sha256") != record.get(
            "reference_terminal_state_sha256"
        ):
            failures.append("reference_state_changed")
        if record.get("reference_trainable") is not False:
            failures.append("reference_trainable")
        return e8_runtime.MethodAuditResult(
            passed=not failures,
            failure_bucket="dpo_reference_identity_failures",
            failures=tuple(failures),
        )
    ''',
)

source = replace_top_level(
    source,
    "_register_builtin_method_specs",
    r'''
    def _register_builtin_method_specs() -> None:
        exponential_paper_runtime = PaperRuntimeSpec(
            liveness_grid=lambda config, record: Path(str(record["round1_grid"])),
            liveness_parameter="representative_c",
            grid_paths=_paper_grid_paths_exponential,
            cell_parameters=_paper_params_exponential,
            formula="alpha*exp(-c*(current_sequence_surprisal/2))",
        )
        for method, build_cell, cell_key, parameters in (
            (METHOD_POSITIVE_ONLY, _build_control_cell, _positive_key, lambda cell: {}),
            (
                METHOD_GLOBAL,
                _build_control_cell,
                _global_key,
                lambda cell: {"lambda": 0.0},
            ),
            (
                METHOD_EXPONENTIAL,
                _build_exponential_cell,
                _exp_key,
                lambda cell: {"lambda": _cell_lambda(cell), "rho": cell.rho},
            ),
        ):
            _register_method_spec(
                MethodSpec(
                    name=method,
                    build_cell=build_cell,
                    cell_key=cell_key,
                    parameters=parameters,
                    cell_initialization=lambda config: None,
                    initialization_identity=_default_initialization_identity,
                    train_cold=_cold_train_paper,
                    liveness_task=lambda config: "countdown",
                    liveness_runner=lambda _method=method, **kwargs: (
                        _run_canonical_method_liveness(_method, **kwargs)
                    ),
                    scientific_kernel="canonical_old_coldstart_imports",
                    paper_runtime=exponential_paper_runtime,
                )
            )

        _register_method_spec(
            MethodSpec(
                name=METHOD_ASYMRE,
                build_cell=_build_asymre_cell,
                cell_key=_asymre_key,
                parameters=lambda cell: {"delta_v": cell.delta_v},
                cell_initialization=lambda config: None,
                initialization_identity=_default_initialization_identity,
                train_cold=_cold_train_paper,
                liveness_task=lambda config: "countdown",
                liveness_runner=lambda **kwargs: _run_canonical_method_liveness(
                    METHOD_ASYMRE, **kwargs
                ),
                scientific_kernel="canonical_old_coldstart_imports",
                paper_runtime=PaperRuntimeSpec(
                    liveness_grid=lambda config, record: _canonical_asymre_grid_path(),
                    liveness_parameter="representative_delta_v",
                    grid_paths=lambda config, record, cell: (
                        _canonical_asymre_grid_path(),
                        _canonical_asymre_grid_path(),
                    ),
                    cell_parameters=_paper_params_asymre,
                    formula="delegated_to_existing_canonical_asymre",
                ),
                single_aggregate_metadata=_asymre_single_metadata,
                matrix_aggregate_metadata=lambda config: (
                    _canonical_baseline_grid_identity(METHOD_ASYMRE)
                ),
            )
        )
        _register_method_spec(
            MethodSpec(
                name=METHOD_TOPR,
                build_cell=_build_topr_cell,
                cell_key=_topr_key,
                parameters=lambda cell: {"beta": cell.beta},
                cell_initialization=lambda config: None,
                initialization_identity=_default_initialization_identity,
                train_cold=_cold_train_paper,
                liveness_task=lambda config: "countdown",
                liveness_runner=lambda **kwargs: _run_canonical_method_liveness(
                    METHOD_TOPR, **kwargs
                ),
                scientific_kernel="canonical_old_coldstart_imports",
                paper_runtime=PaperRuntimeSpec(
                    liveness_grid=lambda config, record: _canonical_topr_grid_path(),
                    liveness_parameter="representative_c",
                    grid_paths=lambda config, record, cell: (
                        _canonical_topr_grid_path(),
                        _canonical_topr_grid_path(),
                    ),
                    cell_parameters=_paper_params_topr,
                    formula="delegated_to_existing_joint_fitted_reference_beta_topr",
                ),
                single_aggregate_metadata=_topr_single_metadata,
                matrix_aggregate_metadata=lambda config: (
                    _canonical_baseline_grid_identity(METHOD_TOPR)
                ),
            )
        )
        _register_method_spec(
            MethodSpec(
                name=METHOD_DPO,
                build_cell=_build_dpo_cell,
                cell_key=_dpo_key,
                parameters=lambda cell: {"beta": cell.beta},
                cell_initialization=lambda config: str(
                    config["dpo"]["initialization_mode"]
                ),
                initialization_identity=_dpo_initialization_identity,
                train_cold=_cold_train_dpo,
                liveness_task=lambda config: str(config["dpo"]["liveness_task"]),
                liveness_runner=_run_dpo_method_liveness,
                scientific_kernel=(
                    "historical_pr268_semantics_port_in_existing_multitask_runner"
                ),
                audit_record=_dpo_method_audit,
                single_aggregate_metadata=_dpo_single_metadata,
                matrix_aggregate_metadata=_dpo_matrix_metadata,
            )
        )
    ''',
)

source = replace_top_level(
    source,
    "_paper_grid_for_cell",
    r'''
    def _paper_grid_for_cell(
        config: Mapping[str, Any],
        record: Mapping[str, Any],
        cell: Cell,
    ) -> tuple[Path, Path]:
        paper_runtime = _method_spec(cell.method).paper_runtime
        if paper_runtime is None:
            raise RuntimeError(
                f"{cell.method} does not use the canonical paper grid runtime"
            )
        return paper_runtime.grid_paths(config, record, cell)
    ''',
)

source = replace_top_level(
    source,
    "_canonical_cold_liveness_cell",
    r'''
    def _canonical_cold_liveness_cell(grid_path: Path) -> Cell:
        """Derive one paper-runtime liveness cell through the method contract."""

        grid = yaml.safe_load(grid_path.read_text(encoding="utf-8"))
        if not isinstance(grid, dict):
            raise TypeError("Canonical liveness grid root must be a mapping")
        liveness = grid["execution"]["liveness"]
        seed_offsets = grid["sweep"]["seed_offsets"]
        method = str(liveness.get("representative_family", METHOD_EXPONENTIAL))
        spec = _method_spec(method)
        if spec.paper_runtime is None:
            raise RuntimeError(f"{method} does not use canonical paper-runtime liveness")
        parameter_key = spec.paper_runtime.liveness_parameter
        if parameter_key not in liveness:
            raise RuntimeError(
                f"Canonical liveness grid for {method} lacks {parameter_key}"
            )
        return spec.build_cell(
            task="countdown",
            method=method,
            seed=int(seed_offsets[0]),
            stage="liveness",
            value=float(liveness[parameter_key]),
            lambda_only=False,
            dpo_initialization=None,
        )
    ''',
)

source = replace_once(
    source,
    '''    method_spec = _method_spec(method)\n    if method_spec.canonical_liveness_grid is None:\n        raise RuntimeError(f"{method} does not use canonical paper-runtime liveness")\n    grid_path = method_spec.canonical_liveness_grid(config, record)\n''',
    '''    paper_runtime = _method_spec(method).paper_runtime\n    if paper_runtime is None:\n        raise RuntimeError(f"{method} does not use canonical paper-runtime liveness")\n    grid_path = paper_runtime.liveness_grid(config, record)\n''',
    "canonical liveness capability",
)

source = replace_once(
    source,
    '''    method_spec = _method_spec(cell.method)\n    grid_path, grid_source_path = _paper_grid_for_cell(config, record, cell)\n    if method_spec.paper_cell_parameters is None:\n        raise RuntimeError(\n            f"{cell.method} does not use canonical paper cell parameters"\n        )\n''',
    '''    method_spec = _method_spec(cell.method)\n    paper_runtime = method_spec.paper_runtime\n    if paper_runtime is None:\n        raise RuntimeError(\n            f"{cell.method} does not use canonical paper cell parameters"\n        )\n    grid_path, grid_source_path = _paper_grid_for_cell(config, record, cell)\n''',
    "paper training capability",
)
source = replace_once(
    source,
    '            "paper_formula": method_spec.paper_formula,\n',
    '            "paper_formula": paper_runtime.formula,\n',
    "paper formula identity",
)
source = replace_once(
    source,
    '    paper_family, alpha, coefficient = method_spec.paper_cell_parameters(cell)\n',
    '    paper_family, alpha, coefficient = paper_runtime.cell_parameters(cell)\n',
    "paper cell parameters",
)
source = replace_once(
    source,
    '        project_cell=lambda cell: _method_spec(cell.method).compatibility_columns(cell),\n',
    '        project_cell=_method_output_columns,\n',
    "plan method projection",
)
source = replace_once(
    source,
    '        cell_identity_fields=spec.compatibility_columns(cell),\n',
    '        cell_identity_fields=_method_output_columns(cell),\n',
    "recovery method projection",
)

source = replace_top_level(
    source,
    "_coldstart_result_row",
    r'''
    def _coldstart_result_row(
        config: Mapping[str, Any],
        cell: Cell,
        value: Mapping[str, Any],
        *,
        source: str,
    ) -> dict[str, Any]:
        if not _is_engineering_self_test(config) and (
            "validation_late_window_pass8_mean" not in value
            or "validation_late_window_greedy_mean" not in value
        ):
            raise RuntimeError(
                f"{cell.key} is missing the paper primary late-window metric"
            )
        return e8_results.common_result_row(
            cell,
            source=source,
            method_columns=_method_output_columns(cell),
            metrics={
                "nan_inf_failure": bool(value["nan_inf_failure"]),
                "late_window_pass8_mean": value.get(
                    "validation_late_window_pass8_mean",
                    value["validation_best_pass8"],
                ),
                "late_window_greedy_mean": value.get(
                    "validation_late_window_greedy_mean",
                    value["validation_best_greedy"],
                ),
                "best_pass8": value["validation_best_pass8"],
                "terminal_pass8": value["validation_terminal_pass8"],
                "best_greedy": value["validation_best_greedy"],
                "terminal_greedy": value["validation_terminal_greedy"],
                "best_greedy_valid_rate": value[
                    "validation_best_greedy_valid_rate"
                ],
                "terminal_greedy_valid_rate": value[
                    "validation_terminal_greedy_valid_rate"
                ],
                "best_step": value["best_step"],
                "terminal_step": value["terminal_step"],
                "stop_reason": value["stop_reason"],
            },
        )
    ''',
)
source = replace_once(
    source,
    '                bucket = _method_spec(cell.method).audit_failure_bucket\n',
    '                bucket = method_audit.failure_bucket\n',
    "audit failure bucket",
)

for forbidden in (
    "compatibility_columns",
    "canonical_liveness_grid",
    "canonical_liveness_parameter",
    "paper_grid_paths",
    "paper_cell_parameters",
    "paper_formula",
    "audit_failure_bucket",
    "MethodResultProjection",
    "ResultRecord",
):
    if forbidden in source:
        raise RuntimeError(f"stale method-contract surface remains: {forbidden}")

EXP.write_text(source, encoding="utf-8")

contract = ROOT / "tests/test_e8_method_integration_contract.sh"
text = contract.read_text(encoding="utf-8")
text = text.replace("import tempfile\n", "")
text = text.replace("import yaml\n\n", "")
compatibility_block = '''def compatibility(cell):\n    return {\n        "delta_v": None,\n        "beta": None,\n        "dpo_initialization": None,\n        "rho": None,\n        "lambda": None,\n        "temperature": parameters(cell)["temperature"],\n    }\n\n'''
if text.count(compatibility_block) != 1:
    raise RuntimeError("dummy compatibility block not found")
text = text.replace(compatibility_block, "", 1)
old_spec = '''    parameters=parameters,\n    compatibility_columns=compatibility,\n    cell_initialization=lambda config: None,\n    initialization_identity=lambda config: {},\n    train_cold=lambda cell, **kwargs: {"complete": True},\n    liveness_task=lambda config: "countdown",\n    liveness_runner=lambda **kwargs: {"complete": True},\n    canonical_liveness_grid=None,\n    canonical_liveness_parameter="representative_temperature",\n    paper_grid_paths=None,\n    paper_cell_parameters=None,\n    paper_formula="dummy_contract_only",\n    audit_record=lambda cell, record: MethodAuditResult(\n        True, {"temperature": parameters(cell)["temperature"]}\n    ),\n    audit_failure_bucket="terminal_contract_failures",\n    scientific_kernel="dummy_contract_only",\n    single_aggregate_metadata=lambda config: {},\n    matrix_aggregate_metadata=lambda config: {},\n'''
new_spec = '''    parameters=parameters,\n    cell_initialization=lambda config: None,\n    initialization_identity=lambda config: {},\n    train_cold=lambda cell, **kwargs: {"complete": True},\n    liveness_task=lambda config: "countdown",\n    liveness_runner=lambda **kwargs: {"complete": True},\n    scientific_kernel="dummy_contract_only",\n    audit_record=lambda cell, record: MethodAuditResult(True),\n'''
if text.count(old_spec) != 1:
    raise RuntimeError("dummy MethodSpec block not found")
text = text.replace(old_spec, new_spec, 1)
text = text.replace(
    '''        project_cell=lambda cell: exp_tuning._method_spec(\n            cell.method\n        ).compatibility_columns(cell),\n''',
    '''        project_cell=exp_tuning._method_output_columns,\n''',
    1,
)
text = text.replace(
    '        cell_identity_fields=compatibility(cells[0]),\n',
    '        cell_identity_fields=exp_tuning._method_output_columns(cells[0]),\n',
    1,
)
start = text.index("    with tempfile.TemporaryDirectory() as temporary:")
end_marker = '        assert parameters(live_cell) == {"temperature": 0.1}\n\n'
end = text.index(end_marker, start) + len(end_marker)
text = (
    text[:start]
    + '    assert spec.liveness_task({}) == "countdown"\n'
    + '    assert spec.liveness_runner()["complete"]\n\n'
    + text[end:]
)
text = text.replace(
    '''    assert not {\n        "group_projection",\n        "group_order",\n        "plot_columns",\n        "plot_projection",\n    }.intersection(method_spec_fields)\n''',
    '''    assert not {\n        "group_projection",\n        "group_order",\n        "plot_columns",\n        "plot_projection",\n        "compatibility_columns",\n        "canonical_liveness_grid",\n        "canonical_liveness_parameter",\n        "paper_grid_paths",\n        "paper_cell_parameters",\n        "paper_formula",\n        "audit_failure_bucket",\n    }.intersection(method_spec_fields)\n''',
    1,
)
contract.write_text(text, encoding="utf-8")

tests = ROOT / "tests/test_e8_multitask_p0.py"
text = tests.read_text(encoding="utf-8")
text = text.replace(
    "assert spec.paper_cell_parameters(cell) == (",
    "assert spec.paper_runtime is not None\n    assert spec.paper_runtime.cell_parameters(cell) == (",
)
text = text.replace(
    'assert spec.paper_formula == "delegated_to_existing_canonical_asymre"',
    'assert spec.paper_runtime.formula == "delegated_to_existing_canonical_asymre"',
)
text = text.replace(
    '''        spec.paper_formula\n        == "delegated_to_existing_joint_fitted_reference_beta_topr"\n''',
    '''        spec.paper_runtime.formula\n        == "delegated_to_existing_joint_fitted_reference_beta_topr"\n''',
)
text = text.replace(
    'assert "method_spec.paper_cell_parameters(cell)" in source',
    'assert "paper_runtime.cell_parameters(cell)" in source',
)
text = text.replace(
    'assert "method_spec.paper_formula" in source',
    'assert "paper_runtime.formula" in source',
)
tests.write_text(text, encoding="utf-8")
PY

python /tmp/trim_e8.py
