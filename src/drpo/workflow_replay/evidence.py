import hashlib
import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Any

import yaml

from .compare import EquivalenceError, OutcomeSnapshot, compare_outcomes
from .model import CASE_ID, SHA256, CaseManifest, validate_case_manifest

TOKEN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}")
FILE_MODE = re.compile(r"[0-7]{6}")
EVENT_TERMINALS = {"run_finished": "READY", "run_blocked": "BLOCKED", "run_stale": "STALE",
                   "run_interrupted": "INTERRUPTED", "run_invalidated": "INVALIDATED"}
R1_FIELDS = {
    "comparison_mode", "expected_file_modes", "expected_authority_result", "expected_gate_results",
    "expected_diagnostic_codes", "expected_recovery_class", "workspace_rule", "evaluator_sha256",
    "evidence_schema_sha256", "order_policy",
}
MAX_BYTES, MAX_JSON_BYTES, MAX_EVENTS = 1 << 20, 1 << 18, 1000


class EvidenceError(ValueError):
    """R1 evidence validation failed closed."""


def _freeze(value: Any) -> Any:
    if isinstance(value, dict):
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_freeze(item) for item in value)
    return value


def canonical_sha256(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(raw).hexdigest()


def _strict(value: Any, label: str, fields: set[str]) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != fields:
        raise EvidenceError(f"{label} must contain exactly {sorted(fields)}")
    return value


def _token(value: Any, label: str, pattern: re.Pattern[str] = TOKEN) -> str:
    if not isinstance(value, str) or pattern.fullmatch(value) is None:
        raise EvidenceError(f"{label} has invalid syntax")
    return value


def _integer(value: Any, label: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise EvidenceError(f"{label} must be an integer >= {minimum}")
    return value


def _json_object(raw: bytes, label: str) -> dict[str, Any]:
    try:
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise EvidenceError(f"cannot decode {label}") from exc
    if not isinstance(value, dict):
        raise EvidenceError(f"{label} must be a JSON object")
    return value


@dataclass(frozen=True)
class R1CaseContract:
    sha256: str
    base: CaseManifest
    r1: Mapping[str, Any]


def validate_r1_case_contract(value: Any) -> R1CaseContract:
    root_fields = {"schema_version", "case_id", "task_class", "historical_task", "benchmark", "r1"}
    root = _strict(value, "R1 case contract", root_fields)
    if isinstance(root["schema_version"], bool) or root["schema_version"] != 2:
        raise EvidenceError("R1 case schema_version must equal integer 2")
    base_payload = {key: item for key, item in root.items() if key != "r1"}
    base_payload["schema_version"] = 1
    try:
        base = validate_case_manifest(base_payload)
    except ValueError as exc:
        raise EvidenceError(f"invalid base case contract: {exc}") from exc
    r1 = _strict(root["r1"], "r1", R1_FIELDS)
    mode = r1["comparison_mode"]
    if mode not in {"exact_artifact", "failure_boundary"}:
        raise EvidenceError("comparison_mode must be exact_artifact or failure_boundary")
    paths = tuple(base.benchmark["expected_changed_paths"])
    modes = r1["expected_file_modes"]
    if not isinstance(modes, dict) or tuple(sorted(modes)) != tuple(sorted(paths)):
        raise EvidenceError("expected_file_modes must cover expected paths exactly")
    if any(not isinstance(item, str) or FILE_MODE.fullmatch(item) is None for item in modes.values()):
        raise EvidenceError("expected_file_modes contains an invalid mode")
    gates = r1["expected_gate_results"]
    if not isinstance(gates, dict):
        raise EvidenceError("expected_gate_results must be a mapping")
    invalid_states = tuple(gates) != tuple(base.benchmark["required_gates"])
    invalid_states |= any(state not in {"PASS", "FAIL", "BLOCKED", "NOT_RUN"} for state in gates.values())
    invalid_states |= r1["expected_authority_result"] not in {"PASS", "FAIL", "BLOCKED", "NOT_RUN"}
    if invalid_states:
        raise EvidenceError("expected authority or gate results are invalid")
    diagnostics = r1["expected_diagnostic_codes"]
    if (not isinstance(diagnostics, list) or any(not isinstance(item, str) or not item for item in diagnostics) or
            diagnostics != sorted(set(diagnostics))):
        raise EvidenceError("expected_diagnostic_codes must be a sorted unique list")
    for key in ("evaluator_sha256", "evidence_schema_sha256"):
        _token(r1[key], key, SHA256)
    terminal = base.benchmark["expected_terminal_state"]
    if mode == "exact_artifact":
        invalid = terminal != "READY" or r1["workspace_rule"] != "changed_as_expected"
        invalid |= r1["expected_authority_result"] != "PASS" or set(gates.values()) != {"PASS"}
        invalid |= bool(r1["expected_diagnostic_codes"]) or r1["expected_recovery_class"] is not None
        if invalid:
            raise EvidenceError("exact_artifact expectations are inconsistent")
        _token(base.benchmark["expected_final_tree_or_semantic_hashes"].get("artifact_sha256"), "artifact_sha256", SHA256)
    else:
        if terminal not in {"BLOCKED", "STALE"} or modes or r1["workspace_rule"] != "unchanged":
            raise EvidenceError("failure_boundary requires an expected stop and unchanged workspace")
        if (not diagnostics or not isinstance(r1["expected_recovery_class"], str) or
                not r1["expected_recovery_class"].strip()):
            raise EvidenceError("failure_boundary requires diagnostic and recovery classes")
    return R1CaseContract(canonical_sha256(root), base, _freeze(dict(r1)))


def load_r1_case_contract(path: str | Path) -> R1CaseContract:
    manifest = Path(path)
    if manifest.is_symlink() or any(parent.is_symlink() for parent in manifest.parents):
        raise EvidenceError("R1 case path must not contain symlinks")
    try:
        payload = yaml.safe_load(manifest.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        raise EvidenceError("cannot read R1 case contract") from exc
    return validate_r1_case_contract(payload)


@dataclass(frozen=True)
class RunIdentity:
    case_id: str
    arm: str
    pair_id: str
    repetition: int
    order_position: int
    backend_id: str
    run_id: str

    @classmethod
    def build(cls, case_id: str, arm: str, pair_id: str, repetition: int, order_position: int,
              backend_id: str) -> "RunIdentity":
        _token(case_id, "case_id", CASE_ID)
        if arm not in {"A", "B"}:
            raise EvidenceError("arm must be A or B")
        _token(pair_id, "pair_id")
        _token(backend_id, "backend_id")
        _integer(repetition, "repetition")
        if order_position not in {0, 1}:
            raise EvidenceError("order_position must be 0 or 1")
        values = (case_id, arm, pair_id, repetition, order_position, backend_id)
        keys = ("case_id", "arm", "pair_id", "repetition", "order_position", "backend_id")
        return cls(*values, canonical_sha256(dict(zip(keys, values, strict=True))))

    @classmethod
    def from_payload(cls, value: Any) -> "RunIdentity":
        keys = ("case_id", "arm", "pair_id", "repetition", "order_position", "backend_id")
        data = _strict(value, "run_identity", {*keys, "run_id"})
        result = cls.build(*(data[key] for key in keys))
        if data["run_id"] != result.run_id:
            raise EvidenceError("run_id does not match deterministic identity")
        return result


def build_opposite_order_schedule(contract: R1CaseContract, backend_id: str) -> tuple[RunIdentity, ...]:
    if contract.r1["order_policy"] != "two_opposite_pairs":
        raise EvidenceError("unsupported order policy")
    return tuple(
        RunIdentity.build(contract.base.case_id, arm, f"pair-{repeat}", repeat, position, backend_id)
        for repeat, order in enumerate((("A", "B"), ("B", "A")))
        for position, arm in enumerate(order)
    )


@dataclass(frozen=True)
class EvidenceLocator:
    kind: str
    relative_path: str
    sha256: str
    byte_size: int

    def __post_init__(self) -> None:
        _token(self.kind, "kind")
        path = _token(self.relative_path, "relative_path", re.compile(r"[^\x00]+"))
        pure = PurePosixPath(path)
        if path.startswith(("/", "-")) or "\\" in path or pure.as_posix() != path or ".." in pure.parts:
            raise EvidenceError("evidence path is unsafe")
        _token(self.sha256, "sha256", SHA256)
        _integer(self.byte_size, "byte_size", 1)

    @classmethod
    def from_payload(cls, value: Any) -> "EvidenceLocator":
        data = _strict(value, "locator", {"kind", "relative_path", "sha256", "byte_size"})
        return cls(data["kind"], data["relative_path"], data["sha256"], data["byte_size"])

    def verify(self, root: str | Path) -> bytes:
        base = Path(root)
        if base.is_symlink() or any(parent.is_symlink() for parent in base.parents) or not base.is_dir():
            raise EvidenceError("evidence root must be a real directory")
        target = base.joinpath(*PurePosixPath(self.relative_path).parts)
        cursor = target
        while cursor != base:
            if cursor.is_symlink():
                raise EvidenceError("evidence path must not contain symlinks")
            cursor = cursor.parent
        try:
            resolved = target.resolve(strict=True)
            resolved.relative_to(base.resolve(strict=True))
        except (OSError, ValueError) as exc:
            raise EvidenceError("evidence path escapes root or is missing") from exc
        if not resolved.is_file():
            raise EvidenceError("evidence target must be a regular file")
        raw = resolved.read_bytes()
        if len(raw) != self.byte_size or len(raw) > MAX_BYTES:
            raise EvidenceError("evidence byte size mismatch or limit exceeded")
        if hashlib.sha256(raw).hexdigest() != self.sha256:
            raise EvidenceError("evidence digest mismatch")
        return raw



R2_FIELDS = {
    "mandatory_behaviors", "forbidden_regressions", "tolerances", "protected_paths",
    "evaluator_sha256", "evidence_schema_sha256", "order_policy",
}


def _sorted_tokens(value: Any, label: str, required: bool = False) -> tuple[str, ...]:
    if not isinstance(value, list) or (required and not value):
        raise EvidenceError(f"{label} must be a {'non-empty ' if required else ''}list")
    items = tuple(_token(item, f"{label}[]") for item in value)
    if items != tuple(sorted(set(items))):
        raise EvidenceError(f"{label} must be sorted and unique")
    return items


def _finite(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise EvidenceError(f"{label} must be finite")
    number = float(value)
    if number != number or number in {float("inf"), float("-inf")}:
        raise EvidenceError(f"{label} must be finite")
    return number


def _safe_paths(value: Any, label: str) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise EvidenceError(f"{label} must be a list")
    paths = tuple(value)
    if any(
        not isinstance(raw, str) or not raw or raw != raw.strip()
        for raw in paths
    ) or paths != tuple(sorted(set(paths))):
        raise EvidenceError(f"{label} must contain sorted unique paths")
    for raw in paths:
        path = PurePosixPath(raw)
        if raw.startswith(("/", "-")) or chr(92) in raw or path.as_posix() != raw or ".." in path.parts:
            raise EvidenceError(f"{label} contains an unsafe path")
    return paths

@dataclass(frozen=True)
class AcceptanceContract:
    sha256: str
    acceptance_sha256: str
    base: CaseManifest
    runtime: Mapping[str, Any]
    mandatory_behaviors: tuple[str, ...]
    forbidden_regressions: tuple[str, ...]
    tolerance_bounds: tuple[tuple[str, tuple[float | None, float | None]], ...]
    protected_paths: tuple[str, ...]

    @property
    def r1(self) -> Mapping[str, Any]:
        return self.runtime


def validate_acceptance_contract(value: Any) -> AcceptanceContract:
    root = _strict(value, "R2 case contract", {
        "schema_version", "case_id", "task_class", "historical_task", "benchmark", "acceptance",
    })
    if isinstance(root["schema_version"], bool) or root["schema_version"] != 3:
        raise EvidenceError("R2 case schema_version must equal integer 3")
    base_payload = {key: item for key, item in root.items() if key != "acceptance"}
    base_payload["schema_version"] = 1
    try:
        base = validate_case_manifest(base_payload)
    except ValueError as exc:
        raise EvidenceError(f"invalid base case contract: {exc}") from exc
    if base.benchmark["expected_terminal_state"] != "READY":
        raise EvidenceError("semantic acceptance requires READY")

    raw = _strict(root["acceptance"], "acceptance", R2_FIELDS)
    mandatory = _sorted_tokens(raw["mandatory_behaviors"], "mandatory_behaviors", True)
    forbidden = _sorted_tokens(raw["forbidden_regressions"], "forbidden_regressions")
    if set(mandatory) & set(forbidden):
        raise EvidenceError("mandatory and forbidden IDs overlap")
    tolerances = raw["tolerances"]
    if not isinstance(tolerances, dict) or not tolerances or tuple(tolerances) != tuple(sorted(tolerances)):
        raise EvidenceError("tolerances must be a sorted non-empty mapping")
    bounds = []
    for name, item in tolerances.items():
        _token(name, "tolerance name")
        bound = _strict(item, f"tolerance {name}", {"minimum", "maximum"})
        minimum = None if bound["minimum"] is None else _finite(bound["minimum"], f"{name}.minimum")
        maximum = None if bound["maximum"] is None else _finite(bound["maximum"], f"{name}.maximum")
        if minimum is None and maximum is None:
            raise EvidenceError("a tolerance requires at least one bound")
        if minimum is not None and maximum is not None and minimum > maximum:
            raise EvidenceError("tolerance minimum exceeds maximum")
        bounds.append((name, (minimum, maximum)))
    protected = _safe_paths(raw["protected_paths"], "protected_paths")
    evaluator = _token(raw["evaluator_sha256"], "evaluator_sha256", SHA256)
    schema = _token(raw["evidence_schema_sha256"], "evidence_schema_sha256", SHA256)
    if raw["order_policy"] != "two_opposite_pairs":
        raise EvidenceError("unsupported semantic order policy")
    digest = canonical_sha256(raw)
    if dict(base.benchmark["expected_final_tree_or_semantic_hashes"]) != {
        "acceptance_contract_sha256": digest, "evaluator_sha256": evaluator,
    }:
        raise EvidenceError("semantic hashes do not bind the acceptance contract")
    runtime = _freeze({
        "comparison_mode": "semantic_acceptance",
        "workspace_rule": "changed_as_expected",
        "evaluator_sha256": evaluator,
        "evidence_schema_sha256": schema,
        "order_policy": raw["order_policy"],
    })
    return AcceptanceContract(
        canonical_sha256(root), digest, base, runtime, mandatory, forbidden, tuple(bounds), protected
    )


def load_acceptance_contract(path: str | Path) -> AcceptanceContract:
    manifest = Path(path)
    if manifest.is_symlink() or any(parent.is_symlink() for parent in manifest.parents):
        raise EvidenceError("R2 case path must not contain symlinks")
    try:
        value = yaml.safe_load(manifest.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        raise EvidenceError("cannot read R2 case contract") from exc
    return validate_acceptance_contract(value)


def build_semantic_opposite_order_schedule(
    contract: AcceptanceContract, backend_id: str
) -> tuple[RunIdentity, ...]:
    if contract.r1["order_policy"] != "two_opposite_pairs":
        raise EvidenceError("unsupported semantic order policy")
    return tuple(
        RunIdentity.build(contract.base.case_id, arm, f"pair-{repeat}", repeat, position, backend_id)
        for repeat, order in enumerate((("A", "B"), ("B", "A")))
        for position, arm in enumerate(order)
    )


@dataclass(frozen=True)
class AcceptanceResult:
    case_id: str
    run_id: str
    outcome_sha256: str
    mandatory_results: tuple[tuple[str, bool], ...]
    forbidden_results: tuple[tuple[str, bool], ...]
    tolerance_values: tuple[tuple[str, float], ...]
    protected_paths_ok: bool
    diagnostic_codes: tuple[str, ...]
    accepted: bool
    failures: tuple[str, ...]
    evidence_sha256: str


def _bool_results(value: Any, label: str, keys: tuple[str, ...]) -> tuple[tuple[str, bool], ...]:
    if not isinstance(value, dict) or tuple(value) != keys or any(
        not isinstance(item, bool) for item in value.values()
    ):
        raise EvidenceError(f"{label} does not match the frozen contract")
    return tuple(value.items())


def _load_acceptance_result(
    raw: bytes, contract: AcceptanceContract, identity: RunIdentity, digest: str,
    outcome_sha256: str,
) -> AcceptanceResult:
    data = _strict(_json_object(raw, "acceptance result"), "acceptance result", {
        "schema_version", "case_id", "run_id", "outcome_sha256",
        "acceptance_contract_sha256",
        "evaluator_sha256", "mandatory_results", "forbidden_results",
        "tolerance_values", "protected_paths_ok", "diagnostic_codes",
    })
    if (
        isinstance(data["schema_version"], bool) or data["schema_version"] != 1
        or data["case_id"] != contract.base.case_id or data["run_id"] != identity.run_id
        or data["outcome_sha256"] != outcome_sha256
        or data["acceptance_contract_sha256"] != contract.acceptance_sha256
        or data["evaluator_sha256"] != contract.r1["evaluator_sha256"]
    ):
        raise EvidenceError("acceptance result identity mismatch")
    mandatory = _bool_results(data["mandatory_results"], "mandatory_results", contract.mandatory_behaviors)
    forbidden = _bool_results(data["forbidden_results"], "forbidden_results", contract.forbidden_regressions)
    metric_keys = tuple(name for name, _ in contract.tolerance_bounds)
    values = data["tolerance_values"]
    if not isinstance(values, dict) or tuple(values) != metric_keys:
        raise EvidenceError("tolerance values do not match the frozen contract")
    values = tuple((name, _finite(item, f"tolerance {name}")) for name, item in values.items())
    if not isinstance(data["protected_paths_ok"], bool):
        raise EvidenceError("protected_paths_ok must be boolean")
    diagnostics = _sorted_tokens(data["diagnostic_codes"], "diagnostic_codes")
    failures = [f"mandatory.{name}" for name, passed in mandatory if not passed]
    failures += [f"forbidden.{name}" for name, detected in forbidden if detected]
    value_map = dict(values)
    for name, (minimum, maximum) in contract.tolerance_bounds:
        if minimum is not None and value_map[name] < minimum:
            failures.append(f"tolerance.{name}.minimum")
        if maximum is not None and value_map[name] > maximum:
            failures.append(f"tolerance.{name}.maximum")
    if not data["protected_paths_ok"]:
        failures.append("protected_paths")
    return AcceptanceResult(
        data["case_id"], data["run_id"], data["outcome_sha256"],
        mandatory, forbidden, values,
        data["protected_paths_ok"], diagnostics, not failures, tuple(failures), digest,
    )


@dataclass(frozen=True)
class NormalizedRun:
    identity: RunIdentity
    evidence_sha256: tuple[tuple[str, str], ...]
    execution_terminal: str
    outcome: OutcomeSnapshot | None
    timing: tuple[tuple[str, int], ...]
    execution_valid: bool
    acceptance: AcceptanceResult | None = None


@dataclass(frozen=True)
class BoundPairReport:
    case_id: str
    equivalent: bool
    mismatches: tuple[str, ...]
    run_ids: tuple[str, str]
    evidence_sha256: tuple[tuple[tuple[str, str], ...], tuple[tuple[str, str], ...]]
    timing: tuple[tuple[tuple[str, int], ...], tuple[tuple[str, int], ...]]
    report_sha256: str


def _report_value(report: BoundPairReport) -> dict[str, Any]:
    return {"case_id": report.case_id, "equivalent": report.equivalent, "mismatches": report.mismatches, "run_ids": report.run_ids,
            "evidence_sha256": report.evidence_sha256, "timing": report.timing}


def _validate_journal(raw: bytes, identity: RunIdentity, terminal: str) -> None:
    try:
        rows = raw.decode("utf-8").splitlines()
    except UnicodeError as exc:
        raise EvidenceError("event journal is not UTF-8") from exc
    if not rows or len(rows) > MAX_EVENTS:
        raise EvidenceError("event journal is empty or oversized")
    previous, terminal_rows = -1, []
    for index, raw in enumerate(rows):
        try:
            row = _strict(
                json.loads(raw), "event",
                {"run_id", "sequence", "event", "monotonic_ns", "payload"},
            )
        except json.JSONDecodeError as exc:
            raise EvidenceError("event journal contains malformed JSON") from exc
        if row["run_id"] != identity.run_id or row["sequence"] != index:
            raise EvidenceError("event identity or sequence mismatch")
        stamp = _integer(row["monotonic_ns"], "monotonic_ns")
        if stamp < previous or not isinstance(row["payload"], dict):
            raise EvidenceError("event timestamp or payload is invalid")
        previous = stamp
        if row["event"] in EVENT_TERMINALS:
            terminal_rows.append((index, EVENT_TERMINALS[row["event"]]))
    if rows and json.loads(rows[0]).get("event") != "run_started":
        raise EvidenceError("event journal must start with run_started")
    if terminal_rows != [(len(rows) - 1, terminal)]:
        raise EvidenceError("journal must end in one matching terminal event")


def _load_outcome(raw: bytes, partial: bool) -> OutcomeSnapshot:
    fields = {
        "case_id", "terminal_state", "safety_boundary", "changed_paths", "file_modes",
        "output_hashes", "authority_result", "gate_results", "provenance",
        "diagnostic_codes", "recovery_class",
    }
    data = _strict(_json_object(raw, "outcome"), "outcome", fields)
    try:
        return OutcomeSnapshot(
            data["case_id"], data["terminal_state"], data["safety_boundary"],
            tuple(sorted(data["changed_paths"])),
            tuple(sorted(data["file_modes"].items())),
            tuple(sorted(data["output_hashes"].items())),
            data["authority_result"], tuple(data["gate_results"]),
            tuple(data["gate_results"].items()),
            tuple(sorted(data["provenance"].items())),
            tuple(sorted(data["diagnostic_codes"])), partial, data["recovery_class"],
        )
    except (AttributeError, TypeError, ValueError) as exc:
        raise EvidenceError("outcome collections are malformed") from exc


def load_run_artifact(artifact_path: str | Path, evidence_root: str | Path, contract: R1CaseContract | AcceptanceContract) -> NormalizedRun:
    root, path = Path(evidence_root), Path(artifact_path)
    try:
        if path.is_symlink() or not path.is_file() or root.is_symlink() or any(parent.is_symlink() for parent in path.parents):
            raise EvidenceError("run artifact path must not contain symlinks")
        path.resolve(strict=True).relative_to(root.resolve(strict=True))
    except (OSError, ValueError) as exc:
        raise EvidenceError("run artifact must be a real file under evidence root") from exc
    fields = {"schema_version", "case_contract_sha256", "run_identity", "identities", "evidence",
              "workspace_before_sha256", "workspace_after_sha256", "execution_terminal", "timing",
              "producer_id"}
    artifact_raw = path.read_bytes()
    if len(artifact_raw) > MAX_JSON_BYTES:
        raise EvidenceError("run artifact is oversized")
    data = _strict(_json_object(artifact_raw, "run artifact"), "run artifact", fields)
    identity = RunIdentity.from_payload(data["run_identity"])
    invalid = (isinstance(data["schema_version"], bool) or data["schema_version"] != 1 or
               identity.case_id != contract.base.case_id or data["case_contract_sha256"] != contract.sha256)
    if invalid:
        raise EvidenceError("run schema, identity, or case-contract digest mismatch")
    ids = _strict(data["identities"], "identities", {
        "base_sha", "input_sha256", "toolchain_sha", "evaluator_sha256",
        "evidence_schema_sha256", "environment_id", "cache_policy", "backend_id", "plan_sha256",
    })
    expected_ids = {
        "base_sha": contract.base.historical_task["base_sha"],
        "input_sha256": contract.base.benchmark["input_spec_sha256"],
        "toolchain_sha": contract.base.benchmark["toolchain_sha"],
        "evaluator_sha256": contract.r1["evaluator_sha256"],
        "evidence_schema_sha256": contract.r1["evidence_schema_sha256"],
        "environment_id": contract.base.benchmark["environment_id"],
        "cache_policy": contract.base.benchmark["cache_policy"],
        "backend_id": identity.backend_id,
    }
    if any(ids.get(key) != wanted for key, wanted in expected_ids.items()):
        raise EvidenceError("run artifact identity mismatch")
    _token(ids["plan_sha256"], "plan_sha256", SHA256)
    _token(data["producer_id"], "producer_id")
    terminal = data["execution_terminal"]
    before = _token(data["workspace_before_sha256"], "workspace_before_sha256", SHA256)
    after = _token(data["workspace_after_sha256"], "workspace_after_sha256", SHA256)
    partial = terminal != "READY" and before != after
    workspace_ok = before != after if contract.r1["workspace_rule"] == "changed_as_expected" else before == after
    evidence = _strict(data["evidence"], "evidence", {"event_log", "outcome", "result", "subject"})
    locators: dict[str, EvidenceLocator | None] = {}
    for key in ("event_log", "subject"):
        locators[key] = EvidenceLocator.from_payload(evidence[key])
    for key in ("outcome", "result"):
        locators[key] = None if evidence[key] is None else EvidenceLocator.from_payload(evidence[key])
    if any(item is not None and item.kind != key for key, item in locators.items()):
        raise EvidenceError("evidence locator kind does not match its role")
    subject = locators["subject"]
    assert subject is not None
    subject.verify(root)
    if subject.sha256 != contract.base.benchmark["input_spec_sha256"]:
        raise EvidenceError("subject evidence does not match frozen input identity")
    outcome_locator = locators["outcome"]
    result = locators["result"]
    acceptance = None
    expected_result = contract.base.benchmark["expected_final_tree_or_semantic_hashes"].get("artifact_sha256")
    if contract.r1["comparison_mode"] == "exact_artifact":
        if result is None or result.sha256 != expected_result:
            raise EvidenceError("result evidence does not match expected exact artifact")
        result.verify(root)
    elif contract.r1["comparison_mode"] == "semantic_acceptance":
        if result is None or outcome_locator is None or not isinstance(contract, AcceptanceContract):
            raise EvidenceError("semantic acceptance requires a result artifact")
        acceptance = _load_acceptance_result(
            result.verify(root), contract, identity, result.sha256,
            outcome_locator.sha256,
        )
    elif result is not None:
        raise EvidenceError("failure-boundary evidence must not include a result artifact")
    event = locators["event_log"]
    assert event is not None
    _validate_journal(event.verify(root), identity, terminal)
    outcome = None if outcome_locator is None else _load_outcome(outcome_locator.verify(root), partial)
    comparable = terminal in {"READY", "BLOCKED", "STALE"}
    if comparable != (outcome is not None) or (outcome and outcome.terminal_state != terminal):
        raise EvidenceError("execution terminal and outcome presence disagree")
    timing = _strict(data["timing"], "timing", {"total_ns", "child_ns", "self_overhead_ns"})
    total, child, overhead = (_integer(timing[key], key) for key in ("total_ns", "child_ns", "self_overhead_ns"))
    if child > total or overhead != total - child:
        raise EvidenceError("timing summary is inconsistent")
    digest_pairs = (("run_artifact", hashlib.sha256(artifact_raw).hexdigest()),) + tuple(
        (key, item.sha256) for key, item in locators.items() if item is not None
    )
    return NormalizedRun(
        identity, digest_pairs, terminal, outcome, tuple(sorted(timing.items())),
        comparable and workspace_ok and not partial, acceptance,
    )


def compare_normalized_runs(contract: R1CaseContract, arm_a: NormalizedRun, arm_b: NormalizedRun) -> BoundPairReport:
    mismatches: list[str] = []
    for arm, run in (("A", arm_a), ("B", arm_b)):
        if not run.execution_valid:
            mismatches.append(f"{arm}.execution_invalid")
    identities = (arm_a.identity, arm_b.identity)
    if {item.arm for item in identities} != {"A", "B"}:
        mismatches.append("pair.arms")
    for name in ("case_id", "pair_id", "repetition", "backend_id"):
        if getattr(identities[0], name) != getattr(identities[1], name):
            mismatches.append(f"pair.identity.{name}")
    if {item.order_position for item in identities} != {0, 1}:
        mismatches.append("pair.identity.order_position")
    if not mismatches and arm_a.outcome is not None and arm_b.outcome is not None:
        outcomes = {arm_a.identity.arm: arm_a.outcome, arm_b.identity.arm: arm_b.outcome}
        mismatches.extend(compare_outcomes(contract.base, outcomes["A"], outcomes["B"]).mismatches)
        wanted = (
            tuple(sorted(contract.r1["expected_file_modes"].items())),
            contract.r1["expected_authority_result"],
            tuple(contract.r1["expected_gate_results"].items()),
            tuple(contract.r1["expected_diagnostic_codes"]),
            contract.r1["expected_recovery_class"],
        )
        for arm, outcome in outcomes.items():
            actual = (
                outcome.file_modes, outcome.authority_result, outcome.gate_results,
                outcome.diagnostic_codes, outcome.recovery_class,
            )
            if actual != wanted:
                mismatches.append(f"{arm}.r1_contract")
    elif not mismatches:
        mismatches.append("pair.missing_outcome")
    mismatches = list(dict.fromkeys(mismatches))
    run_ids = tuple(item.run_id for item in identities)
    evidence = (arm_a.evidence_sha256, arm_b.evidence_sha256)
    timing = (arm_a.timing, arm_b.timing)
    draft = BoundPairReport(contract.base.case_id, not mismatches, tuple(mismatches), run_ids,
                            evidence, timing, "")
    return BoundPairReport(draft.case_id, draft.equivalent, draft.mismatches, draft.run_ids,
                           draft.evidence_sha256, draft.timing, canonical_sha256(_report_value(draft)))


def release_bound_efficiency(report: BoundPairReport, payload: Any) -> Any:
    if not report.equivalent or not isinstance(payload, dict):
        raise EquivalenceError("bound efficiency release requires an equivalent pair")
    if canonical_sha256(_report_value(report)) != report.report_sha256:
        raise EquivalenceError("pair report digest does not match its contents")
    if tuple(payload.get("run_ids", ())) != report.run_ids:
        raise EquivalenceError("efficiency run identities do not match the pair report")
    evidence = tuple(tuple(tuple(item) for item in arm) for arm in payload.get("evidence_sha256", ()))
    if evidence != report.evidence_sha256:
        raise EquivalenceError("efficiency evidence identities do not match the pair report")
    if "timing" in payload and payload["timing"] != report.timing:
        raise EquivalenceError("efficiency timing does not match the pair report")
    return report.timing


@dataclass(frozen=True)
class SemanticPairReport:
    case_id: str
    acceptance_pattern: str
    arm_acceptance: tuple[tuple[str, bool], ...]
    arm_failures: tuple[tuple[str, tuple[str, ...]], ...]
    issues: tuple[str, ...]
    pair_comparable: bool
    efficiency_release_allowed: bool
    run_ids: tuple[str, str]
    evidence_sha256: tuple[tuple[tuple[str, str], ...], tuple[tuple[str, str], ...]]
    timing: tuple[tuple[tuple[str, int], ...], tuple[tuple[str, int], ...]]
    report_sha256: str


def _semantic_report_value(report: SemanticPairReport) -> dict[str, Any]:
    return {
        "case_id": report.case_id,
        "acceptance_pattern": report.acceptance_pattern,
        "arm_acceptance": report.arm_acceptance,
        "arm_failures": report.arm_failures,
        "issues": report.issues,
        "pair_comparable": report.pair_comparable,
        "efficiency_release_allowed": report.efficiency_release_allowed,
        "run_ids": report.run_ids,
        "evidence_sha256": report.evidence_sha256,
        "timing": report.timing,
    }


def _path_in_roots(raw: str, roots: tuple[str, ...]) -> bool:
    path = PurePosixPath(raw)
    return any(path == PurePosixPath(root) or PurePosixPath(root) in path.parents for root in roots)


def _semantic_execution_valid(run: NormalizedRun, contract: AcceptanceContract) -> bool:
    outcome = run.outcome
    if not run.execution_valid or outcome is None or outcome.terminal_state != "READY":
        return False
    benchmark = contract.base.benchmark
    historical = contract.base.historical_task
    roots = tuple(benchmark["expected_changed_paths"])
    provenance = dict(outcome.provenance)
    expected_provenance = {
        "benchmark_toolchain_sha": benchmark["toolchain_sha"],
        "cache_policy": benchmark["cache_policy"],
        "environment_id": benchmark["environment_id"],
        "historical_base_sha": historical["base_sha"],
        "input_spec_sha256": benchmark["input_spec_sha256"],
    }
    if historical["frozen_implementation_sha"] is not None:
        expected_provenance["frozen_implementation_sha"] = historical["frozen_implementation_sha"]
    protected = any(
        _path_in_roots(path, contract.protected_paths)
        for path in outcome.changed_paths
    )
    return (
        outcome.case_id == contract.base.case_id
        and bool(outcome.changed_paths)
        and tuple(path for path, _ in outcome.file_modes) == outcome.changed_paths
        and bool(outcome.output_hashes)
        and all(_path_in_roots(path, roots) for path in outcome.changed_paths)
        and not protected
        and outcome.authority_result == "PASS"
        and outcome.gate_plan == tuple(benchmark["required_gates"])
        and set(dict(outcome.gate_results).values()) == {"PASS"}
        and all(provenance.get(key) == value for key, value in expected_provenance.items())
        and not outcome.diagnostic_codes
        and not outcome.partial_mutation
        and outcome.recovery_class is None
    )


def compare_semantic_runs(
    contract: AcceptanceContract, arm_a: NormalizedRun, arm_b: NormalizedRun
) -> SemanticPairReport:
    issues: list[str] = []
    identities = (arm_a.identity, arm_b.identity)
    if {item.arm for item in identities} != {"A", "B"}:
        issues.append("pair.arms")
    for name in ("case_id", "pair_id", "repetition", "backend_id"):
        if getattr(identities[0], name) != getattr(identities[1], name):
            issues.append(f"pair.identity.{name}")
    if {item.order_position for item in identities} != {0, 1}:
        issues.append("pair.identity.order_position")

    runs = {arm_a.identity.arm: arm_a, arm_b.identity.arm: arm_b}
    accepted: dict[str, bool] = {}
    failures: dict[str, tuple[str, ...]] = {}
    for arm in ("A", "B"):
        run = runs.get(arm)
        result = None if run is None else run.acceptance
        valid = run is not None and _semantic_execution_valid(run, contract)
        if not valid:
            issues.append(f"{arm}.execution_invalid")
        if result is None:
            issues.append(f"{arm}.missing_acceptance")
            accepted[arm], failures[arm] = False, ("missing_acceptance",)
        else:
            accepted[arm], failures[arm] = valid and result.accepted, result.failures

    pattern = (
        "BOTH_ACCEPTED" if accepted["A"] and accepted["B"]
        else "A_ACCEPTED_B_REJECTED" if accepted["A"]
        else "A_REJECTED_B_ACCEPTED" if accepted["B"]
        else "BOTH_REJECTED"
    )
    comparable = not issues and pattern == "BOTH_ACCEPTED"
    draft = SemanticPairReport(
        contract.base.case_id, pattern,
        tuple((arm, accepted[arm]) for arm in ("A", "B")),
        tuple((arm, failures[arm]) for arm in ("A", "B")),
        tuple(dict.fromkeys(issues)), comparable, comparable,
        tuple(item.run_id for item in identities),
        (arm_a.evidence_sha256, arm_b.evidence_sha256),
        (arm_a.timing, arm_b.timing), "",
    )
    return SemanticPairReport(
        draft.case_id, draft.acceptance_pattern, draft.arm_acceptance, draft.arm_failures,
        draft.issues, draft.pair_comparable, draft.efficiency_release_allowed,
        draft.run_ids, draft.evidence_sha256, draft.timing,
        canonical_sha256(_semantic_report_value(draft)),
    )


def release_semantic_efficiency(report: SemanticPairReport, payload: Any) -> Any:
    if not report.efficiency_release_allowed or not isinstance(payload, dict):
        raise EquivalenceError("semantic efficiency release requires two accepted arms")
    if canonical_sha256(_semantic_report_value(report)) != report.report_sha256:
        raise EquivalenceError("semantic pair report digest mismatch")
    if tuple(payload.get("run_ids", ())) != report.run_ids:
        raise EquivalenceError("semantic efficiency run identities mismatch")
    evidence = tuple(tuple(tuple(item) for item in arm) for arm in payload.get("evidence_sha256", ()))
    if evidence != report.evidence_sha256:
        raise EquivalenceError("semantic efficiency evidence identities mismatch")
    if "timing" in payload and payload["timing"] != report.timing:
        raise EquivalenceError("semantic efficiency timing mismatch")
    return report.timing
