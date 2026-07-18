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


@dataclass(frozen=True)
class NormalizedRun:
    identity: RunIdentity
    evidence_sha256: tuple[tuple[str, str], ...]
    execution_terminal: str
    outcome: OutcomeSnapshot | None
    timing: tuple[tuple[str, int], ...]
    execution_valid: bool


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


def load_run_artifact(artifact_path: str | Path, evidence_root: str | Path, contract: R1CaseContract) -> NormalizedRun:
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
    result = locators["result"]
    expected_result = contract.base.benchmark["expected_final_tree_or_semantic_hashes"].get("artifact_sha256")
    if contract.r1["comparison_mode"] == "exact_artifact":
        if result is None or result.sha256 != expected_result:
            raise EvidenceError("result evidence does not match expected exact artifact")
        result.verify(root)
    elif result is not None:
        raise EvidenceError("failure-boundary evidence must not include a result artifact")
    event = locators["event_log"]
    assert event is not None
    _validate_journal(event.verify(root), identity, terminal)
    outcome_locator = locators["outcome"]
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
        comparable and workspace_ok and not partial,
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
