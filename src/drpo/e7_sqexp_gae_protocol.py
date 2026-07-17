"""Frozen matrix, configuration, and source RunSpec validation."""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

try:
    import yaml
except ImportError as exc:  # pragma: no cover
    raise SystemExit("PyYAML is required; install the project dependencies") from exc

from drpo.e7_sqexp_gae_artifacts import sha256_bytes, sha256_file

EXPERIMENT_ID = "EXT-H-E7-SQEXP-GAE-01"
RUNNER_VERSION = "1.0.0-first-complete-regeneration"
SCIENTIFIC_STATUS = "code_first_development_pilot_not_run"
EXPECTED_DATASETS = (
    "hopper-medium-expert-v2",
    "walker2d-medium-v2",
    "walker2d-medium-replay-v2",
)
DEVELOPMENT_SEEDS = (200, 201, 202, 203)
HELD_OUT_SEEDS = (204, 205, 206, 207)
ESTIMATORS = ("one_step_td", "behavior_gae")
ACTOR_MODES = ("a2c", "ppo_clip_k4")
EXP_COEFFICIENTS = (64.0, 128.0, 256.0)
CONTROL_IDS = ("positive_only", "sqexp_c64", "sqexp_c128", "sqexp_c256")
EXPECTED_CRITIC_JOBS = 12
EXPECTED_BRANCHES = 192
GAMMA = 0.99
GAE_LAMBDA = 0.95
EXPECTILE_TAU = 0.7
PPO_CLIP_EPSILON = 0.2
PPO_OLD_POLICY_CADENCE = 4
CRITIC_STEPS = 100_000
ACTOR_STEPS = 1_000_000
BATCH_SIZE = 256
LEARNING_RATE = 3e-4
EVALUATION_INTERVAL = 50_000
EVALUATION_EPISODES = 10
REFERENCE_DISTANCE = 2.0
LATE_WINDOW_START = 800_000

@dataclass(frozen=True)
class DatasetSpec:
    id: str
    path: str
    sha256: str
    format: str
    env_id: str
    score_protocol: str
    reference_min_score: float | None
    reference_max_score: float | None


@dataclass(frozen=True)
class FrozenProtocol:
    experiment_id: str
    run_kind: str
    datasets: tuple[str, ...]
    development_seeds: tuple[int, ...]
    held_out_seeds: tuple[int, ...]
    estimators: tuple[str, ...]
    actor_modes: tuple[str, ...]
    exp_coefficients: tuple[float, ...]
    gamma: float
    gae_lambda: float
    expectile_tau: float
    ppo_clip_epsilon: float
    ppo_old_policy_cadence: int
    critic_steps: int
    actor_steps: int
    batch_size: int
    learning_rate: float
    evaluation_interval: int
    evaluation_episodes: int
    reference_distance: float
    late_window_start: int
    source_run_spec: str
    formal_evidence_allowed: bool
    config_sha256: str


@dataclass(frozen=True)
class CriticJob:
    dataset_id: str
    seed: int

    @property
    def id(self) -> str:
        return f"{self.dataset_id}__seed{self.seed}__critic"


@dataclass(frozen=True)
class ActorBranch:
    dataset_id: str
    seed: int
    estimator: str
    actor_mode: str
    control_id: str
    coefficient: float | None

    @property
    def id(self) -> str:
        return (
            f"{self.dataset_id}__seed{self.seed}__{self.estimator}__"
            f"{self.actor_mode}__{self.control_id}__steps1m"
        )

    @property
    def pair_key(self) -> tuple[str, int, str, str]:
        return (self.dataset_id, self.seed, self.actor_mode, self.control_id)


def _read_mapping(path: str | Path) -> tuple[dict[str, Any], str]:
    source = Path(path)
    text = source.read_text(encoding="utf-8")
    raw = json.loads(text) if source.suffix.lower() == ".json" else yaml.safe_load(text)
    if not isinstance(raw, dict):
        raise ValueError(f"expected mapping in {source}")
    return raw, sha256_bytes(text.encode("utf-8"))


def load_protocol(path: str | Path) -> FrozenProtocol:
    raw, digest = _read_mapping(path)
    protocol = FrozenProtocol(
        experiment_id=str(raw.get("experiment_id")),
        run_kind=str(raw.get("run_kind")),
        datasets=tuple(str(x) for x in raw.get("datasets", ())),
        development_seeds=tuple(int(x) for x in raw.get("development_seeds", ())),
        held_out_seeds=tuple(int(x) for x in raw.get("held_out_seeds", ())),
        estimators=tuple(str(x) for x in raw.get("advantage_estimators", ())),
        actor_modes=tuple(str(x) for x in raw.get("actor_update_modes", ())),
        exp_coefficients=tuple(float(x) for x in raw.get("exp_coefficients", ())),
        gamma=float(raw.get("gamma")),
        gae_lambda=float(raw.get("gae_lambda")),
        expectile_tau=float(raw.get("expectile_tau")),
        ppo_clip_epsilon=float(raw.get("ppo_clip_epsilon")),
        ppo_old_policy_cadence=int(raw.get("ppo_old_policy_cadence")),
        critic_steps=int(raw.get("critic_steps")),
        actor_steps=int(raw.get("actor_steps")),
        batch_size=int(raw.get("batch_size")),
        learning_rate=float(raw.get("learning_rate")),
        evaluation_interval=int(raw.get("evaluation_interval")),
        evaluation_episodes=int(raw.get("evaluation_episodes")),
        reference_distance=float(raw.get("reference_distance")),
        late_window_start=int(raw.get("late_window_start")),
        source_run_spec=str(raw.get("source_run_spec")),
        formal_evidence_allowed=bool(raw.get("formal_evidence_allowed")),
        config_sha256=digest,
    )
    validate_protocol(protocol)
    return protocol


def validate_protocol(protocol: FrozenProtocol) -> None:
    expected: dict[str, Any] = {
        "experiment_id": EXPERIMENT_ID,
        "run_kind": "pilot",
        "datasets": EXPECTED_DATASETS,
        "development_seeds": DEVELOPMENT_SEEDS,
        "held_out_seeds": HELD_OUT_SEEDS,
        "estimators": ESTIMATORS,
        "actor_modes": ACTOR_MODES,
        "exp_coefficients": EXP_COEFFICIENTS,
        "gamma": GAMMA,
        "gae_lambda": GAE_LAMBDA,
        "expectile_tau": EXPECTILE_TAU,
        "ppo_clip_epsilon": PPO_CLIP_EPSILON,
        "ppo_old_policy_cadence": PPO_OLD_POLICY_CADENCE,
        "critic_steps": CRITIC_STEPS,
        "actor_steps": ACTOR_STEPS,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "evaluation_interval": EVALUATION_INTERVAL,
        "evaluation_episodes": EVALUATION_EPISODES,
        "reference_distance": REFERENCE_DISTANCE,
        "late_window_start": LATE_WINDOW_START,
    }
    for field, frozen in expected.items():
        actual = getattr(protocol, field)
        if isinstance(frozen, float):
            if not math.isclose(float(actual), frozen, rel_tol=0.0, abs_tol=1e-12):
                raise ValueError(f"frozen {field} changed: {actual!r} != {frozen!r}")
        elif actual != frozen:
            raise ValueError(f"frozen {field} changed: {actual!r} != {frozen!r}")
    if protocol.formal_evidence_allowed:
        raise ValueError("this regeneration task may not enable formal evidence")
    if set(protocol.development_seeds) & set(protocol.held_out_seeds):
        raise ValueError("development and held-out seeds overlap")
    if not protocol.source_run_spec:
        raise ValueError("source_run_spec is required")


def _dataset_value(raw: Mapping[str, Any], *names: str) -> Any:
    for name in names:
        if name in raw:
            return raw[name]
    raise ValueError(f"dataset record lacks any of {names}")


def load_source_run_spec(path: str | Path) -> tuple[tuple[DatasetSpec, ...], str]:
    raw, digest = _read_mapping(path)
    records = raw.get("datasets")
    if not isinstance(records, list):
        raise ValueError("source RunSpec must contain a datasets list")
    by_id: dict[str, DatasetSpec] = {}
    for item in records:
        if not isinstance(item, Mapping):
            raise ValueError("RunSpec dataset records must be mappings")
        dataset_id = str(_dataset_value(item, "id", "dataset_id"))
        if dataset_id not in EXPECTED_DATASETS:
            continue
        checksum = str(_dataset_value(item, "sha256", "dataset_sha256")).lower()
        if len(checksum) != 64 or any(ch not in "0123456789abcdef" for ch in checksum):
            raise ValueError(f"invalid dataset SHA-256 for {dataset_id}")
        raw_path = Path(str(_dataset_value(item, "path", "dataset_path", "relative_path"))).expanduser()
        if not raw_path.is_absolute():
            raw_path = (Path(path).expanduser().resolve().parent / raw_path).resolve()
        spec = DatasetSpec(
            id=dataset_id,
            path=str(raw_path),
            sha256=checksum,
            format=str(item.get("format", "legacy_d4rl_hdf5")),
            env_id=str(_dataset_value(item, "env_id", "evaluation_env_id")),
            score_protocol=str(item.get("score_protocol", "d4rl_v2_percent")),
            reference_min_score=(
                None
                if item.get("reference_min_score") is None
                else float(item["reference_min_score"])
            ),
            reference_max_score=(
                None
                if item.get("reference_max_score") is None
                else float(item["reference_max_score"])
            ),
        )
        if dataset_id in by_id:
            raise ValueError(f"duplicate dataset record: {dataset_id}")
        by_id[dataset_id] = spec
    missing = [name for name in EXPECTED_DATASETS if name not in by_id]
    if missing:
        raise ValueError(f"source RunSpec lacks frozen datasets: {missing}")
    return tuple(by_id[name] for name in EXPECTED_DATASETS), digest


def control_points() -> tuple[tuple[str, float | None], ...]:
    return (
        ("positive_only", None),
        ("sqexp_c64", 64.0),
        ("sqexp_c128", 128.0),
        ("sqexp_c256", 256.0),
    )


def build_critic_jobs(protocol: FrozenProtocol) -> list[CriticJob]:
    jobs = [CriticJob(dataset, seed) for dataset in protocol.datasets for seed in protocol.development_seeds]
    if len(jobs) != EXPECTED_CRITIC_JOBS or len({job.id for job in jobs}) != len(jobs):
        raise AssertionError("critic expansion must contain exactly 12 unique jobs")
    return jobs


def build_actor_branches(protocol: FrozenProtocol) -> list[ActorBranch]:
    branches = [
        ActorBranch(dataset, seed, estimator, actor_mode, control_id, coefficient)
        for dataset in protocol.datasets
        for seed in protocol.development_seeds
        for estimator in protocol.estimators
        for actor_mode in protocol.actor_modes
        for control_id, coefficient in control_points()
    ]
    ids = [branch.id for branch in branches]
    if len(branches) != EXPECTED_BRANCHES or len(ids) != len(set(ids)):
        raise AssertionError("actor expansion must contain exactly 192 unique branches")
    if any(branch.seed in HELD_OUT_SEEDS for branch in branches):
        raise AssertionError("held-out seeds entered the development matrix")
    return branches


