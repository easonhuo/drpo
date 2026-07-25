    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    agg._write_csv(path, rows)  # noqa: SLF001


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise RuntimeError(f"missing required JSONL file: {path}")
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if not rows:
        raise RuntimeError(f"empty required JSONL file: {path}")
    return rows


def _slope_per_100k(
    steps: Sequence[int], scores: Sequence[float], start: int, end: int
) -> float | None:
    pairs = [
        (float(step), float(score))
        for step, score in zip(steps, scores, strict=True)
        if start <= step <= end and math.isfinite(score)
    ]
    if len(pairs) < 2:
        return None
    x_mean = _mean([x for x, _ in pairs])
    y_mean = _mean([y for _, y in pairs])
    denominator = sum((x - x_mean) ** 2 for x, _ in pairs)
    if denominator <= 0.0:
        return None
    slope = sum((x - x_mean) * (y - y_mean) for x, y in pairs) / denominator
    return slope * 100_000.0


def _expected_config_controls() -> list[dict[str, Any]]:
    return [
        {
            "id": item["id"],
            "weight_method": item["weight_method"],
            "advantage_baseline": int(item["advantage_baseline"]),
            "remoteness_scale": item["remoteness_scale"],
        }
        for item in CONTROL_SPECS
    ]


def _load_config(path: str | Path) -> tuple[dict[str, Any], str]:
    source = Path(path)
    raw = json.loads(source.read_text(encoding="utf-8"))
    required = {
        "experiment_id": EXPERIMENT_ID,
        "profile_id": PROFILE_ID,
        "parent_experiment_id": "EXT-H-E7-BENCH-01",
        "predecessor_experiment_id": "EXT-H-E7-SQEXP-GAE-HOPPER-REPLAY-POSBASE-500K-01",
        "run_kind": "pilot",
        "status": "not_run",
        "scientific_status": SCIENTIFIC_STATUS,
        "dataset": DATASET_ID,
        "development_seeds": list(SEEDS),
        "held_out_seeds": list(HELD_OUT_SEEDS),
        "steps": STEPS,
        "evaluation_interval": EVAL_INTERVAL,
        "evaluation_episodes": EVAL_EPISODES,
        "actor_update_mode": "a2c",
        "advantage_mode": "gae_lambda_0p95",
        "batch_size": BATCH_SIZE,
        "actor_lr": ACTOR_LR,
        "critic_lr": CRITIC_LR,
        "positive_baselines": [int(value) for value in BASELINES],
        "drpo_remoteness_scales": list(REMOTENESS_SCALES),
        "expected_controls": EXPECTED_CONTROLS,
        "expected_branches": EXPECTED_BRANCHES,
        "screening_only": True,
        "formal_evidence_allowed": False,
    }
    changed = [key for key, value in required.items() if raw.get(key) != value]
    if changed:
        raise ValueError(f"Hopper replay Positive-only/DRPO grid changed: {changed}")
    if raw.get("controls") != _expected_config_controls():
        raise ValueError("Positive-only/DRPO control matrix changed")
    if raw.get("weight_control") != {
        "formula": THRESHOLDED_FORMULA,
        "coordinate": "normalized_squared_standardized_distance",
        "positive_only_weight_at_zero": 0.0,
        "drpo_weight_at_zero": 1.0,
        "reference_distance": REFERENCE_DISTANCE,
        "taper_lambda": TAPER_LAMBDA,
        "remoteness_threshold": REMOTENESS_THRESHOLD,
    }:
        raise ValueError("Positive-only/DRPO taper contract changed")
    shift = raw.get("advantage_shift", {})
    if shift != {
        "formula": "shifted_advantage = raw_trajectory_snapshot_gae - fixed_baseline",
        "baseline_units": "raw_trajectory_snapshot_gae",
        "shift_applied_before_sign_split": True,
        "replay_sample_deletion": False,
        "replay_sample_masking": False,
        "positive_actor_rule": "shifted_advantage > 0 remains unattenuated",
        "negative_actor_rule": (
