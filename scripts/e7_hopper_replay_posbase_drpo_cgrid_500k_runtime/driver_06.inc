def _aggregate(work: Path) -> dict[str, Any]:
    branch_dirs, experiment_id = agg._branch_dirs(work)  # noqa: SLF001
    if experiment_id != EXPERIMENT_ID or len(branch_dirs) != EXPECTED_BRANCHES:
        raise RuntimeError("Positive-only/DRPO aggregate identity or branch count mismatch")
    rows: list[dict[str, Any]] = []
    curves: list[dict[str, Any]] = []
    snapshot_curves: list[dict[str, Any]] = []
    geometry_curves: list[dict[str, Any]] = []
    dynamics_curves: list[dict[str, Any]] = []

    for branch_dir in branch_dirs:
        if not (branch_dir / "COMPLETED.json").is_file():
            raise RuntimeError(f"incomplete Positive-only/DRPO branch: {branch_dir.name}")
        branch = json.loads((branch_dir / "branch_config.json").read_text())
        manifest = json.loads((branch_dir / "branch_manifest.json").read_text())
        values = branch["template_values"]
        if branch.get("experiment_id") != EXPERIMENT_ID or branch.get("profile_id") != PROFILE_ID:
            raise RuntimeError("Positive-only/DRPO branch identity mismatch")
        seed = int(branch["seed"])
        if seed not in SEEDS or seed in HELD_OUT_SEEDS:
            raise RuntimeError("Positive-only/DRPO branch used a forbidden seed")
        control_id = str(values["control_id"])
        control = CONTROL_BY_ID.get(control_id)
        if control is None:
            raise RuntimeError("unknown Positive-only/DRPO control in branch")
        method = str(values["weight_method"])
        baseline = float(values["advantage_baseline"])
        scale = None if method == "positive_only" else float(values["remoteness_scale"])
        if (
            method != control["weight_method"]
            or baseline != float(control["advantage_baseline"])
            or scale != control["remoteness_scale"]
        ):
            raise RuntimeError("Positive-only/DRPO branch control changed")
        if int(values["batch_size"]) != BATCH_SIZE:
            raise RuntimeError("Positive-only/DRPO branch batch size changed")

        summary_path = agg._only(  # noqa: SLF001
            (branch_dir / "trainer_output").glob("*_summary.json"), "trainer summary"
        )
        steps, scores = agg._read_history(json.loads(summary_path.read_text()))  # noqa: SLF001
        if tuple(steps) != EXPECTED_EVAL_STEPS:
            raise RuntimeError(f"evaluation trajectory changed for {branch_dir.name}: {steps}")
        if not all(math.isfinite(score) for score in scores):
            raise RuntimeError(f"non-finite score in {branch_dir.name}")
        early = [score for step, score in zip(steps, scores, strict=True) if EARLY_START <= step <= EARLY_END]
        middle = [score for step, score in zip(steps, scores, strict=True) if MIDDLE_START <= step <= MIDDLE_END]
        late = [score for step, score in zip(steps, scores, strict=True) if LATE_START <= step <= LATE_END]
        trend = _slope_per_100k(steps, scores, TREND_START, TREND_END)
        best_index = max(range(len(scores)), key=scores.__getitem__)

        control_manifest = manifest.get("posbase_drpo_control")
        if not isinstance(control_manifest, dict):
            raise RuntimeError("branch manifest is missing Positive-only/DRPO control")
        if float(control_manifest.get("fixed_baseline", float("nan"))) != baseline:
            raise RuntimeError("branch manifest baseline mismatch")
        if control_manifest.get("replay_sample_deletion") is not False:
            raise RuntimeError("branch deleted replay samples")
        if control_manifest.get("replay_sample_masking") is not False:
            raise RuntimeError("branch masked replay samples")
        optimizer = control_manifest.get("actor_training_dynamics", {}).get("optimizer_control", {})
        if optimizer.get("actor_lr_unchanged") is not True or optimizer.get("critic_lr_unchanged") is not True:
            raise RuntimeError("branch optimizer audit failed")

