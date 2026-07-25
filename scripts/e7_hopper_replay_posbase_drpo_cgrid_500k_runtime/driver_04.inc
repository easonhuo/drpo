        branches: list[base.Branch] = []
        for control in CONTROL_SPECS:
            method = str(control["weight_method"])
            baseline = float(control["advantage_baseline"])
            scale = control["remoteness_scale"]
            coefficient = 0.0 if scale is None else TAPER_LAMBDA / float(scale)
            for seed in SEEDS:
                branches.append(
                    base.Branch(
                        branch_id=(
                            f"{dataset.id}__seed{seed}__{control['id']}__gae__a2c__steps500k"
                        ),
                        branch_kind="injected",
                        dataset=dataset,
                        seed=seed,
                        template_values={
                            "steps": str(STEPS),
                            "stage": "positive_baseline_by_drpo_c_interaction_500k",
                            "actor_update_mode": "a2c",
                            "advantage_estimator": "gae",
                            "weight_method": method,
                            "weight_at_zero": "0" if method == "positive_only" else "1",
                            "exp_coefficient": f"{coefficient:.17g}",
                            "reference_distance": f"{REFERENCE_DISTANCE:.17g}",
                            "remoteness_threshold": f"{REMOTENESS_THRESHOLD:.17g}",
                            "remoteness_scale": "1" if scale is None else f"{float(scale):.17g}",
                            "taper_lambda": f"{TAPER_LAMBDA:.17g}",
                            "advantage_baseline": f"{baseline:.17g}",
                            "advantage_baseline_units": "raw_trajectory_snapshot_gae",
                            "control_id": str(control["id"]),
                            "batch_size": str(BATCH_SIZE),
                            "evaluation_interval": str(EVAL_INTERVAL),
                            "diagnostics_interval": str(suite.DIAGNOSTICS_INTERVAL),
                            "sampled_values_per_update": str(suite.SAMPLED_VALUES_PER_UPDATE),
                            "execution_mode": "full",
                        },
                        negative_control=None,
                    )
                )
        if len(branches) != EXPECTED_BRANCHES or len(
            {branch.branch_id for branch in branches}
        ) != EXPECTED_BRANCHES:
            raise RuntimeError("Positive-only/DRPO branch matrix is not exact")
        if {branch.seed for branch in branches} != set(SEEDS):
            raise RuntimeError("Positive-only/DRPO seed set changed")
        if {branch.seed for branch in branches} & set(HELD_OUT_SEEDS):
            raise RuntimeError("held-out seeds entered the Positive-only/DRPO matrix")
        return branches

    suite._gae_branches = gae_branches  # type: ignore[assignment]  # noqa: SLF001

    def branch_command(**kwargs: Any) -> tuple[list[str], dict[str, Any]]:
        command, branch_config = ORIGINAL_BRANCH_COMMAND(**kwargs)
        if command[1:3] != ["-m", "drpo.e7_squared_exp_night_bootstrap"]:
            raise RuntimeError("unexpected canonical bootstrap command shape")
        command = [command[0], str(WRAPPER_PATH), *command[3:]]
        branch_config["profile_id"] = PROFILE_ID
        values = branch_config["template_values"]
        branch_config["advantage_shift"] = {
            "formula": "shifted_advantage = raw_trajectory_snapshot_gae - fixed_baseline",
            "fixed_baseline": float(values["advantage_baseline"]),
            "baseline_units": "raw_trajectory_snapshot_gae",
            "shift_before_sign_split": True,
            "replay_sample_deletion": False,
            "replay_sample_masking": False,
        }
        branch_dir = Path(kwargs["branch_dir"])
        base.atomic_write_json(branch_dir / "branch_config.json", branch_config)
        return command, branch_config

    suite.branch_command = branch_command
    suite.aggregate_results = lambda work_dir: _aggregate(Path(work_dir))


