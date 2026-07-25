        "branch_count": len(rows),
        "control_count": len(summaries),
        "interpretation": (
            "500k Positive-only baseline by DRPO remoteness-scale interaction pilot; "
            "finite-horizon development-seed screening only, not convergence, steady-state "
            "ranking, held-out confirmation, or formal evidence"
        ),
        "files": {
            "branch_results": str(aggregate_dir / "branch_results.csv"),
            "training_curves": str(aggregate_dir / "training_curves_long.csv"),
            "advantage_snapshot_curves": str(aggregate_dir / "advantage_snapshot_curves_long.csv"),
            "geometry_diagnostics": str(aggregate_dir / "geometry_diagnostics_long.csv"),
            "actor_training_dynamics": str(aggregate_dir / "actor_training_dynamics_long.csv"),
            "control_summary": str(aggregate_dir / "control_summary.csv"),
            "paired_drpo_vs_positive_only": str(
                aggregate_dir / "paired_drpo_vs_positive_only_same_baseline.csv"
            ),
            "paired_shifted_drpo_vs_b0": str(
                aggregate_dir / "paired_shifted_drpo_vs_b0_same_c.csv"
            ),
            "paired_comparison_summary": str(aggregate_dir / "paired_comparison_summary.csv"),
            "finite_horizon_screening_order": str(
                aggregate_dir / "finite_horizon_screening_order.csv"
            ),
            "terminal_audit": str(aggregate_dir / "terminal_audit.json"),
        },
    }
    _atomic_json(aggregate_dir / "aggregate_summary.json", summary)
    _atomic_json(work / "RUN_SUMMARY.json", summary)
    return summary


def main() -> int:
    _, digest = _load_config(GRID)
    if COMMAND == "validate":
        payload = {
            "status": "PASS",
            "experiment_id": EXPERIMENT_ID,
            "profile_id": PROFILE_ID,
            "grid_sha256": digest,
            "dataset": DATASET_ID,
            "controls": list(CONTROL_ORDER),
            "positive_baselines": list(BASELINES),
            "drpo_remoteness_scales": list(REMOTENESS_SCALES),
            "seeds": list(SEEDS),
            "expected_controls": EXPECTED_CONTROLS,
            "expected_branches": EXPECTED_BRANCHES,
            "steps": STEPS,
            "evaluation_interval": EVAL_INTERVAL,
            "all_replay_samples_retained": True,
            "advantage_shift_before_sign_split": True,
            "paired_drpo_vs_positive_only_same_baseline": True,
            "paired_shifted_drpo_vs_b0_same_c": True,
            "raw_actor_gradient_and_adam_update_diagnostics": True,
            "near_far_effective_negative_mass_diagnostics": True,
            "held_out_seeds_touched": False,
        }
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0

    _install_profile()
    args = [
        "plan",
        "--contract",
        CONTRACT,
        "--run-spec",
        RUN_SPEC,
        "--grid",
        GRID,
        "--work-dir",
        str(WORK_ROOT),
        "--max-workers",
        MAX_WORKERS,
    ]
    suite.main(args)
    if COMMAND == "plan":
        return 0
    os.environ[FULL_RUN_ENV] = "1"
    args[0] = "run"
    args.append("--resume")
    suite.main(args)
    return 0


raise SystemExit(main())