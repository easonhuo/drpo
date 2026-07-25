            "positive_only zeros shifted_advantage < 0; DRPO applies the detached remoteness taper"
        ),
    }:
        raise ValueError("Positive-only/DRPO advantage-shift semantics changed")
    return raw, sha256_file(source)


def _reset_profile_hooks() -> None:
    suite.configure_execution = ORIGINAL_CONFIGURE
    suite.load_grid = ORIGINAL_LOAD_GRID
    suite.load_run_spec = ORIGINAL_LOAD_RUN_SPEC
    suite._gae_branches = ORIGINAL_GAE_BRANCHES  # noqa: SLF001
    suite.branch_command = ORIGINAL_BRANCH_COMMAND
    suite.aggregate_results = ORIGINAL_AGGREGATE
    suite._is_tuning = ORIGINAL_IS_TUNING  # noqa: SLF001
    suite._is_p3 = ORIGINAL_IS_P3  # noqa: SLF001
    suite.active_scientific_status = ORIGINAL_ACTIVE_STATUS
    suite.active_expected_branch_count = ORIGINAL_ACTIVE_COUNT


def _install_profile() -> None:
    _reset_profile_hooks()
    suite.GAE_EXPERIMENT_ID = EXPERIMENT_ID
    suite.TUNING_PROFILE_ID = PROFILE_ID
    suite.TUNING_SEEDS = SEEDS
    suite.TUNING_EXPECTED_BRANCHES = EXPECTED_BRANCHES
    suite.TUNING_RUNNER_VERSION = RUNNER_VERSION
    suite.TUNING_FULL_RUN_ENV = FULL_RUN_ENV
    suite.EXPECTED_STEPS = STEPS
    agg.GAE_EXPERIMENT_ID = EXPERIMENT_ID

    def is_tuning() -> bool:
        return (
            suite._ACTIVE_EXPERIMENT_ID == EXPERIMENT_ID  # noqa: SLF001
            and suite._ACTIVE_PROFILE_ID == PROFILE_ID  # noqa: SLF001
        )

    suite._is_tuning = is_tuning  # type: ignore[assignment]  # noqa: SLF001
    suite._is_p3 = lambda: False  # type: ignore[assignment]  # noqa: SLF001
    suite.active_scientific_status = lambda: SCIENTIFIC_STATUS
    suite.active_expected_branch_count = lambda: EXPECTED_BRANCHES

    def configure_execution(
        grid_path: str | Path,
        *,
        liveness_pair: bool = False,
        liveness_steps: int | None = None,
    ) -> None:
        if liveness_pair or liveness_steps is not None:
            raise ValueError("the 500k Positive-only/DRPO profile has no liveness submatrix")
        loaded, _ = _load_config(grid_path)
        suite._ACTIVE_EXPERIMENT_ID = str(loaded["experiment_id"])  # noqa: SLF001
        suite._ACTIVE_PROFILE_ID = str(loaded["profile_id"])  # noqa: SLF001
        suite._LIVENESS_STEPS = None  # noqa: SLF001

    suite.configure_execution = configure_execution
    suite.load_grid = _load_config

    def load_run_spec(path: str | Path) -> tuple[dict[str, Any], str]:
        loaded, digest = ORIGINAL_LOAD_RUN_SPEC(path)
        loaded = copy.deepcopy(loaded)
        by_id = {str(item["id"]): item for item in loaded["datasets"]}
        if DATASET_ID not in by_id:
            raise ValueError("canonical run spec is missing hopper-medium-replay-v2")
        loaded["datasets"] = [copy.deepcopy(by_id[DATASET_ID])]
        loaded["seeds"] = list(SEEDS)
        argv = [str(value) for value in loaded["trainer_argv_template"]]
        expected_source = {
            "--batch": "256",
            "--lr": "0.0003",
            "--eval_interval": "50000",
            "--eval_episodes": "10",
            "--steps": "{steps}",
        }
        for flag, expected in expected_source.items():
            if suite._flag_value(argv, flag) != expected:  # noqa: SLF001
                raise ValueError(f"canonical source {flag} changed before 500k injection")
        argv[argv.index("--batch") + 1] = "{batch_size}"
        argv[argv.index("--eval_interval") + 1] = "{evaluation_interval}"
        loaded["trainer_argv_template"] = argv
        return loaded, digest

    suite.load_run_spec = load_run_spec

    def gae_branches(
        run_spec: Mapping[str, Any], grid: Mapping[str, Any]
    ) -> list[base.Branch]:
        del grid
        datasets = [base.DatasetSpec.from_mapping(item) for item in run_spec["datasets"]]
        if len(datasets) != 1 or datasets[0].id != DATASET_ID:
            raise ValueError("Positive-only/DRPO profile must contain only hopper-medium-replay-v2")
        dataset = datasets[0]
