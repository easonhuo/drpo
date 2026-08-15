#!/usr/bin/env bash
set -Eeuo pipefail

python - <<'PY'
from pathlib import Path


def replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise SystemExit(f"{label}: expected one exact match, found {count}")
    return text.replace(old, new, 1)


def replace_region(text: str, start: str, end: str, new: str, label: str, *, after: int = 0) -> str:
    i = text.find(start, after)
    if i < 0:
        raise SystemExit(f"{label}: start marker not found")
    j = text.find(end, i + len(start))
    if j < 0:
        raise SystemExit(f"{label}: end marker not found")
    return text[:i] + new + text[j:]


source_path = Path("src/drpo/e8_multitask_exp_tuning.py")
text = source_path.read_text(encoding="utf-8")

# 1) Restore all nine tasks in the cold-start suite.
old_suite_start = '    else:\n        expected_tasks = set(TASK_NAMES) - {"spiral_matrix"}\n'
old_suite_end = '    reference = config["reference"]\n'
new_suite = '''    else:
        expected_tasks = set(TASK_NAMES)
        if (
            current_experiment != COLDSTART_EXPERIMENT_ID
            or len(tasks) != 9
            or len(set(tasks)) != 9
            or set(tasks) != expected_tasks
        ):
            raise ValueError("The cold-start suite must be Countdown plus the exact eight P0 tasks")
        if set(config["suite"].get("p0_tasks", ())) != expected_tasks - {"countdown"}:
            raise ValueError("Cold-start suite.p0_tasks must be the exact eight P0 tasks")
        if tuple(config["suite"].get("external_tasks", ())) != ("countdown",):
            raise ValueError("Countdown must be the only cold-start external task")
'''
text = replace_region(text, old_suite_start, old_suite_end, new_suite, "cold suite")

# 2) Cold-start negative-bank contract: fixed reference-remoteness selection is provenance only.
negative_start = '    negative = config["negative_sampling"]\n'
negative_end = '    calibration = config["remoteness_calibration"]\n'
new_negative = '''    negative = config["negative_sampling"]
    if int(negative["negatives_per_prompt"]) != 16:
        raise ValueError("Every training prompt must retain exactly 16 negatives")
    if _is_coldstart(config):
        if (
            negative.get("consumer") != "all_unique_negatives_per_prompt"
            or negative.get("deduplicate_rule")
            != "first_canonical_completion_occurrence"
            or negative.get("denominator") != "unique_negative_count_per_prompt"
            or bool(negative.get("near_far_selection", True))
            or bool(negative.get("weight_sum_normalization", True))
            or bool(negative.get("gradient_budget_matching", True))
        ):
            raise ValueError("Cold-start negative consumption must match the paper trainer")
        reference_bank = negative.get("reference_remoteness_bank")
        if not isinstance(reference_bank, Mapping):
            raise ValueError("Cold-start requires the derived reference-remoteness bank contract")
        expected_reference_bank = {
            "enabled": True,
            "scope": "non_countdown_training_only",
            "source_candidates": "all_deterministic_verified_wrong_mutations",
            "reference_policy": "zero_update_base_plus_fresh_lora",
            "coordinate": "mean_completion_token_surprisal",
            "selected_negatives_per_prompt": 16,
            "selection": "evenly_spaced_reference_rank_including_extremes",
            "coverage_threshold": None,
            "reference_rank_role": "provenance_and_diagnostic_only",
            "static_reference_rank_enters_training_weight": False,
            "current_policy_surprisal_recomputed_each_update": True,
            "original_p0_bank_preserved": True,
            "audit_quantiles": ["min", "q25", "median", "q75", "max"],
        }
        if dict(reference_bank) != expected_reference_bank:
            raise ValueError("Reference-remoteness bank selection contract drifted")
    else:
        if _tuple_floats(negative["near_far_mix"]) != (0.5, 0.5):
            raise ValueError("Near/far branch mass must remain 0.5/0.5")
        if not bool(negative["selection_stop_gradient"]):
            raise ValueError("Current near/far selection must be stop-gradient")
        if bool(negative["weight_sum_normalization"]):
            raise ValueError("Weight-sum normalization is forbidden")

'''
text = replace_region(text, negative_start, negative_end, new_negative, "negative contract")

# 3) Replace the stale cold-start sweep/runtime/result-gate validator.
sweep_anchor = text.index('    sweep = config["sweep"]\n')
cold_sweep_start = '    else:\n        task_lambda = sweep.get("task_lambda")\n'
cold_sweep_end = '    execution = config["execution"]\n'
new_cold_sweep = '''    else:
        task_lambda = sweep.get("task_lambda")
        if not isinstance(task_lambda, Mapping) or set(task_lambda) != set(tasks):
            raise ValueError("Cold-start task_lambda must contain the exact nine tasks")
        countdown_sentinels = (
            0.105360516,
            0.430782916,
            0.916290732,
            1.897119985,
            2.302585093,
            2.995732274,
        )
        if _tuple_floats(sweep.get("countdown_sentinel_coefficients", ())) != countdown_sentinels:
            raise ValueError("Countdown sentinel coefficients drifted")
        if _task_lambdas(config, "countdown") != countdown_sentinels:
            raise ValueError("Countdown task_lambda must equal the six diagnostic sentinels")
        if tuple(int(value) for value in sweep.get("countdown_seed_offsets", ())) != (
            PAPER_SEED_OFFSETS
        ):
            raise ValueError("Countdown must preserve the two paper seed offsets")
        transfer_positive_seeds = tuple(
            int(value) for value in sweep.get("transfer_positive_only_seed_offsets", ())
        )
        if transfer_positive_seeds != (4000, 5000, 6000, 7000):
            raise ValueError("Transfer Positive-only must use the four frozen historical seed streams")
        if int(sweep.get("task_transfer_seed_offset", -1)) != 4000:
            raise ValueError("Transfer Exp response-shape localization must use seed 4000")
        if int(sweep.get("tuning_seed", -1)) != 4000:
            raise ValueError("Cold-start tuning_seed must remain 4000")

        transfer_tasks = set(tasks) - {"countdown"}
        grid_hashes = sweep.get("task_grid_hashes")
        provenance = sweep.get("task_grid_provenance")
        expected_hashes = {
            "word_sorting": "d24edbd6099f1d4f081318b305e62b39834db7ab94af6b4279d706f94e8d6de3",
            "spiral_matrix": "805966b9e3e1774e748d1d96ca64667e23276782681e15c1e072e5536a02199a",
            "mini_sudoku": "2340c4729b70ae5bafb7b6bf07049f38056b18368749bba3b967b4bbd950e29c",
            "maze": "399732f8572faf670a0486cd5f838d3956bfa56cabbb5ae79b8521fbbfc45d33",
            "word_ladder": "4fff6be5b923b9071ae0ee949e734087330463072d9be02015a661e51a2689e4",
            "knights_knaves": "37f675c933e909ac1ca8a6464c8ed72497758b31d1c8a82c855cad10961c4cab",
            "graph_color": "65c02a38a4339888d2e485c277fa1668a5e0309fab857e3ae2b2c0dc862f9d",
            "wikisql": "e42e516dbbe0bfd0f9fd00f5dec22949f88503253a2235c8c81bd908af4779a0",
        }
        if not isinstance(grid_hashes, Mapping) or dict(grid_hashes) != expected_hashes:
            raise ValueError("Transfer task-grid hashes drifted")
        if not isinstance(provenance, Mapping) or set(provenance) != transfer_tasks:
            raise ValueError("Transfer task-grid provenance must cover the exact eight tasks")
        if any(not str(provenance[task]).strip() for task in transfer_tasks):
            raise ValueError("Transfer task-grid provenance entries must be non-empty")
        for task in transfer_tasks:
            values = _task_lambdas(config, str(task))
            if len(values) != 20 or len(set(values)) != 20:
                raise ValueError(f"{task} must contain exactly 20 distinct Exp coefficients")
            if stable_hash(list(values)) != str(grid_hashes[task]):
                raise ValueError(f"{task} coefficient grid does not match its locked hash")
        if int(sweep.get("expected_cells", -1)) != 208:
            raise ValueError("Cold-start must contain 16 Countdown plus 8x24 transfer cells")

        initialization = config.get("initialization", {})
        if (
            initialization.get("source") != "base_model"
            or int(initialization.get("optimizer_updates", -1)) != 0
            or bool(initialization.get("external_adapter_allowed", True))
        ):
            raise ValueError("Cold-start must use a zero-update base-model LoRA initialization")
        canonical = config.get("canonical_coldstart", {})
        expected_paths = {
            "arena": "src/drpo/countdown_qwen_arena_onefile.py",
            "scan_common": "src/drpo/countdown_e8_alpha1_c_scan_common.py",
            "scan_runtime": "src/drpo/countdown_e8_alpha1_c_scan_runtime.py",
            "scan_trainer": "src/drpo/countdown_e8_alpha1_c_scan_trainer.py",
            "paper_common": "src/drpo/countdown_e8_alpha1_highc_scan_common.py",
            "paper_runtime": "src/drpo/countdown_e8_alpha1_highc_scan_runtime.py",
            "base_config": "configs/countdown_e8_base_rl_replay_0p5b.yaml",
            "round1_grid": (
                "configs/countdown_e8_oracle_offline_v2_alpha1_highc_scan_0p5b.yaml"
            ),
            "extension_grid": (
                "configs/countdown_e8_oracle_offline_v2_linear_c_extension_0p5b.yaml"
            ),
            "bank_generator": "src/drpo/countdown_e8_oracle_bank_v2.py",
            "bank_config": "configs/countdown_e8_oracle_offline_bank_v2_0p5b.yaml",
            "bank_converter": "scripts/v2_bank_convert.py",
            "p0_bank_pipeline": "src/drpo/e8_multitask_p0.py",
            "p0_task_adapters": "src/drpo/e8_multitask_tasks.py",
            "p0_config": "configs/e8_multitask_p0.yaml",
            "p0_launcher": "scripts/run_e8_multitask_p0.sh",
            "result_reference": (
                "experiments/results/e8_paper_aligned_linear_scan_round1_pilot/"
                "RESULT_SUMMARY.json"
            ),
        }
        if canonical.get("paths") != expected_paths:
            raise ValueError("Cold-start canonical paths must point to the old implementation")
        blob_shas = canonical.get("expected_git_blob_shas", {})
        if set(blob_shas) != set(expected_paths) or any(
            len(str(value)) != 40 for value in blob_shas.values()
        ):
            raise ValueError("Cold-start must pin every old source/config Git blob SHA")
        if canonical.get("scientific_kernel") != "import_only_no_loss_reimplementation":
            raise ValueError("Cold-start scientific kernel must be imported, not reimplemented")
        if canonical.get("countdown_entry") != "countdown_e8_alpha1_highc_scan_runtime.worker":
            raise ValueError("Countdown dispatch must remain the exact paper worker")
        if canonical.get("transfer_entry") != "countdown_e8_alpha1_c_scan_trainer.train_cell":
            raise ValueError("Transfer dispatch must remain the locked paper trainer")

        runtime = config.get("task_runtime", {})
        expected_whitelist = (
            "model.max_length",
            "model.max_new_tokens",
            "evaluation.batch_size",
            "evaluation.greedy_prompt_rows",
            "evaluation.passk_prompt_rows",
            "evaluation.pass_ks",
        )
        if tuple(runtime.get("override_whitelist", ())) != expected_whitelist or set(runtime) != {
            "override_whitelist",
            *tasks,
        }:
            raise ValueError("Task runtime overrides must use the exact six-field whitelist")
        for task in tasks:
            task_runtime = runtime[task]
            if set(task_runtime) != {
                "max_length",
                "max_new_tokens",
                "evaluation_batch_size",
                "greedy_prompt_rows",
                "passk_prompt_rows",
                "auxiliary_pass_ks",
            }:
                raise ValueError(f"{task} contains a non-whitelisted runtime override")
            is_countdown = task == "countdown"
            expected_length = 256 if is_countdown else 512
            expected_new_tokens = 80 if is_countdown else 128
            expected_batch = 8 if is_countdown else 16
            expected_passk_rows = 500 if is_countdown else 128
            expected_aux = (64,) if is_countdown else ()
            if (
                int(task_runtime["max_length"]) != expected_length
                or int(task_runtime["max_new_tokens"]) != expected_new_tokens
                or int(task_runtime["evaluation_batch_size"]) != expected_batch
                or int(task_runtime["greedy_prompt_rows"]) != 500
                or int(task_runtime["passk_prompt_rows"]) != expected_passk_rows
                or tuple(int(value) for value in task_runtime["auxiliary_pass_ks"])
                != expected_aux
            ):
                raise ValueError(f"{task} task-interface length/evaluation contract drifted")

        selection = config.get("selection", {})
        if (
            selection.get("primary_metric") != "validation_late_window_pass8_mean"
            or not bool(selection.get("finite_required", False))
            or selection.get("terminal_valid_rate_role")
            != "diagnostic_only_not_selection_eligibility"
            or tuple(selection.get("tie_breakers", ()))
            != (
                "validation_terminal_pass8",
                "validation_late_window_greedy_mean",
                "smaller_lambda",
            )
        ):
            raise ValueError("Cold-start selection policy drifted")

'''
text = replace_region(
    text,
    cold_sweep_start,
    cold_sweep_end,
    new_cold_sweep,
    "cold sweep validator",
    after=sweep_anchor,
)

execution_start = '    execution = config["execution"]\n'
execution_end = '\n\ndef coefficient_from_rho'
new_execution = '''    execution = config["execution"]
    expected_capacity = 16
    if int(execution["max_concurrent_cells"]) != expected_capacity:
        raise ValueError(f"The scheduler must expose exactly {expected_capacity} slots")
    if tuple(int(value) for value in execution["gpu_ids"]) != tuple(range(8)):
        raise ValueError("The default GPU pool must remain 0--7")
    expected_waves = 13 if profile == SWEEP_PROFILE_COLDSTART else (
        7 if profile == SWEEP_PROFILE_DENSE else 5
    )
    if int(execution["slots_per_gpu"]) != 2 or int(execution["expected_waves"]) != expected_waves:
        raise ValueError(
            f"The frozen topology is two slots per GPU and {expected_waves} waves"
        )
    if _is_coldstart(config) and execution.get("scheduler") != "dynamic_slot_queue":
        raise ValueError("Cold-start execution must use the recovery-aware slot scheduler")
    if _is_coldstart(config) and (
        not bool(execution.get("wave_barriers", False))
        or not bool(execution.get("identity_checked_resume", False))
        or not bool(execution.get("retry_incomplete_requires_explicit_flag", False))
        or not bool(execution.get("fail_closed", False))
        or not bool(execution.get("test_partition_forbidden", False))
        or execution.get("oom_policy")
        != "fail_cell_no_automatic_scientific_parameter_mutation"
    ):
        raise ValueError("Cold-start recovery/OOM safety contract drifted")
'''
text = replace_region(text, execution_start, execution_end, new_execution, "execution validator")

# 4) 208-cell geometry and 13 exact 16-cell scheduling waves.
build_cells_start = '    if _is_coldstart(config):\n        cells: list[Cell] = []\n'
build_cells_end = '    coarse = _tuple_floats(config["sweep"]["coarse_rho"])\n'
new_build_cells = '''    if _is_coldstart(config):
        cells: list[Cell] = []
        countdown_coefficients = _task_lambdas(config, "countdown")
        for seed_offset in tuple(int(value) for value in config["sweep"]["countdown_seed_offsets"]):
            cells.append(
                Cell("countdown", METHOD_POSITIVE_ONLY, None, seed_offset, "countdown_sentinel")
            )
            cells.append(
                Cell("countdown", METHOD_GLOBAL, 1.0, seed_offset, "countdown_sentinel", 0.0)
            )
            cells.extend(
                Cell(
                    "countdown",
                    METHOD_EXPONENTIAL,
                    math.exp(-coefficient),
                    seed_offset,
                    "countdown_sentinel",
                    coefficient,
                )
                for coefficient in countdown_coefficients
            )
        positive_seeds = tuple(
            int(value) for value in config["sweep"]["transfer_positive_only_seed_offsets"]
        )
        exp_seed = int(config["sweep"]["task_transfer_seed_offset"])
        for task in tasks:
            if task == "countdown":
                continue
            cells.extend(
                Cell(task, METHOD_POSITIVE_ONLY, None, seed_offset, "task_transfer")
                for seed_offset in positive_seeds
            )
            cells.extend(
                Cell(
                    task,
                    METHOD_EXPONENTIAL,
                    math.exp(-coefficient),
                    exp_seed,
                    "task_transfer",
                    coefficient,
                )
                for coefficient in _task_lambdas(config, task)
            )
        result = tuple(cells)
        if len(result) != 208 or len({cell.key for cell in result}) != 208:
            raise AssertionError("Internal 208-cell cold-start identity failure")
        return result
'''
text = replace_region(text, build_cells_start, build_cells_end, new_build_cells, "build_cells")

old_waves = '''    if _is_coldstart(config):
        waves = tuple(
            tuple(cells[index : index + capacity]) for index in range(0, len(cells), capacity)
        )
        if len(waves) != 24 or tuple(len(wave) for wave in waves) != (8,) * 23 + (4,):
            raise AssertionError("Cold-start nominal geometry must be 23x8 plus one 4-cell batch")
        return waves
'''
new_waves = '''    if _is_coldstart(config):
        waves = tuple(
            tuple(cells[index : index + capacity]) for index in range(0, len(cells), capacity)
        )
        if len(waves) != 13 or any(len(wave) != 16 for wave in waves):
            raise AssertionError("Cold-start geometry must be 13 exact 16-cell scheduling waves")
        return waves
'''
text = replace_once(text, old_waves, new_waves, "build_waves")
text = replace_once(
    text,
    '        "wave_is_scheduling_barrier": not _is_coldstart(config),\n',
    '        "wave_is_scheduling_barrier": bool(config["execution"].get("wave_barriers", False)),\n',
    "plan barrier marker",
)

# 5) Transfer task base config: 512/128, eval batch 16, Pass@8 only.
old_base_override = '''    derived = copy.deepcopy(original)
    runtime = config["task_runtime"][task]
    derived["model"]["max_length"] = int(runtime["max_length"])
    derived["model"]["max_new_tokens"] = int(runtime["max_new_tokens"])
    derived["evaluation"]["batch_size"] = int(runtime["evaluation_batch_size"])
    allowed = {
        "model.max_length",
        "model.max_new_tokens",
        "evaluation.batch_size",
    }
'''
new_base_override = '''    derived = copy.deepcopy(original)
    runtime = config["task_runtime"][task]
    derived["model"]["max_length"] = int(runtime["max_length"])
    derived["model"]["max_new_tokens"] = int(runtime["max_new_tokens"])
    derived["evaluation"]["batch_size"] = int(runtime["evaluation_batch_size"])
    derived["evaluation"]["pass_ks"] = [8] + [
        int(value) for value in runtime["auxiliary_pass_ks"]
    ]
    allowed = {
        "model.max_length",
        "model.max_new_tokens",
        "evaluation.batch_size",
        "evaluation.pass_ks",
    }
'''
text = replace_once(text, old_base_override, new_base_override, "task base overrides")

# 6) Derived non-Countdown training bank, selected once by zero-update reference surprisal.
insert_marker = '\n\ndef write_canonical_cold_inputs(\n'
helpers = r'''

def _evenly_spaced_rank_indices(candidate_count: int, selected_count: int = 16) -> tuple[int, ...]:
    if selected_count < 2:
        raise ValueError("Reference-remoteness selection requires at least two selected ranks")
    if candidate_count < selected_count:
        raise ValueError(
            f"Reference-remoteness selection requires >= {selected_count} candidates; "
            f"found {candidate_count}"
        )
    indices = tuple(
        (index * (candidate_count - 1)) // (selected_count - 1)
        for index in range(selected_count)
    )
    if len(set(indices)) != selected_count or indices[0] != 0 or indices[-1] != candidate_count - 1:
        raise AssertionError("Even rank selection must be unique and include both extremes")
    return indices


def _reference_surprisal_summary(values: Sequence[float]) -> dict[str, float]:
    array = np.asarray([float(value) for value in values], dtype=float)
    if array.size == 0 or not np.all(np.isfinite(array)):
        raise RuntimeError("Reference-surprisal audit requires finite non-empty values")
    q25, median, q75 = np.quantile(array, [0.25, 0.5, 0.75])
    return {
        "min": float(array.min()),
        "q25": float(q25),
        "median": float(median),
        "q75": float(q75),
        "max": float(array.max()),
        "range": float(array.max() - array.min()),
        "iqr": float(q75 - q25),
    }


def _verified_wrong_candidates(
    adapter: Any,
    instance: TaskInstance,
    source_row: Mapping[str, Any],
) -> list[dict[str, Any]]:
    generation_seed = int(source_row["generation_seed"])
    rng = random.Random(
        int(
            stable_hash(
                {
                    "task": str(source_row["task"]),
                    "prompt_id": str(source_row["prompt_id"]),
                    "seed": generation_seed,
                }
            )[:16],
            16,
        )
    )
    candidates: list[dict[str, Any]] = []
    seen: set[str] = set()
    for mutation in adapter.mutation_candidates(instance, rng):
        result = adapter.verify(
            instance,
            mutation.completion,
            mutation_class=mutation.mutation_class,
        )
        canonical = str(result.canonical_completion)
        if canonical in seen or not adapter.accept_negative(result):
            continue
        seen.add(canonical)
        candidates.append(
            {
                "completion": str(mutation.completion),
                "canonical_completion": canonical,
                "verifier_score": float(result.score),
                "binary_correct": bool(result.correct),
                "format_valid": bool(result.format_valid),
                "error_class": str(result.error_class),
                "verification_details": dict(result.details),
            }
        )
    original = {
        str(item.get("canonical_completion", item["completion"]))
        for item in source_row["negatives"]
    }
    missing = sorted(original - seen)
    if missing:
        raise RuntimeError(
            f"{source_row['task']}/{source_row['prompt_id']} cannot reconstruct the original "
            f"P0 negative universe: {missing[:3]}"
        )
    if len(candidates) < 16:
        raise RuntimeError(
            f"{source_row['task']}/{source_row['prompt_id']} has only {len(candidates)} "
            "deterministic verified wrong candidates"
        )
    return candidates


def _score_reference_candidates(
    *,
    arena: Any,
    model: Any,
    tokenizer: Any,
    prompt: str,
    candidates: Sequence[Mapping[str, Any]],
    max_length: int,
    batch_size: int,
) -> list[float]:
    device = next(model.parameters()).device
    scores: list[float] = []
    for start in range(0, len(candidates), batch_size):
        chunk = candidates[start : start + batch_size]
        encoded = [
            arena.encode_prompt_completion(
                tokenizer,
                prompt,
                str(item["canonical_completion"]),
                max_length,
            )
            for item in chunk
        ]
        packed = arena.pad_encoded(encoded, int(tokenizer.pad_token_id))
        packed = arena.move_to_device(packed, device)
        with torch.no_grad():
            surprisal = -arena.completion_stats(model, packed)["seq_lp"]
        scores.extend(float(value) for value in surprisal.detach().cpu())
    if len(scores) != len(candidates) or not all(math.isfinite(value) for value in scores):
        raise RuntimeError("Reference-policy candidate scoring is incomplete or non-finite")
    return scores


def _derive_reference_remoteness_banks(
    config: Mapping[str, Any],
    output_root: Path,
    *,
    base_model_path: str,
) -> dict[str, Any]:
    """Create the fixed training bank without modifying the model-independent P0 bank."""

    if not _is_coldstart(config) or _is_engineering_self_test(config):
        raise RuntimeError("Reference-remoteness bank derivation is formal cold-start only")
    if torch is None:
        raise RuntimeError("Reference-remoteness bank derivation requires Torch")
    splits, inputs = _load_prepared(output_root, config)
    selector = dict(config["negative_sampling"]["reference_remoteness_bank"])
    base_identity = model_identity(base_model_path, None)["model"]
    pending: list[str] = []
    identities: dict[str, str] = {}
    for task_value in config["suite"]["p0_tasks"]:
        task = str(task_value)
        record = splits["tasks"][task]
        identity = stable_hash(
            {
                "schema_version": 1,
                "experiment_id": experiment_id(config),
                "config_hash": stable_config_hash(config),
                "task": task,
                "source_train_sha256": sha256_file(Path(record["paths"]["train"])),
                "source_bank_sha256": record["bank_sha256"],
                "p0_config_sha256": record["p0_config_sha256"],
                "base_model_identity": base_identity,
                "task_runtime": dict(config["task_runtime"][task]),
                "selector": selector,
            }
        )
        identities[task] = identity
        existing = record.get("reference_remoteness_bank")
        if isinstance(existing, Mapping):
            path = Path(str(existing.get("path", "")))
            if (
                existing.get("identity_hash") == identity
                and path.is_file()
                and sha256_file(path) == existing.get("sha256")
                and int(existing.get("rows", -1)) == int(config["split"]["p0_train_rows"])
            ):
                continue
        pending.append(task)

    if pending:
        modules = _canonical_cold_modules(config)
        arena = modules["arena"]
        _seed_everything(int(config["initialization"]["seed"]))
        tokenizer = arena.load_tokenizer(str(Path(base_model_path).resolve()))
        base_config = yaml.safe_load(
            _canonical_paths(config)["base_config"].read_text(encoding="utf-8")
        )
        if not isinstance(base_config, Mapping):
            raise TypeError("Canonical base config is not a mapping")
        model = arena.load_model(
            str(Path(base_model_path).resolve()),
            adapter_path=None,
            trainable_adapter=True,
            load_in_4bit=bool(base_config["model"].get("load_in_4bit", False)),
            dtype=str(base_config["model"].get("dtype", "auto")),
            gradient_checkpointing=False,
            parameterization="lora",
        )
        model.eval()
        original_clean_expression = arena.clean_expression
        try:
            arena.clean_expression = lambda value: str(value)
            for task in pending:
                record = splits["tasks"][task]
                train_rows = read_jsonl(Path(record["paths"]["train"]))
                adapter, instances = _load_task_adapter_and_instances(
                    task,
                    inputs=inputs[task],
                    validation_rows=train_rows,
                )
                derived_rows: list[dict[str, Any]] = []
                audit_rows: list[dict[str, Any]] = []
                runtime = config["task_runtime"][task]
                for source_row in train_rows:
                    prompt_id = str(source_row["prompt_id"])
                    candidates = _verified_wrong_candidates(
                        adapter,
                        instances[prompt_id],
                        source_row,
                    )
                    scores = _score_reference_candidates(
                        arena=arena,
                        model=model,
                        tokenizer=tokenizer,
                        prompt=str(source_row["prompt"]),
                        candidates=candidates,
                        max_length=int(runtime["max_length"]),
                        batch_size=int(runtime["evaluation_batch_size"]),
                    )
                    scored = []
                    for candidate, score in zip(candidates, scores, strict=True):
                        scored.append({**candidate, "reference_surprisal": float(score)})
                    scored.sort(
                        key=lambda item: (
                            float(item["reference_surprisal"]),
                            stable_hash(
                                {
                                    "task": task,
                                    "prompt_id": prompt_id,
                                    "canonical_completion": item["canonical_completion"],
                                }
                            ),
                        )
                    )
                    selected_indices = _evenly_spaced_rank_indices(len(scored), 16)
                    selected: list[dict[str, Any]] = []
                    for slot, rank in enumerate(selected_indices):
                        item = dict(scored[rank])
                        item.update(
                            {
                                "negative_id": f"{prompt_id}_refrem_{slot:03d}",
                                "reference_rank": int(rank),
                                "reference_candidate_count": len(scored),
                                "reference_rank_role": "provenance_and_diagnostic_only",
                            }
                        )
                        selected.append(item)
                    derived = dict(source_row)
                    derived["negatives"] = selected
                    derived["reference_remoteness_selection"] = {
                        "identity_hash": identities[task],
                        "reference_policy": "zero_update_base_plus_fresh_lora",
                        "coordinate": "mean_completion_token_surprisal",
                        "candidate_count": len(scored),
                        "selected_ranks": list(selected_indices),
                        "training_weight_uses_reference_rank": False,
                        "current_policy_surprisal_recomputed_each_update": True,
                    }
                    derived_rows.append(derived)
                    audit_rows.append(
                        {
                            "task": task,
                            "prompt_id": prompt_id,
                            "candidate_count": len(scored),
                            "selected_ranks": list(selected_indices),
                            "candidate_reference_surprisal": _reference_surprisal_summary(
                                [float(item["reference_surprisal"]) for item in scored]
                            ),
                            "selected_reference_surprisal": _reference_surprisal_summary(
                                [float(item["reference_surprisal"]) for item in selected]
                            ),
                            "coverage_threshold": None,
                            "coverage_gate": False,
                        }
                    )
                root = output_root / "reference_remoteness" / task
                bank_path_value = root / "train.jsonl"
                audit_path = root / "prompt_audit.jsonl"
                atomic_jsonl(bank_path_value, derived_rows)
                atomic_jsonl(audit_path, audit_rows)
                ranges = [
                    float(row["selected_reference_surprisal"]["range"])
                    for row in audit_rows
                ]
                summary = {
                    "schema_version": 1,
                    "experiment_id": experiment_id(config),
                    "config_hash": stable_config_hash(config),
                    "task": task,
                    "identity_hash": identities[task],
                    "source_train": record["paths"]["train"],
                    "source_train_sha256": sha256_file(Path(record["paths"]["train"])),
                    "source_p0_bank_preserved": True,
                    "path": str(bank_path_value.resolve()),
                    "sha256": sha256_file(bank_path_value),
                    "rows": len(derived_rows),
                    "selected_negatives_per_prompt": 16,
                    "selection": "evenly_spaced_reference_rank_including_extremes",
                    "candidate_pool": "all_deterministic_verified_wrong_mutations",
                    "reference_rank_enters_training_weight": False,
                    "current_policy_surprisal_recomputed_each_update": True,
                    "coverage_threshold": None,
                    "selected_range_median": float(np.median(np.asarray(ranges, dtype=float))),
                    "prompt_audit": str(audit_path.resolve()),
                    "prompt_audit_sha256": sha256_file(audit_path),
                    "complete": len(derived_rows) == int(config["split"]["p0_train_rows"]),
                    "scientific_status": "not_run",
                }
                if not summary["complete"]:
                    raise RuntimeError(f"Reference-remoteness bank is incomplete for {task}")
                atomic_json(root / "summary.json", summary)
                record["reference_remoteness_bank"] = summary
                atomic_json(output_root / "split_manifest.json", splits)
        finally:
            arena.clean_expression = original_clean_expression
            del model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    manifest = write_canonical_cold_inputs(config, output_root, splits)
    if not all(
        bool(manifest["tasks"][str(task)].get("reference_remoteness_bank_applied"))
        for task in config["suite"]["p0_tasks"]
    ):
        raise RuntimeError("Canonical transfer inputs did not bind every derived reference bank")
    return manifest
'''
if text.count(insert_marker) != 1:
    raise SystemExit("reference helper insertion marker mismatch")
text = text.replace(insert_marker, helpers + insert_marker, 1)

# 7) Canonical inputs consume the derived transfer bank but preserve original P0 bank identity.
old_canonical_branch = '''        if task == "countdown":
            train_path = Path(str(record["bank"])).resolve()
            validation_path = Path(str(record["countdown_validation_source"])).resolve()
            train_rows = read_jsonl(train_path)
            validation_rows = read_jsonl(validation_path)
            exact_countdown_sources = True
        else:
            train_rows = [
                _canonical_train_row(row) for row in read_jsonl(Path(source_paths["train"]))
            ]
            validation_rows = [
                _canonical_validation_row(row)
                for row in read_jsonl(Path(source_paths["validation"]))
            ]
            train_path = task_root / "train.jsonl"
            validation_path = task_root / "validation.jsonl"
            atomic_jsonl(train_path, train_rows)
            atomic_jsonl(validation_path, validation_rows)
            exact_countdown_sources = False
'''
new_canonical_branch = '''        if task == "countdown":
            train_path = Path(str(record["bank"])).resolve()
            validation_path = Path(str(record["countdown_validation_source"])).resolve()
            train_rows = read_jsonl(train_path)
            validation_rows = read_jsonl(validation_path)
            exact_countdown_sources = True
            reference_selection_applied = False
            reference_selection_identity = None
        else:
            reference_record = record.get("reference_remoteness_bank")
            if isinstance(reference_record, Mapping):
                reference_path = Path(str(reference_record.get("path", "")))
                if (
                    not reference_path.is_file()
                    or sha256_file(reference_path) != reference_record.get("sha256")
                    or not reference_record.get("complete")
                ):
                    raise RuntimeError(f"Derived reference-remoteness bank identity failed for {task}")
                train_source = reference_path
                reference_selection_applied = True
                reference_selection_identity = str(reference_record["identity_hash"])
            else:
                # Initial prepare deliberately preserves P0 semantics. Formal calibration
                # replaces this with the derived training-only bank before any cell can run.
                train_source = Path(source_paths["train"])
                reference_selection_applied = False
                reference_selection_identity = None
            train_rows = [
                _canonical_train_row(row) for row in read_jsonl(train_source)
            ]
            validation_rows = [
                _canonical_validation_row(row)
                for row in read_jsonl(Path(source_paths["validation"]))
            ]
            train_path = task_root / "train.jsonl"
            validation_path = task_root / "validation.jsonl"
            atomic_jsonl(train_path, train_rows)
            atomic_jsonl(validation_path, validation_rows)
            exact_countdown_sources = False
'''
text = replace_once(text, old_canonical_branch, new_canonical_branch, "canonical derived bank")
text = replace_once(
    text,
    '            "countdown_exact_source_files": exact_countdown_sources,\n            "negative_consumer": "all_unique_negatives_per_prompt",\n',
    '            "countdown_exact_source_files": exact_countdown_sources,\n'
    '            "reference_remoteness_bank_applied": reference_selection_applied,\n'
    '            "reference_remoteness_bank_identity_hash": reference_selection_identity,\n'
    '            "negative_consumer": "all_unique_negatives_per_prompt",\n',
    "canonical provenance fields",
)

# Bind calibration/cell identity to the derived training bank.
text = replace_once(
    text,
    '        "canonical_train_sha256": record["train_sha256"],\n        "canonical_base_config_sha256": sha256_file(Path(str(record["base_config"]))),\n',
    '        "canonical_train_sha256": record["train_sha256"],\n'
    '        "reference_remoteness_bank_identity_hash": record.get(\n'
    '            "reference_remoteness_bank_identity_hash"\n'
    '        ),\n'
    '        "canonical_base_config_sha256": sha256_file(Path(str(record["base_config"]))),\n',
    "calibration derived identity",
)

# Formal calibration derives the transfer bank before the no-calibration identity gate.
old_calibrate_head = '''def cmd_calibrate(
    config: Mapping[str, Any],
    output_root: Path,
    *,
    base_model_path: str,
    tasks: Sequence[str] | None,
    force: bool,
) -> dict[str, Any]:
    splits, inputs = _load_ready_inputs(
'''
new_calibrate_head = '''def cmd_calibrate(
    config: Mapping[str, Any],
    output_root: Path,
    *,
    base_model_path: str,
    tasks: Sequence[str] | None,
    force: bool,
) -> dict[str, Any]:
    if _is_coldstart(config) and not _is_engineering_self_test(config):
        _derive_reference_remoteness_banks(
            config,
            output_root,
            base_model_path=base_model_path,
        )
    splits, inputs = _load_ready_inputs(
'''
text = replace_once(text, old_calibrate_head, new_calibrate_head, "calibrate derivation")
old_cal_task = '''    if task not in config["suite"]["tasks"]:
        raise ValueError(f"Unknown calibration task: {task}")
    splits, _ = _load_ready_inputs(
'''
new_cal_task = '''    if task not in config["suite"]["tasks"]:
        raise ValueError(f"Unknown calibration task: {task}")
    if not _is_engineering_self_test(config):
        _derive_reference_remoteness_banks(
            config,
            output_root,
            base_model_path=base_model_path,
        )
    splits, _ = _load_ready_inputs(
'''
text = replace_once(text, old_cal_task, new_cal_task, "calibrate-task derivation")

# 8) Countdown keeps the exact worker; transfer tasks use the locked trainer at arbitrary task-local c.
old_grid_for_cell = '''def _paper_grid_for_cell(record: Mapping[str, Any], cell: Cell) -> Path:
    coefficient = 0.0 if cell.lambda_value is None else float(cell.lambda_value)
    return Path(str(record[_paper_grid_name(coefficient)]))
'''
new_grid_for_cell = '''def _paper_grid_for_cell(record: Mapping[str, Any], cell: Cell) -> Path:
    if cell.task != "countdown":
        # Transfer c values are passed directly to the locked trainer. The round-1
        # grid supplies only the frozen training/runtime profile.
        return Path(str(record["round1_grid"]))
    coefficient = 0.0 if cell.lambda_value is None else float(cell.lambda_value)
    return Path(str(record[_paper_grid_name(coefficient)]))
'''
text = replace_once(text, old_grid_for_cell, new_grid_for_cell, "paper grid dispatch")
text = replace_once(
    text,
    '            "canonical_dispatch": "countdown_e8_alpha1_highc_scan_runtime.worker",\n',
    '            "canonical_dispatch": (\n'
    '                "countdown_e8_alpha1_highc_scan_runtime.worker"\n'
    '                if cell.task == "countdown"\n'
    '                else "countdown_e8_alpha1_c_scan_trainer.train_cell"\n'
    '            ),\n',
    "cell dispatch identity",
)
old_task_interface = '''    @contextmanager
    def task_interface() -> Any:
        original_evaluate_rows = arena.evaluate_rows
        original_clean_expression = arena.clean_expression
        try:
            if evaluator is not None:
                arena.evaluate_rows = evaluator
                # Non-arithmetic task outputs are already canonicalized by the P0
                # verifier.  Arithmetic-only line/equality stripping would corrupt
                # Sudoku, graph, and SQL outputs.
                arena.clean_expression = lambda value: str(value)
            yield
        finally:
            arena.evaluate_rows = original_evaluate_rows
            arena.clean_expression = original_clean_expression

    alpha = 0.0 if cell.method == METHOD_POSITIVE_ONLY else 1.0
    coefficient = 0.0 if cell.method in {METHOD_POSITIVE_ONLY, METHOD_GLOBAL} else float(
        cell.lambda_value
    )
    with task_interface():
        returncode = runtime.worker(
            argparse.Namespace(
                family="exponential",
                alpha=alpha,
                c=coefficient,
                seed_offset=int(cell.seed),
                output_dir=str(canonical_output),
                model_path=str(Path(base_model_path).resolve()),
                bank=str(bank),
                val=str(validation),
                base_config=str(base_config_path),
                grid_config=str(grid_path),
            )
        )
    if int(returncode) != 0:
        raise RuntimeError(f"Paper runtime failed for {cell.key}")
'''
new_task_interface = '''    scan_trainer = modules["scan_trainer"]
    paper_common = modules["paper_common"]

    @contextmanager
    def task_interface() -> Any:
        original_evaluate_rows = arena.evaluate_rows
        original_clean_expression = arena.clean_expression
        original_trainer_evaluate = scan_trainer._evaluate_validation
        try:
            if evaluator is not None:
                arena.evaluate_rows = evaluator
                # Non-arithmetic task outputs are already canonicalized by the P0
                # verifier. Arithmetic-only cleanup would corrupt structured outputs.
                arena.clean_expression = lambda value: str(value)

                def transfer_evaluate(**kwargs: Any) -> dict[str, Any]:
                    kwargs["pass64_enabled"] = False
                    return original_trainer_evaluate(**kwargs)

                scan_trainer._evaluate_validation = transfer_evaluate
            yield
        finally:
            arena.evaluate_rows = original_evaluate_rows
            arena.clean_expression = original_clean_expression
            scan_trainer._evaluate_validation = original_trainer_evaluate

    alpha = 0.0 if cell.method == METHOD_POSITIVE_ONLY else 1.0
    coefficient = 0.0 if cell.method in {METHOD_POSITIVE_ONLY, METHOD_GLOBAL} else float(
        cell.lambda_value
    )
    with task_interface():
        if cell.task == "countdown":
            returncode = runtime.worker(
                argparse.Namespace(
                    family="exponential",
                    alpha=alpha,
                    c=coefficient,
                    seed_offset=int(cell.seed),
                    output_dir=str(canonical_output),
                    model_path=str(Path(base_model_path).resolve()),
                    bank=str(bank),
                    val=str(validation),
                    base_config=str(base_config_path),
                    grid_config=str(grid_path),
                )
            )
            if int(returncode) != 0:
                raise RuntimeError(f"Paper runtime failed for {cell.key}")
        else:
            paper_cell = paper_common.Cell(
                alpha=alpha,
                coefficient=coefficient,
                seed_offset=int(cell.seed),
                family="exponential",
            )
            scan_trainer.train_cell(
                cell=paper_cell,
                model_path=Path(base_model_path).resolve(),
                bank=bank,
                val=validation,
                base_config_path=base_config_path,
                grid_config_path=grid_path,
                output_dir=canonical_output,
                repo=_repo_root(),
                smoke=False,
            )
'''
text = replace_once(text, old_task_interface, new_task_interface, "locked transfer dispatch")
text = replace_once(
    text,
    '        "task_interface_adapter_only": cell.task != "countdown",\n        "adapter_path_argument": None,\n',
    '        "task_interface_adapter_only": cell.task != "countdown",\n'
    '        "reference_remoteness_bank_identity_hash": record.get(\n'
    '            "reference_remoteness_bank_identity_hash"\n'
    '        ),\n'
    '        "static_reference_rank_enters_training_weight": False,\n'
    '        "current_policy_surprisal_recomputed_each_update": True,\n'
    '        "adapter_path_argument": None,\n',
    "cell reference-bank provenance",
)

# 9) Result-independent Countdown protocol diagnostic; no stochastic performance gate.
scheduler_gate_start = 'def _scheduler_countdown_reproduction_gate(\n'
scheduler_gate_end = '\n\ndef cmd_run_dynamic(\n'
new_protocol_diag = '''def _countdown_protocol_diagnostic(
    config: Mapping[str, Any],
    output_root: Path,
    *,
    destination: Path | None = None,
) -> dict[str, Any]:
    """Audit Countdown implementation identity without gating any scientific outcome."""

    countdown_cells = [cell for cell in build_cells(config) if cell.task == "countdown"]
    if _is_engineering_self_test(config):
        diagnostic = {
            "schema_version": 1,
            "experiment_id": experiment_id(config),
            "status": "NOT_RUN_ENGINEERING",
            "countdown_cells": len(countdown_cells),
            "result_gate": False,
            "controls_task_transfer_release": False,
            "scientific_evidence": False,
        }
    else:
        expected_sources = dict(config["canonical_coldstart"]["expected_git_blob_shas"])
        identity_failures: list[str] = []
        for cell in countdown_cells:
            path = output_root / "cells" / cell.key / "cell_manifest.json"
            if not path.is_file():
                identity_failures.append(f"{cell.key}:missing")
                continue
            value = json.loads(path.read_text(encoding="utf-8"))
            if (
                value.get("complete") is not True
                or value.get("evaluation_status") != "complete"
                or value.get("nan_inf_failure") is not False
                or value.get("countdown_protocol_exact") is not True
                or value.get("canonical_dispatch")
                != "countdown_e8_alpha1_highc_scan_runtime.worker"
                or value.get("canonical_source_git_blob_shas") != expected_sources
                or int(value.get("terminal_step", -1)) != 1200
                or value.get("stop_reason") != "max_steps"
            ):
                identity_failures.append(f"{cell.key}:protocol_identity")
        diagnostic = {
            "schema_version": 1,
            "experiment_id": experiment_id(config),
            "status": "PASS" if not identity_failures and len(countdown_cells) == 16 else "FAIL",
            "countdown_cells": len(countdown_cells),
            "identity_failures": identity_failures,
            "result_gate": False,
            "controls_task_transfer_release": False,
            "scientific_evidence": True,
        }
    if destination is not None:
        atomic_json(destination, diagnostic)
    return diagnostic
'''
text = replace_region(text, scheduler_gate_start, scheduler_gate_end, new_protocol_diag, "protocol diagnostic")

# 10) Hard 16-cell wave barriers, two slots/GPU, recovery-aware but result-independent.
run_dynamic_start = 'def cmd_run_dynamic(\n'
run_dynamic_end = '\n\ndef cmd_run_all(\n'
new_run_dynamic = '''def cmd_run_dynamic(
    config: Mapping[str, Any],
    config_path: Path,
    output_root: Path,
    *,
    base_model_path: str,
    force: bool,
    retry_incomplete: bool,
) -> dict[str, Any]:
    """Run recovery-aware 16-cell waves; every wave is a scheduling barrier only."""

    if not _is_coldstart(config):
        raise RuntimeError("Dynamic scheduling is frozen for the cold-start profile only")
    _require_calibration_gate(config, output_root, base_model_path=base_model_path)
    _require_liveness_gate(config, output_root, base_model_path=base_model_path)
    cells = build_cells(config)
    waves = build_waves(config)
    gpu_ids = tuple(int(value) for value in config["execution"]["gpu_ids"])
    slots_per_gpu = int(config["execution"]["slots_per_gpu"])
    slot_count = len(gpu_ids) * slots_per_gpu
    if slot_count != int(config["execution"]["max_concurrent_cells"]) or slot_count != 16:
        raise RuntimeError("Declared 16-slot capacity is internally inconsistent")

    lock = threading.Lock()
    checkpoint_lock = threading.Lock()
    results: list[dict[str, Any]] = []
    event_path = output_root / "scheduler" / "queue_events.jsonl"
    event_path.parent.mkdir(parents=True, exist_ok=True)
    scheduler_run_id = f"queue-{int(time.time())}-{os.getpid()}"
    recovery_package_value = os.environ.get("E8_COLDSTART_RECOVERY_PACKAGE", "").strip()
    recovery_package = Path(recovery_package_value).resolve() if recovery_package_value else None
    recovery_interval = int(os.environ.get("E8_COLDSTART_RECOVERY_INTERVAL_CELLS", "5"))
    if recovery_interval <= 0:
        raise ValueError("E8_COLDSTART_RECOVERY_INTERVAL_CELLS must be positive")
    initially_reusable, _ = _reusable_cell_manifests(config, output_root)
    last_checkpoint_count = (len(initially_reusable) // recovery_interval) * recovery_interval

    def record(event: Mapping[str, Any]) -> None:
        with lock:
            append_jsonl(event_path, {"scheduler_run_id": scheduler_run_id, **dict(event)})

    def run_one(cell: Cell, *, wave_index: int, slot: int) -> dict[str, Any]:
        nonlocal last_checkpoint_count
        gpu_id = gpu_ids[slot % len(gpu_ids)]
        cell_root = output_root / "cells" / cell.key
        manifest_path = cell_root / "cell_manifest.json"
        reusable_complete = False
        if manifest_path.is_file():
            try:
                reusable_complete = bool(
                    json.loads(manifest_path.read_text(encoding="utf-8")).get("complete")
                )
            except (OSError, json.JSONDecodeError):
                reusable_complete = False
        child_force = force or (retry_incomplete and cell_root.exists() and not reusable_complete)
        record(
            {
                "event": "start",
                "wave": wave_index,
                "cell_key": cell.key,
                "slot": slot,
                "gpu_id": gpu_id,
                "unix_time": time.time(),
                "retry_incomplete": child_force and not force,
            }
        )
        result = _run_subprocess_cell(
            config_path=config_path.resolve(),
            output_root=output_root.resolve(),
            base_model_path=base_model_path,
            cell=cell,
            gpu_id=gpu_id,
            force=child_force,
        )
        if int(result["returncode"]) == 0 and recovery_package is not None:
            try:
                with checkpoint_lock:
                    current_reusable, _ = _reusable_cell_manifests(config, output_root)
                    completed_count = len(current_reusable)
                    if completed_count >= last_checkpoint_count + recovery_interval:
                        checkpoint = _publish_recovery_checkpoint(
                            config,
                            output_root,
                            package_output=recovery_package,
                        )
                        last_checkpoint_count = int(checkpoint["completed_cells"])
                        result["recovery_checkpoint"] = checkpoint["package"]
                        result["recovery_checkpoint_completed_cells"] = last_checkpoint_count
            except Exception as exc:
                result["returncode"] = 74
                result["recovery_checkpoint_error"] = f"{type(exc).__name__}: {exc}"
        result.update({"wave": wave_index, "slot": slot})
        record({"event": "finish", **result, "unix_time": time.time()})
        return result

    wave_records: list[dict[str, Any]] = []
    stopped = False
    for wave_index, wave in enumerate(waves, start=1):
        wave_results: list[dict[str, Any]] = []
        with ThreadPoolExecutor(max_workers=len(wave)) as executor:
            futures = [
                executor.submit(run_one, cell, wave_index=wave_index, slot=slot)
                for slot, cell in enumerate(wave)
            ]
            for future in as_completed(futures):
                wave_results.append(future.result())
        wave_results.sort(key=lambda row: str(row["cell_key"]))
        results.extend(wave_results)
        failures = [row for row in wave_results if int(row["returncode"]) != 0]
        wave_records.append(
            {
                "wave": wave_index,
                "expected_cells": len(wave),
                "returned_cells": len(wave_results),
                "failed_cells": [row["cell_key"] for row in failures],
                "complete": not failures and len(wave_results) == len(wave),
                "scheduling_barrier": True,
                "scientific_result_gate": False,
            }
        )
        if failures or len(wave_results) != len(wave):
            stopped = True
            break

    results.sort(key=lambda row: str(row["cell_key"]))
    failures = [row for row in results if int(row["returncode"]) != 0]
    returned_keys = {str(row["cell_key"]) for row in results}
    completed_keys = {str(row["cell_key"]) for row in results if int(row["returncode"]) == 0}
    unscheduled = [cell.key for cell in cells if cell.key not in returned_keys]
    protocol_diagnostic = _countdown_protocol_diagnostic(
        config,
        output_root,
        destination=output_root / "scheduler" / "countdown_protocol_diagnostic.json",
    ) if not stopped and not unscheduled else {
        "status": "PENDING",
        "result_gate": False,
        "controls_task_transfer_release": False,
    }
    manifest = {
        "schema_version": 1,
        "experiment_id": experiment_id(config),
        "scheduler": "dynamic_slot_queue",
        "scheduler_run_id": scheduler_run_id,
        "wave_barriers": True,
        "wave_count": len(waves),
        "slot_count": slot_count,
        "gpu_ids": list(gpu_ids),
        "slots_per_gpu": slots_per_gpu,
        "waves": wave_records,
        "countdown_protocol_diagnostic": protocol_diagnostic,
        "countdown_result_controls_transfer_release": False,
        "expected_cells": len(cells),
        "completed_cells": len(completed_keys),
        "results": results,
        "failed_cells": [row["cell_key"] for row in failures],
        "unscheduled_cells": unscheduled,
        "queue_events": str(event_path.resolve()),
        "complete": not failures and not unscheduled and len(completed_keys) == len(cells),
        "scientific_status": "not_run" if _is_engineering_self_test(config) else "pilot",
        "engineering_placeholder_backend": _is_engineering_self_test(config),
    }
    atomic_json(output_root / "scheduler" / "dynamic_run.json", manifest)
    if failures or unscheduled:
        raise RuntimeError(
            "Cold-start scheduling stopped fail-closed; "
            f"failed={manifest['failed_cells']} unscheduled={len(unscheduled)}"
        )
    return manifest
'''
text = replace_region(text, run_dynamic_start, run_dynamic_end, new_run_dynamic, "wave scheduler")
text = replace_once(
    text,
    '        raise RuntimeError("Cold-start has no wave barriers; use run-all dynamic scheduling")\n',
    '        raise RuntimeError("Cold-start waves are owned by run-all to preserve recovery barriers")\n',
    "run-wave message",
)

# 11) Cold aggregate: no .95 eligibility, no numeric Countdown gate, correct 208 geometry.
aggregate_start = 'def _aggregate_coldstart(\n'
aggregate_end = '\n\ndef cmd_aggregate(\n'
new_aggregate = '''def _aggregate_coldstart(
    config: Mapping[str, Any],
    output_root: Path,
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    provenance_path = output_root / "source_provenance.json"
    provenance = (
        json.loads(provenance_path.read_text(encoding="utf-8")) if provenance_path.is_file() else {}
    )
    run_id = str(provenance.get("run_id", output_root.name))
    source_commit = str(provenance.get("source_commit", "unrecorded"))
    plot_rows: list[dict[str, Any]] = []
    for row in rows:
        plot_rows.append(
            {
                "experiment_id": experiment_id(config),
                "run_id": run_id,
                "source_commit": source_commit,
                "task": row["task"],
                "method": row["method"],
                "lambda": row["lambda"],
                "rho": row["rho"],
                "seed": row["seed"],
                "stage": row["stage"],
                "late_window_pass8_mean": row["late_window_pass8_mean"],
                "late_window_greedy_mean": row["late_window_greedy_mean"],
                "best_validation_pass8": row["best_pass8"],
                "terminal_pass8": row["terminal_pass8"],
                "best_validation_greedy": row["best_greedy"],
                "terminal_greedy": row["terminal_greedy"],
                "best_greedy_valid_rate": row["best_greedy_valid_rate"],
                "terminal_greedy_valid_rate": row["terminal_greedy_valid_rate"],
                "best_step": row["best_step"],
                "terminal_step": row["terminal_step"],
                "stop_reason": row["stop_reason"],
                "nan_inf_failure": row["nan_inf_failure"],
                "complete": True,
            }
        )
    _write_csv(output_root / "aggregate" / "plot_curve_points.csv", plot_rows)

    summaries: dict[str, Any] = {}
    summary_rows: list[dict[str, Any]] = []
    for task_value in config["suite"]["tasks"]:
        task = str(task_value)
        task_rows = [row for row in rows if row["task"] == task]
        positive_rows = [row for row in task_rows if row["method"] == METHOD_POSITIVE_ONLY]
        global_rows = [row for row in task_rows if row["method"] == METHOD_GLOBAL]
        exp_rows = [row for row in task_rows if row["method"] == METHOD_EXPONENTIAL]
        expected_counts = (2, 2, 12) if task == "countdown" else (4, 0, 20)
        if (len(positive_rows), len(global_rows), len(exp_rows)) != expected_counts:
            raise RuntimeError(f"{task} cold-start cell counts differ from {expected_counts}")

        def aggregate_group(group: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
            first = group[0]
            return {
                "task": task,
                "method": first["method"],
                "lambda": first["lambda"],
                "rho": first["rho"],
                "seeds": sorted(int(row["seed"]) for row in group),
                "late_window_pass8_mean": float(
                    np.mean([float(row["late_window_pass8_mean"]) for row in group])
                ),
                "late_window_greedy_mean": float(
                    np.mean([float(row["late_window_greedy_mean"]) for row in group])
                ),
                "terminal_pass8_mean": float(
                    np.mean([float(row["terminal_pass8"]) for row in group])
                ),
                "terminal_greedy_valid_rate_mean": float(
                    np.mean([float(row["terminal_greedy_valid_rate"]) for row in group])
                ),
                "best_pass8_mean": float(np.mean([float(row["best_pass8"]) for row in group])),
                "nan_inf_failure": any(bool(row["nan_inf_failure"]) for row in group),
            }

        coefficient_groups: dict[tuple[str, float | None], list[dict[str, Any]]] = {}
        for row in task_rows:
            coefficient_groups.setdefault((str(row["method"]), row["lambda"]), []).append(row)
        grouped = [aggregate_group(group) for group in coefficient_groups.values()]
        positive = next(row for row in grouped if row["method"] == METHOD_POSITIVE_ONLY)
        grouped_exp = [row for row in grouped if row["method"] == METHOD_EXPONENTIAL]
        selectable = [row for row in grouped_exp if not row["nan_inf_failure"]]
        selected = (
            max(
                selectable,
                key=lambda row: (
                    float(row["late_window_pass8_mean"]),
                    float(row["terminal_pass8_mean"]),
                    float(row["late_window_greedy_mean"]),
                    -float(row["lambda"]),
                ),
            )
            if selectable
            else None
        )
        min_lambda = min(float(row["lambda"]) for row in grouped_exp)
        max_lambda = max(float(row["lambda"]) for row in grouped_exp)
        selected_on_edge = bool(
            selected is not None
            and (
                math.isclose(float(selected["lambda"]), min_lambda)
                or math.isclose(float(selected["lambda"]), max_lambda)
            )
        )
        task_summary = {
            "task": task,
            "positive_only": positive,
            "global": next((row for row in grouped if row["method"] == METHOD_GLOBAL), None),
            "selectable_exp_count": len(selectable),
            "selected_exp": selected,
            "selected_on_grid_edge": selected_on_edge,
            "terminal_valid_rate_role": "diagnostic_only_not_selection_eligibility",
            "all_exp_below_positive_only": all(
                float(row["late_window_pass8_mean"]) < float(positive["late_window_pass8_mean"])
                for row in grouped_exp
            ),
            "grouped_curve": sorted(
                grouped,
                key=lambda row: (
                    row["method"] != METHOD_POSITIVE_ONLY,
                    -1.0 if row["lambda"] is None else float(row["lambda"]),
                ),
            ),
        }
        summaries[task] = task_summary
        summary_rows.append(
            {
                "task": task,
                "positive_only_late_window_pass8_mean": positive["late_window_pass8_mean"],
                "selected_lambda": None if selected is None else selected["lambda"],
                "selected_rho": None if selected is None else selected["rho"],
                "selected_late_window_pass8_mean": (
                    None if selected is None else selected["late_window_pass8_mean"]
                ),
                "selected_on_grid_edge": selected_on_edge,
                "all_exp_below_positive_only": task_summary["all_exp_below_positive_only"],
            }
        )
    _write_csv(output_root / "aggregate" / "task_summary.csv", summary_rows)

    protocol_diagnostic = _countdown_protocol_diagnostic(
        config,
        output_root,
        destination=output_root / "aggregate" / "countdown_protocol_diagnostic.json",
    )
    summary = {
        "schema_version": 1,
        "experiment_id": experiment_id(config),
        "run_id": run_id,
        "source_commit": source_commit,
        "cell_count": len(rows),
        "plot_curve_point_count": len(plot_rows),
        "tasks": summaries,
        "excluded_tasks": dict(config["suite"]["excluded_tasks"]),
        "initialization": dict(config["initialization"]),
        "positive_only_and_exp_share_fresh_initialization": True,
        "scientific_kernel": "canonical_old_coldstart_imports",
        "canonical_source_git_blob_shas": dict(
            config["canonical_coldstart"]["expected_git_blob_shas"]
        ),
        "countdown_protocol_diagnostic": protocol_diagnostic,
        "countdown_result_gate": False,
        "primary_metric": "validation_late_window_pass8_mean",
        "terminal_valid_rate_role": "diagnostic_only_not_selection_eligibility",
        "test_partition_accessed": False,
        "transfer_exp_single_seed_response_shape_localization": True,
        "transfer_positive_only_seed_count": 4,
        "fresh_seed_confirmation_required_for_winner_claim": True,
        "method_ranking_allowed": False,
        "significance_claim_allowed": False,
        "fixed_horizon_is_convergence": False,
        "task_performance_reported_separately": True,
        "structure_diagnostic_reported_separately": True,
        "nan_inf_reported_separately": True,
        "scientific_status": "not_run" if _is_engineering_self_test(config) else "pilot",
        "engineering_placeholder_backend": _is_engineering_self_test(config),
    }
    atomic_json(output_root / "aggregate" / "aggregate_summary.json", summary)
    return summary
'''
text = replace_region(text, aggregate_start, aggregate_end, new_aggregate, "cold aggregate")

# 12) Terminal audit uses protocol identity only, never Countdown stochastic outcome.
old_audit_cold = '''    elif _is_coldstart(config):
        aggregate_path = output_root / "aggregate" / "aggregate_summary.json"
        reproduction_path = output_root / "aggregate" / "countdown_reproduction_gate.json"
        if reproduction_path.is_file():
            reproduction_gate_status = str(
                json.loads(reproduction_path.read_text(encoding="utf-8")).get("status")
            )
        aggregate_complete = aggregate_path.is_file() and int(
            json.loads(aggregate_path.read_text(encoding="utf-8")).get("cell_count", 0)
        ) == len(cells) and reproduction_gate_status == (
            "NOT_RUN_ENGINEERING" if _is_engineering_self_test(config) else "PASS"
        )
'''
new_audit_cold = '''    elif _is_coldstart(config):
        aggregate_path = output_root / "aggregate" / "aggregate_summary.json"
        protocol_path = output_root / "aggregate" / "countdown_protocol_diagnostic.json"
        if protocol_path.is_file():
            reproduction_gate_status = str(
                json.loads(protocol_path.read_text(encoding="utf-8")).get("status")
            )
        aggregate_complete = aggregate_path.is_file() and int(
            json.loads(aggregate_path.read_text(encoding="utf-8")).get("cell_count", 0)
        ) == len(cells) and reproduction_gate_status == (
            "NOT_RUN_ENGINEERING" if _is_engineering_self_test(config) else "PASS"
        )
'''
text = replace_once(text, old_audit_cold, new_audit_cold, "audit cold protocol")
text = replace_once(
    text,
    '        "countdown_reproduction_gate_status": reproduction_gate_status,\n',
    '        "countdown_protocol_diagnostic_status": reproduction_gate_status,\n'
    '        "countdown_result_gate": False if _is_coldstart(config) else None,\n'
    '        "transfer_exp_single_seed_response_shape_localization": _is_coldstart(config),\n',
    "audit diagnostic fields",
)

# 13) Engineering self-test audits the real hard wave barriers and 2 slots/GPU.
queue_audit_start = 'def _audit_engineering_queue(\n'
queue_audit_end = '\n\ndef cmd_engineering_self_test(\n'
new_queue_audit = '''def _audit_engineering_queue(
    config: Mapping[str, Any],
    output_root: Path,
    scheduler: Mapping[str, Any],
) -> dict[str, Any]:
    cells = build_cells(config)
    events = [
        json.loads(line)
        for line in (output_root / "scheduler" / "queue_events.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]
    events = [row for row in events if row["scheduler_run_id"] == scheduler["scheduler_run_id"]]
    active_by_gpu = {int(gpu): 0 for gpu in config["execution"]["gpu_ids"]}
    maximum_by_gpu = dict(active_by_gpu)
    starts: dict[str, float] = {}
    finishes: dict[str, float] = {}
    waves_by_cell: dict[str, int] = {}
    for event in events:
        gpu_id = int(event["gpu_id"])
        cell_key = str(event["cell_key"])
        if event["event"] == "start":
            active_by_gpu[gpu_id] += 1
            maximum_by_gpu[gpu_id] = max(maximum_by_gpu[gpu_id], active_by_gpu[gpu_id])
            starts[cell_key] = float(event["unix_time"])
            waves_by_cell[cell_key] = int(event["wave"])
        elif event["event"] == "finish":
            active_by_gpu[gpu_id] -= 1
            finishes[cell_key] = float(event["unix_time"])
    expected_keys = {cell.key for cell in cells}
    if set(starts) != expected_keys or set(finishes) != expected_keys:
        raise RuntimeError(f"Engineering queue did not observe all {len(cells)} starts/finishes")
    slots_per_gpu = int(config["execution"]["slots_per_gpu"])
    if any(value != 0 for value in active_by_gpu.values()) or max(maximum_by_gpu.values()) > slots_per_gpu:
        raise RuntimeError("Engineering queue exceeded the declared per-GPU capacity")
    waves = build_waves(config)
    for wave_index in range(1, len(waves)):
        current_keys = {cell.key for cell in waves[wave_index - 1]}
        next_keys = {cell.key for cell in waves[wave_index]}
        if min(starts[key] for key in next_keys) < max(finishes[key] for key in current_keys):
            raise RuntimeError(f"Wave {wave_index + 1} started before wave {wave_index} finished")
    if set(waves_by_cell.values()) != set(range(1, len(waves) + 1)):
        raise RuntimeError("Engineering queue did not record the exact wave identities")
    return {
        "all_cells_observed": True,
        "maximum_active_by_gpu": maximum_by_gpu,
        "wave_barriers_respected": True,
        "wave_count": len(waves),
        "slots_per_gpu": slots_per_gpu,
    }
'''
text = replace_region(text, queue_audit_start, queue_audit_end, new_queue_audit, "engineering queue audit")

source_path.write_text(text, encoding="utf-8")

# Replace the reviewed configuration atomically from the prepared temp payload.
config_path = Path("configs/e8_multitask_exp_coldstart.yaml")
config_text = Path(".drpo_tmp/e8_multitask_exp_coldstart.yaml.new").read_text(encoding="utf-8")
config_text = replace_once(
    config_text,
    '  positive_only_entry: countdown_e8_alpha1_highc_scan_runtime.worker\n'
    '  exponential_entry: countdown_e8_alpha1_highc_scan_runtime.worker_or_locked_train_cell_for_transfer_grid\n',
    '  countdown_entry: countdown_e8_alpha1_highc_scan_runtime.worker\n'
    '  transfer_entry: countdown_e8_alpha1_c_scan_trainer.train_cell\n',
    "canonical config entries",
)
config_path.write_text(config_text, encoding="utf-8")

# Replace only the cold-start test tail; all earlier P0/dense tests remain byte-for-byte.
test_path = Path("tests/test_e8_multitask_p0.py")
tests = test_path.read_text(encoding="utf-8")
marker = '\ndef test_exp_coldstart_matrix_contains_exact_countdown_and_task_transfer_cells() -> None:\n'
if tests.count(marker) != 1:
    raise SystemExit("cold-start test-tail marker mismatch")
prefix = tests.split(marker, 1)[0]
new_tests = r'''

def test_exp_coldstart_matrix_is_208_cells_in_13_hard_waves() -> None:
    from drpo import e8_multitask_exp_tuning as exp_tuning

    config = exp_tuning.load_config(Path("configs/e8_multitask_exp_coldstart.yaml"))
    cells = exp_tuning.build_cells(config)
    waves = exp_tuning.build_waves(config)
    assert len(cells) == 208
    assert len({cell.key for cell in cells}) == 208
    assert len(waves) == 13
    assert [len(wave) for wave in waves] == [16] * 13
    assert config["execution"]["max_concurrent_cells"] == 16
    assert config["execution"]["slots_per_gpu"] == 2
    assert config["execution"]["wave_barriers"] is True
    assert set(config["suite"]["tasks"]) == set(exp_tuning.TASK_NAMES)
    assert set(config["suite"]["p0_tasks"]) == set(exp_tuning.TASK_NAMES) - {"countdown"}

    countdown = [cell for cell in cells if cell.task == "countdown"]
    assert len(countdown) == 16
    assert {cell.seed for cell in countdown} == {4000, 5000}
    assert sum(cell.method == exp_tuning.METHOD_POSITIVE_ONLY for cell in countdown) == 2
    assert sum(cell.method == exp_tuning.METHOD_GLOBAL for cell in countdown) == 2
    assert sum(cell.method == exp_tuning.METHOD_EXPONENTIAL for cell in countdown) == 12
    assert {float(cell.lambda_value) for cell in countdown if cell.method == exp_tuning.METHOD_EXPONENTIAL} == set(
        config["sweep"]["countdown_sentinel_coefficients"]
    )

    for task in config["suite"]["p0_tasks"]:
        task_cells = [cell for cell in cells if cell.task == task]
        assert len(task_cells) == 24
        positives = [cell for cell in task_cells if cell.method == exp_tuning.METHOD_POSITIVE_ONLY]
        exp_cells = [cell for cell in task_cells if cell.method == exp_tuning.METHOD_EXPONENTIAL]
        assert {cell.seed for cell in positives} == {4000, 5000, 6000, 7000}
        assert len(exp_cells) == 20
        assert {cell.seed for cell in exp_cells} == {4000}
        assert exp_tuning.stable_hash(list(exp_tuning._task_lambdas(config, task))) == config["sweep"]["task_grid_hashes"][task]
        assert config["sweep"]["task_grid_provenance"][task]


def test_exp_coldstart_has_no_stochastic_result_gate_or_valid_rate_eligibility() -> None:
    from drpo import e8_multitask_exp_tuning as exp_tuning

    config = exp_tuning.load_config(Path("configs/e8_multitask_exp_coldstart.yaml"))
    serialized = json.dumps(config, sort_keys=True)
    assert "countdown_reproduction_gate" not in serialized
    assert "require_peak_above_positive_only" not in serialized
    assert "late_window_pass8_absolute_tolerance" not in serialized
    assert "terminal_valid_rate_minimum" not in config["selection"]
    assert config["selection"]["terminal_valid_rate_role"] == (
        "diagnostic_only_not_selection_eligibility"
    )
    assert config["reporting"]["countdown_role"] == (
        "diagnostic_regression_sentinel_not_result_gate"
    )


def test_exp_coldstart_reference_remoteness_contract_is_static_selection_dynamic_weighting() -> None:
    from drpo import e8_multitask_exp_tuning as exp_tuning

    config = exp_tuning.load_config(Path("configs/e8_multitask_exp_coldstart.yaml"))
    contract = config["negative_sampling"]["reference_remoteness_bank"]
    assert contract["source_candidates"] == "all_deterministic_verified_wrong_mutations"
    assert contract["selected_negatives_per_prompt"] == 16
    assert contract["coverage_threshold"] is None
    assert contract["reference_rank_role"] == "provenance_and_diagnostic_only"
    assert contract["static_reference_rank_enters_training_weight"] is False
    assert contract["current_policy_surprisal_recomputed_each_update"] is True
    assert contract["original_p0_bank_preserved"] is True
    assert exp_tuning._evenly_spaced_rank_indices(16) == tuple(range(16))
    indices = exp_tuning._evenly_spaced_rank_indices(41)
    assert len(indices) == 16
    assert len(set(indices)) == 16
    assert indices[0] == 0 and indices[-1] == 40


def test_verified_wrong_candidate_reconstruction_uses_full_deterministic_universe() -> None:
    from drpo import e8_multitask_exp_tuning as exp_tuning

    class FakeResult:
        def __init__(self, value: str) -> None:
            self.score = 0.0
            self.correct = False
            self.format_valid = True
            self.error_class = "wrong"
            self.canonical_completion = value
            self.details = {"fake": True}

    class FakeAdapter:
        def mutation_candidates(self, instance, rng):
            del instance
            order = list(range(25))
            rng.shuffle(order)
            for value in order:
                yield SimpleNamespace(completion=f"wrong-{value}", mutation_class="wrong")
            yield SimpleNamespace(completion="wrong-0", mutation_class="wrong")

        def verify(self, instance, completion, mutation_class=None):
            del instance, mutation_class
            return FakeResult(completion)

        def accept_negative(self, result):
            return result.format_valid and not result.correct

    instance = TaskInstance("fake", "p0", "prompt", "oracle", {}, {})
    source_row = {
        "task": "fake",
        "prompt_id": "p0",
        "generation_seed": 17,
        "negatives": [
            {"completion": f"wrong-{value}", "canonical_completion": f"wrong-{value}"}
            for value in range(16)
        ],
    }
    first = exp_tuning._verified_wrong_candidates(FakeAdapter(), instance, source_row)
    second = exp_tuning._verified_wrong_candidates(FakeAdapter(), instance, source_row)
    assert first == second
    assert len(first) == 25
    assert len({row["canonical_completion"] for row in first}) == 25


def test_exp_coldstart_rejects_adapter_runtime_or_grid_drift() -> None:
    from drpo import e8_multitask_exp_tuning as exp_tuning

    config = exp_tuning.load_config(Path("configs/e8_multitask_exp_coldstart.yaml"))
    changed = json.loads(json.dumps(config))
    changed["initialization"]["external_adapter_allowed"] = True
    with pytest.raises(ValueError, match="zero-update"):
        exp_tuning.validate_config(changed)
    changed = json.loads(json.dumps(config))
    changed["execution"]["slots_per_gpu"] = 1
    with pytest.raises(ValueError, match="two slots"):
        exp_tuning.validate_config(changed)
    changed = json.loads(json.dumps(config))
    changed["task_runtime"]["word_sorting"]["evaluation_batch_size"] = 8
    with pytest.raises(ValueError, match="word_sorting"):
        exp_tuning.validate_config(changed)
    changed = json.loads(json.dumps(config))
    changed["sweep"]["task_lambda"]["maze"][0] = 0.11
    with pytest.raises(ValueError, match="maze"):
        exp_tuning.validate_config(changed)


def test_exp_coldstart_imports_locked_kernel_and_forbids_multitask_loader() -> None:
    from drpo import e8_multitask_exp_tuning as exp_tuning

    config = exp_tuning.load_config(Path("configs/e8_multitask_exp_coldstart.yaml"))
    assert config["canonical_coldstart"]["countdown_entry"] == (
        "countdown_e8_alpha1_highc_scan_runtime.worker"
    )
    assert config["canonical_coldstart"]["transfer_entry"] == (
        "countdown_e8_alpha1_c_scan_trainer.train_cell"
    )
    audit = exp_tuning.audit_canonical_coldstart_sources(config)
    assert audit["verified"]
    assert audit["git_blob_shas"] == config["canonical_coldstart"]["expected_git_blob_shas"]
    with pytest.raises(RuntimeError, match="old canonical"):
        exp_tuning._load_reference_model("base-model", None, config, train_mode=True)


def test_task_base_config_transfer_has_batch16_and_pass8_only(tmp_path: Path) -> None:
    from drpo import e8_multitask_exp_tuning as exp_tuning

    config = exp_tuning.load_config(Path("configs/e8_multitask_exp_coldstart.yaml"))
    task_root = tmp_path / "graph_color"
    task_root.mkdir()
    path, changed = exp_tuning._task_base_config(
        config,
        task="graph_color",
        canonical_paths=exp_tuning._canonical_paths(config),
        task_root=task_root,
    )
    assert changed == [
        "evaluation.batch_size",
        "evaluation.pass_ks",
        "model.max_length",
        "model.max_new_tokens",
    ]
    value = yaml.safe_load(path.read_text())
    assert value["model"]["max_length"] == 512
    assert value["model"]["max_new_tokens"] == 128
    assert value["evaluation"]["batch_size"] == 16
    assert value["evaluation"]["pass_ks"] == [8]


def test_exp_coldstart_scheduler_enforces_wave_barriers_and_two_slots_per_gpu(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from drpo import e8_multitask_exp_tuning as exp_tuning

    config = exp_tuning.load_config(Path("configs/e8_multitask_exp_coldstart.yaml"))
    cells = exp_tuning.build_cells(config)
    waves = exp_tuning.build_waves(config)
    monkeypatch.setattr(exp_tuning, "_require_calibration_gate", lambda *args, **kwargs: None)
    monkeypatch.setattr(exp_tuning, "_require_liveness_gate", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        exp_tuning,
        "_countdown_protocol_diagnostic",
        lambda *args, **kwargs: {
            "status": "FAIL",
            "result_gate": False,
            "controls_task_transfer_release": False,
        },
    )
    lock = threading.Lock()
    starts: dict[str, float] = {}
    finishes: dict[str, float] = {}
    active: dict[int, int] = {}
    maximum: dict[int, int] = {}

    def fake_run(**kwargs: object) -> dict[str, object]:
        cell = kwargs["cell"]
        gpu_id = int(kwargs["gpu_id"])
        with lock:
            starts[cell.key] = time.monotonic()
            active[gpu_id] = active.get(gpu_id, 0) + 1
            maximum[gpu_id] = max(maximum.get(gpu_id, 0), active[gpu_id])
        time.sleep(0.002)
        with lock:
            active[gpu_id] -= 1
            finishes[cell.key] = time.monotonic()
        return {
            "cell_key": cell.key,
            "gpu_id": gpu_id,
            "returncode": 0,
            "log": "mock.log",
            "started_unix": 0.0,
            "finished_unix": 1.0,
        }

    monkeypatch.setattr(exp_tuning, "_run_subprocess_cell", fake_run)
    result = exp_tuning.cmd_run_dynamic(
        config,
        Path("configs/e8_multitask_exp_coldstart.yaml"),
        tmp_path,
        base_model_path="base-model",
        force=False,
        retry_incomplete=True,
    )
    assert result["complete"]
    assert result["completed_cells"] == 208
    assert result["wave_barriers"] is True
    assert len(result["waves"]) == 13
    assert result["countdown_protocol_diagnostic"]["status"] == "FAIL"
    assert result["countdown_result_controls_transfer_release"] is False
    assert set(maximum) == set(range(8))
    assert max(maximum.values()) <= 2
    for index in range(1, len(waves)):
        assert min(starts[cell.key] for cell in waves[index]) >= max(
            finishes[cell.key] for cell in waves[index - 1]
        )
    assert {cell.task for cell in cells[16:32]} != {"countdown"}


def test_exp_coldstart_aggregate_does_not_filter_by_terminal_valid_rate(tmp_path: Path) -> None:
    from drpo import e8_multitask_exp_tuning as exp_tuning

    config = exp_tuning._engineering_self_test_config(
        exp_tuning.load_config(Path("configs/e8_multitask_exp_coldstart.yaml"))
    )
    p0.atomic_json(
        tmp_path / "source_provenance.json",
        {"run_id": "cold-test", "source_commit": "a" * 40},
    )
    target_task = "word_sorting"
    target_lambda = config["sweep"]["task_lambda"][target_task][0]
    for cell in exp_tuning.build_cells(config):
        late = 0.10 if cell.method == exp_tuning.METHOD_POSITIVE_ONLY else 0.20
        valid = 1.0
        if (
            cell.task == target_task
            and cell.method == exp_tuning.METHOD_EXPONENTIAL
            and float(cell.lambda_value) == float(target_lambda)
        ):
            late = 0.90
            valid = 0.10
        p0.atomic_json(
            tmp_path / "cells" / cell.key / "cell_manifest.json",
            {
                "complete": True,
                "evaluation_status": "complete",
                "validation_best_pass8": late,
                "validation_late_window_pass8_mean": late,
                "validation_terminal_pass8": late,
                "validation_best_greedy": late / 2,
                "validation_late_window_greedy_mean": late / 2,
                "validation_terminal_greedy": late / 2,
                "validation_best_greedy_valid_rate": valid,
                "validation_terminal_greedy_valid_rate": valid,
                "best_step": 900,
                "terminal_step": 1200,
                "stop_reason": "max_steps",
                "nan_inf_failure": False,
            },
        )
    summary = exp_tuning.cmd_aggregate(config, tmp_path)
    selected = summary["tasks"][target_task]["selected_exp"]
    assert selected["lambda"] == pytest.approx(float(target_lambda))
    assert selected["late_window_pass8_mean"] == pytest.approx(0.90)
    assert selected["terminal_greedy_valid_rate_mean"] == pytest.approx(0.10)
    assert summary["terminal_valid_rate_role"] == "diagnostic_only_not_selection_eligibility"
    assert summary["countdown_result_gate"] is False


def test_coldstart_engineering_self_test_exercises_208_cell_recovery_and_barriers(
    tmp_path: Path,
) -> None:
    from drpo import e8_multitask_exp_tuning as exp_tuning

    config = exp_tuning.load_config(Path("configs/e8_multitask_exp_coldstart.yaml"))
    source_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    result = exp_tuning.cmd_engineering_self_test(
        config,
        tmp_path / "engineering-self-test",
        source_commit=source_commit,
    )
    assert result["complete"]
    assert result["scientific_status"] == "not_run"
    assert result["resume_completed_cells"] == 208
    assert result["aggregate_cell_count"] == 208
    assert result["queue_audit"]["wave_barriers_respected"]
    assert result["queue_audit"]["wave_count"] == 13
    assert result["queue_audit"]["slots_per_gpu"] == 2
    assert result["repeat_run_preserved_cell_hashes"]
    assert result["tampered_package_rejected"]


def test_coldstart_runbook_embeds_bootstrap_and_current_protocol() -> None:
    runbook = Path("docs/experiments/EXT-C-E8-MULTITASK-EXP-COLDSTART-01_RUNBOOK.md").read_text(
        encoding="utf-8"
    )
    bootstrap = Path("scripts/bootstrap_e8_multitask_exp_coldstart.sh").read_text(
        encoding="utf-8"
    )
    embedded = runbook.split("<!-- ONE_CLICK_BOOTSTRAP_START -->", 1)[1].split(
        "<!-- ONE_CLICK_BOOTSTRAP_END -->", 1
    )[0]
    assert embedded == f"\n```bash\n{bootstrap.rstrip()}\n```\n"
    assert "208 cells" in runbook
    assert "13" in runbook and "16-cell" in runbook
    assert "Spiral Matrix" in runbook
    assert "Pass@64" in runbook
    assert "结果门禁" in runbook
    assert "reference-remoteness" in runbook
    assert "terminal valid rate" in runbook.lower()
    assert "0.002" not in runbook
    assert "峰值必须高于" not in runbook
    assert "run_experiment_guard_hardened.py" in Path(
        "scripts/run_e8_multitask_exp_coldstart.sh"
    ).read_text(encoding="utf-8")
'''
test_path.write_text(prefix + new_tests, encoding="utf-8")

# Runbook: replace only Section 0; preserve the one-click bootstrap block byte-for-byte.
runbook_path = Path("docs/experiments/EXT-C-E8-MULTITASK-EXP-COLDSTART-01_RUNBOOK.md")
runbook = runbook_path.read_text(encoding="utf-8")
section0_start = runbook.index("## 0. 交付目标与不可变边界")
section1_start = runbook.index("## 1. 唯一需要执行的代码块")
new_section0 = '''## 0. 交付目标与不可变边界

这份文件是服务器本地 AI 的唯一操作入口。执行者不需要预先知道 DRPO 在哪里，不需要
手动切分支，不需要判断应该从训练、聚合还是打包恢复。完整执行第 1 节唯一代码块后，
脚本会自动发现仓库、创建隔离 worktree、运行门禁并在同一 runtime 内自动恢复。

唯一仓库：<https://github.com/easonhuo/drpo>

审核实现：<https://github.com/easonhuo/drpo/pull/309>

实验 ID：`EXT-C-E8-MULTITASK-EXP-COLDSTART-01`

当前科学/执行边界（代码会 fail closed，不接受现场覆盖）：

- 无 SFT warm start、无外部/reference adapter；所有 cell 从同一 Qwen2.5-0.5B base 的
  zero-update LoRA 初始化语义出发；
- 9 个任务全部进入本轮：Countdown + Word Sorting、Spiral Matrix、Mini Sudoku、Maze、
  Word Ladder、Knights & Knaves、Graph Coloring、WikiSQL；
- Countdown 只承担回归/外部有效性 sentinel：2 个历史 seed ×（Positive-only、Global、
  6 个论文网格 Exp c）= 16 cells。它的代码、数据、seed 和终态身份会审计，但任何
  stochastic performance 数值都**不是结果门禁**，不会决定后续任务是否启动；
- 其余 8 个任务各 24 cells：4 个 Positive-only 历史 seed + 单一 tuning seed 上 20 个
  task-local Exp c。总计 **208 cells**。这只能支持 response-shape/localization；若以后要做
  最优 c 的正式多 seed 排名，必须另行登记确认实验；
- 非 Countdown 训练不改原 P0 bank。prepare 后、训练前，从现有 task adapter/verifier
  **重建每个 prompt 的全部 deterministic verified-wrong mutations**，用 zero-update
  reference policy 的 mean completion-token surprisal 排序，再均匀取 16 个 rank（含两端）
  形成 derived `reference-remoteness` training bank。reference rank/surprisal 只做 provenance
  和诊断，不参与训练权重，也没有 coverage threshold；训练中的 Exp taper 仍在每次 update
  按当前 policy surprisal 重新计算；
- Countdown 精确保留 256/80、evaluation batch 8、Greedy 500、Pass@8 500，并保留
  Pass@64 作为 Countdown 原协议的辅助诊断；8 个 transfer 任务使用 512/128、evaluation
  batch 16、Greedy 500、Pass@8 128，**不运行隐藏 Pass@64**；
- 固定 1200 optimizer updates、LR/optimizer/warmup/max-grad-norm、LoRA r/alpha/dropout、
  数据 split、task prompt/verifier、采样 temperature/top-p 与论文 current-surprisal Exp
  公式均不变；
- 调度容量固定为 8 GPU × 2 cells/GPU = **16-cell wave**。208 cells 正好 13 个 wave；
  wave N 的 16 个 cell 全部结束后才允许 wave N+1 开始。这个 barrier 只是 scheduling /
  recovery 边界，不是科学结果门禁；OOM 时 cell 失败并保留证据，禁止自动改 batch、λ、
  loss、数据或其他科学参数；
- 选择 Exp c 时 primary 为 late-window Pass@8 mean；tie 依次使用 terminal Pass@8、
  late-window greedy、较小 c。terminal valid rate 只报告/审计，**不作为 selection
  eligibility**；
- task performance、valid/structure 诊断事件与 NaN/Inf 数值崩溃分开报告。工程 self-test、
  liveness、恢复验收和有限步运行都不能冒充正式科学结果或方法排名。

正式模式默认运行 `full`。PR/stack 未合并、实验未登记、execution gate 不是 `ready`，
或没有唯一 READY RunSpec 绑定执行时的精确 `main` SHA 时，入口会在创建训练环境、下载
模型或接触 GPU 前停止。不得绕过。

'''
runbook_path.write_text(runbook[:section0_start] + new_section0 + runbook[section1_start:], encoding="utf-8")

# Liveness anchor must remain one of the six Countdown sentinel c values.
launcher_path = Path("scripts/run_e8_multitask_exp_coldstart.sh")
launcher = launcher_path.read_text(encoding="utf-8")
launcher = replace_once(
    launcher,
    '    --lambda 0.693147181 \\\n',
    '    --lambda 0.916290732 \\\n',
    "liveness sentinel",
)
launcher_path.write_text(launcher, encoding="utf-8")
PY

# Temporary authoring files are never part of the resulting implementation commit.
rm -rf .drpo_tmp
rm -f .github/workflows/e8-coldstart-repair-temp.yml
