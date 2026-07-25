            }
            for threshold in geometry_module.WEIGHT_THRESHOLDS:
                label = str(threshold).replace(".", "p")
                payload[f"negative_weight_gt_{label}_fraction"] = (
                    self.threshold_counts[threshold] / count if count else None
                )
            payload.update(
                geometry_module._quantile_payload("negative_distance", self.distance_samples)
            )
            payload.update(
                geometry_module._quantile_payload("negative_weight", self.factor_samples)
            )
            for key, value in payload.items():
                if isinstance(value, float) and not math.isfinite(value):
                    raise FloatingPointError(f"non-finite geometry diagnostic field: {key}")
            _append_jsonl(self.jsonl_path, payload)
            _atomic_json(self.latest_path, payload)
            self._reset_interval()

    bootstrap.GeometryDiagnostics = NearFarGeometryDiagnostics

    original_patch = bootstrap.patch_canonical_module
    audit = TrainingAudit(branch_dir)

    def patched_module(*args: Any, **kwargs: Any) -> Any:
        result = original_patch(*args, **kwargs)
        module = args[0]
        contract = args[1]
        audit.install(module, contract.target_class)
        return result

    bootstrap.patch_canonical_module = patched_module
    result = int(bootstrap.main(argv))
    training_audit = audit.validate_complete()

    manifest_path = branch_dir / "branch_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    snapshot = manifest.get("trajectory_snapshot", {})
    shift = snapshot.get("advantage_shift")
    if not isinstance(shift, dict) or float(shift.get("fixed_baseline", float("nan"))) != baseline:
        raise RuntimeError("completed manifest is missing the exact advantage baseline")
    if not snapshots_path.is_file():
        raise RuntimeError("completed branch is missing snapshot-series diagnostics")
    public_control = manifest.get("weight_control", {})
    if public_control.get("method") != method:
        raise RuntimeError("completed manifest method mismatch")
    control = {
        "experiment_id": EXPERIMENT_ID,
        "profile_id": PROFILE_ID,
        "control_id": values.get("control_id"),
        "weight_method": method,
        "fixed_baseline": baseline,
        "baseline_units": "raw_trajectory_snapshot_gae",
        "shift_before_sign_split": True,
        "replay_sample_deletion": False,
        "replay_sample_masking": False,
        "remoteness_scale": None if method == "positive_only" else scale,
        "derived_exp_coefficient": (
            0.0 if method == "positive_only" else float(values["exp_coefficient"])
        ),
        "reference_distance": REFERENCE_DISTANCE,
        "remoteness_threshold": float(values["remoteness_threshold"]),
        "taper_lambda": float(values["taper_lambda"]),
        "trajectory_snapshot_batch_size": EXPECTED_BATCH,
        "snapshot_advantage_shift": shift,
        "actor_training_dynamics": training_audit,
    }
    _atomic_json(branch_dir / "POSBASE_DRPO_CONTROL.json", control)
    _atomic_json(branch_dir / "OPTIMIZER_CONTROL.json", training_audit["optimizer_control"])
    manifest["posbase_drpo_control"] = control
    manifest["actor_training_dynamics"] = training_audit
    manifest["trajectory_snapshot_batch_size"] = EXPECTED_BATCH
    _atomic_json(manifest_path, manifest)
    return result


def main() -> int:
    if sys.argv[1:] == ["--self-test"]:
        return _self_test()
    return _run_bootstrap(sys.argv[1:])


raise SystemExit(main())