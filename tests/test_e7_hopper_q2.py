from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import h5py
import numpy as np
import torch

MODULE_PATH = Path(__file__).parents[1] / "src" / "drpo" / "e7_hopper_q2.py"
spec = importlib.util.spec_from_file_location("e7_hopper_q2", MODULE_PATH)
assert spec and spec.loader
mod = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = mod
spec.loader.exec_module(mod)


def _rebuttal_toy():
    from dataclasses import replace

    settings, base = mod.load_rebuttal_config(Path(__file__).parents[1] / "configs/e7_hopper_pq_is_rebuttal_v1.yaml", True)
    base = replace(base, hidden_sizes=(4,))
    settings = {**settings, "diagnostic_batch_size": 8, "variance_batch_size": 8}
    torch.manual_seed(23)
    rng = np.random.default_rng(61)
    n = 32
    data = mod.OfflineData(
        rng.normal(size=(n, 3)).astype(np.float32), np.tanh(rng.normal(size=(n, 2))).astype(np.float32),
        rng.normal(size=n).astype(np.float32), rng.normal(size=(n, 3)).astype(np.float32),
        np.zeros(n, dtype=bool), np.zeros(n, dtype=bool), np.arange(n),
        np.full(n, -2.0, dtype=np.float32), np.arange(n),
    )
    critic = mod.ValueNetwork(3, (4,))
    actor = mod._new_rebuttal_policy(base, 3, 2)
    norm = mod.Normalizer(np.array([30.], np.float32), np.array([7.], np.float32))
    context = {"base": base, "settings": settings, "data": data, "obs": data.observations,
               "next_obs": data.next_observations, "target_norm": norm, "center": .3, "scale": 2.,
               "diagnostic_ids": np.arange(n), "diagnostic_ids_sha256": "toy", "train_ids": np.arange(n)}
    a, delta = mod._rebuttal_advantage_arrays(critic, context, np.arange(n), torch.device("cpu"))
    context.update(initial_advantage=a, initial_delta=delta)
    return settings, base, actor, critic, context


def _topr_config(tmp_path):
    import yaml

    root = Path(__file__).parents[1]
    raw = yaml.safe_load((root / "configs/e7_hopper_pq_is_rebuttal_v1.yaml").read_text())
    raw.update(experiment_id=mod.TOPR_EXPERIMENT_ID, base_config=str(root / "configs/e7_hopper_q2_medium_replay_v2.yaml"),
               seeds=[200, 201], actor_steps=500000, comparison_reference="drpo",
               methods_by_advantage_mode={"frozen": ["drpo", "topr"], "refreshed": []},
               default_task="hopper-medium-replay-v2", variance_steps=[])
    raw["q2"] = {**raw["q2"], "enabled": False, "pairs": []}
    raw["smoke"].update(seeds=[200, 201], variance_steps=[], critic_steps=2, critic_eval_interval=1,
                        actor_steps=4, late_steps=[3, 4], max_transitions=None)
    raw["tasks"] = {
        "hopper-medium-replay-v2": {"dataset_basename": "hopper.hdf5", "dataset_sha256": "a" * 64,
            "dataset_id": "hopper-medium-replay-v2", "env_id": "Hopper-v4", "reference_min_score": -20., "reference_max_score": 3234.3},
        "halfcheetah-medium-v2": {"dataset_basename": "cheetah.hdf5", "dataset_sha256": "b" * 64,
            "dataset_id": "halfcheetah-medium-v2", "env_id": "HalfCheetah-v4", "reference_min_score": -280., "reference_max_score": 12135.},
    }
    path = tmp_path / "topr.yaml"
    path.write_text(yaml.safe_dump(raw))
    return path


def test_topr_negative_cap_positive_invariance_detach_and_large_log_ratios():
    settings, *_ = _rebuttal_toy()
    log_ratio = torch.tensor([np.log(.1), 0., np.log(10.), np.log(.1), 0., np.log(10.), 1000., -1000.],
                             dtype=torch.float64, requires_grad=True)
    behavior = torch.zeros_like(log_ratio, requires_grad=True)
    a = torch.tensor([-2., -2., -2., 3., 3., 3., -2., -2.], dtype=torch.float64, requires_grad=True)
    distance = torch.ones_like(a, requires_grad=True)
    coefficient = mod.rebuttal_actor_coefficients(a, log_ratio, distance, behavior, "topr", settings)
    expected = torch.tensor([-.2, -2., -2., 3., 3., 3., -2., 0.], dtype=torch.float64)
    assert not coefficient.requires_grad and torch.isfinite(coefficient).all()
    torch.testing.assert_close(coefficient, expected)
    loss = mod.rebuttal_actor_loss(log_ratio, coefficient)
    torch.testing.assert_close(loss, -(expected * log_ratio).sum() / len(a))
    loss.backward()
    torch.testing.assert_close(log_ratio.grad, -expected / len(a))
    assert a.grad is None and behavior.grad is None and distance.grad is None


def test_topr_five_fixed_betas_preserve_canonical_and_positive_coefficients():
    settings, *_ = _rebuttal_toy()
    assert list(mod.TOPR_METHOD_BETAS.values()) == [.25, .5, 1., 2., 4.]
    lp = torch.tensor([np.log(.1), 0., np.log(10.), np.log(.1), 1000., -1000.], dtype=torch.float64, requires_grad=True)
    a = torch.tensor([-2., -2., -2., 3., -2., -2.], dtype=torch.float64, requires_grad=True)
    behavior = torch.zeros_like(lp, requires_grad=True)
    canonical = a.detach().clamp_min(0) - torch.exp(torch.minimum(lp.detach(), torch.zeros_like(lp))) * (-a.detach()).clamp_min(0)
    for method, beta in mod.TOPR_METHOD_BETAS.items():
        coefficient = mod.rebuttal_actor_coefficients(a, lp, torch.ones_like(lp), behavior, method, settings)
        assert not coefficient.requires_grad and torch.isfinite(coefficient).all()
        torch.testing.assert_close(coefficient, torch.tensor([-2 * .1 ** beta, -2., -2., 3., -2., 0.], dtype=torch.float64))
        gradient, = torch.autograd.grad(mod.rebuttal_actor_loss(lp, coefficient), lp, retain_graph=True)
        torch.testing.assert_close(gradient, -coefficient / len(lp))
        if beta == 1:
            assert method == "topr" and torch.equal(coefficient, canonical)


def test_topr_profile_task_identity_jobs_and_legacy_matrix_are_separate(tmp_path):
    import yaml

    old, base = mod.load_rebuttal_config(Path(__file__).parents[1] / "configs/e7_hopper_pq_is_rebuttal_v1.yaml")
    assert old["identity"] == "bd6e5b673dd7a33fdf8ae782bbe1eaa241a60b8fbac9682084273865eb8edebe"
    assert base.weight_decay == .0001 and "topr" not in mod.REBUTTAL_METHODS
    assert len(mod.rebuttal_matrix(old["seeds"])) == 36
    old_jobs = mod.rebuttal_jobs(old)
    assert len(old_jobs) == 30 and sum(j["job_kind"] == "q2_pair" for j in old_jobs) == 6
    config = _topr_config(tmp_path)
    default, _ = mod.load_rebuttal_config(config)
    chosen, chosen_base = mod.load_rebuttal_config(config, task="halfcheetah-medium-v2")
    assert default["task_id"] == "hopper-medium-replay-v2"
    assert chosen["identity"] != default["identity"]
    assert chosen["actual_task"]["dataset_sha256"] == chosen_base.dataset_sha256 == "b" * 64
    assert chosen_base.dataset_basename == "cheetah.hdf5" and chosen_base.env_id == "HalfCheetah-v4"
    assert chosen_base.rollout_dataset_id == "halfcheetah-medium-v2"
    assert chosen_base.normalized_score_reference_min == -280. and chosen_base.normalized_score_reference_max == 12135.
    jobs = mod.rebuttal_jobs(chosen)
    assert len(jobs) == 4 and all(j["job_kind"] == "independent" for j in jobs)
    assert {(j["seed"], j["method"]) for j in jobs} == {(s, m) for s in (200, 201) for m in ("drpo", "topr")}
    raw = yaml.safe_load(config.read_text())
    raw["tasks"]["halfcheetah-medium-v2"]["worker_threads"] = 4
    config.write_text(yaml.safe_dump(raw))
    threaded, _ = mod.load_rebuttal_config(config, task="halfcheetah-medium-v2")
    assert threaded["worker_threads"] == 4 and threaded["identity"] != chosen["identity"]
    assert default["worker_threads"] == chosen["worker_threads"] == 1


def test_topr_fixed_population_weight_diagnostics_and_no_negative_case():
    _, _, actor, critic, context = _rebuttal_toy()
    context["initial_advantage"] = np.tile(np.array([-1., 1.]), 16)
    lp = mod.current_action_log_prob(actor, torch.from_numpy(context["obs"]), torch.from_numpy(context["data"].actions))[0].detach().numpy()
    desired_log_ratio = np.tile(np.log([.1, 1., 10., 1.]), 8)
    context["data"].behavior_log_prob[:] = lp - desired_log_ratio
    row, _ = mod.collect_rebuttal_diagnostics(actor, critic, context, "topr", "frozen", 0, None, torch.device("cpu"))
    negative = context["initial_advantage"] < 0
    weights = np.exp(np.minimum((lp - context["data"].behavior_log_prob)[negative], 0.))
    np.testing.assert_allclose(row["topr_negative_weight_mean"], weights.mean(), rtol=1e-6)
    for key, quantile in [("median", .5), ("p90", .9), ("p99", .99)]:
        np.testing.assert_allclose(row[f"topr_negative_weight_{key}"], np.quantile(weights, quantile), rtol=1e-6)
    assert row["topr_negative_log_ratio_lt0_fraction"] == .5
    context["initial_advantage"][:] = 1
    empty, _ = mod.collect_rebuttal_diagnostics(actor, critic, context, "topr", "frozen", 1, None, torch.device("cpu"))
    assert all(empty[key] is None for key in row if key.startswith("topr_negative_"))
    assert empty["topr_beta"] == 1 and empty["topr_identity"] == "canonical_topr_weight_adaptation"
    context["initial_advantage"][:] = np.tile(np.array([-1., 1.]), 16)
    for method, beta in mod.TOPR_METHOD_BETAS.items():
        row, _ = mod.collect_rebuttal_diagnostics(actor, critic, context, method, "frozen", 2, None, torch.device("cpu"))
        assert row["topr_beta"] == beta and row["topr_identity"] == mod.TOPR_METHOD_IDENTITIES[method]
        np.testing.assert_allclose(row["topr_negative_weight_mean"], np.mean(weights ** beta), rtol=1e-6)


def test_topr_preparation_without_q2_does_not_calibrate_or_hash_missing_files(tmp_path, monkeypatch):
    import json
    from dataclasses import replace

    settings, base = mod.load_rebuttal_config(_topr_config(tmp_path), True)
    settings.update(actor_batch_size=8, critic_batch_size=8, diagnostic_size=8, diagnostic_batch_size=8)
    base = replace(base, hidden_sizes=(4,), critic_batch_size=8)
    dataset = tmp_path / "data.hdf5"
    rng = np.random.default_rng(11)
    with h5py.File(dataset, "w") as f:
        f["observations"] = rng.normal(size=(80, 3)).astype(np.float32)
        f["next_observations"] = rng.normal(size=(80, 3)).astype(np.float32)
        f["actions"] = np.tanh(rng.normal(size=(80, 2))).astype(np.float32)
        f["rewards"] = rng.normal(size=80).astype(np.float32)
        f["terminals"] = np.arange(80) % 4 == 3
        f["infos/action_log_probs"] = np.full(80, -2., np.float32)

    def forbidden(*args, **kwargs):
        raise AssertionError("Q2 calibration must not execute when disabled")

    monkeypatch.setattr(mod, "calibrate_rebuttal_q2", forbidden)
    prepare = tmp_path / "prepare"
    mod.prepare_rebuttal_context(dataset, prepare, settings, base, torch.device("cpu"), "c" * 40)
    assert not list(prepare.glob("q2_calibration*"))
    record = json.loads((prepare / "PREPARE_COMPLETE.json").read_text())
    assert len(record["prepared_input_hashes"]) == 6
    for name, expected in record["prepared_input_hashes"].items():
        assert mod.sha256_file(prepare / name) == expected


def test_rebuttal_actor_snapshots_preserve_step_models_and_default_output(tmp_path, monkeypatch):
    import json
    import shutil

    from test_e7_q2_acceptance_pipeline import _rebuttal_worker_fixture

    module, args, original = _rebuttal_worker_fixture(tmp_path, monkeypatch, "frozen", "topr")
    settings, _ = module.load_rebuttal_config(args.config, True)
    settings.update(checkpoint_interval=2, variance_steps=[])
    assert not settings.get("retain_actor_snapshots", False)
    assert module.train_rebuttal_branch(args) == 0
    old_cell = module._rebuttal_cell(original, 200, "frozen", "topr")
    assert not (old_cell / "models").exists() and not (old_cell / "actor_snapshot_index.json").exists()
    baseline = torch.load(old_cell / "checkpoint.pt", map_location="cpu", weights_only=False)
    retained = tmp_path / "retained"
    shutil.copytree(original / "prepare", retained / "prepare")
    settings["retain_actor_snapshots"] = True
    args.work_dir = str(retained)
    save, captured = module._atomic_rebuttal_checkpoint, {}

    def capture(path, state):
        save(path, state)
        if path.name == "checkpoint.pt" and state["step"] > 0:
            captured[state["step"]] = {k: v.detach().cpu().clone() for k, v in state["actor"].items()}

    monkeypatch.setattr(module, "_atomic_rebuttal_checkpoint", capture)
    assert module.train_rebuttal_branch(args) == 0
    cell = module._rebuttal_cell(retained, 200, "frozen", "topr")
    index = json.loads((cell / "actor_snapshot_index.json").read_text())
    assert [r["step"] for r in index["snapshots"]] == [2, 4]
    context = module._load_rebuttal_context(retained / "prepare", Path(args.dataset_path), settings, module.load_rebuttal_config(args.config, True)[1])
    for record in index["snapshots"]:
        path = Path(record["path"])
        assert path.name == f"actor_step_{record['step']:09d}.pt"
        assert path.stat().st_size == record["size_bytes"] and module.sha256_file(path) == record["sha256"]
        snapshot = torch.load(path, map_location="cpu", weights_only=True)
        assert snapshot["identity"] == baseline["identity"] and snapshot["step"] == record["step"]
        assert set(snapshot) == {"identity", "step", "actor", "obs_normalizer", "actor_structure"}
        reloaded = module.SquashedGaussianPolicy(**snapshot["actor_structure"])
        reloaded.load_state_dict(snapshot["actor"])
        for key, value in snapshot["actor"].items():
            assert value.device.type == "cpu"
            torch.testing.assert_close(value, captured[record["step"]][key], rtol=0, atol=0)
            torch.testing.assert_close(value, reloaded.state_dict()[key], rtol=0, atol=0)
        np.testing.assert_array_equal(snapshot["obs_normalizer"]["mean"], context["obs_norm"].mean)
        np.testing.assert_array_equal(snapshot["obs_normalizer"]["std"], context["obs_norm"].std)
    for key, value in baseline["actor"].items():
        torch.testing.assert_close(value, captured[4][key], rtol=0, atol=0)
    assert any(not torch.equal(captured[2][key], captured[4][key]) for key in captured[2])
    resumed = tmp_path / "snapshot_resume"
    shutil.copytree(retained / "prepare", resumed / "prepare")
    args.work_dir, args.resume_from = str(resumed), str(retained)
    assert module.train_rebuttal_branch(args) == 0
    exported = module._rebuttal_cell(resumed, 200, "frozen", "topr")
    assert json.loads((exported / "actor_snapshot_index.json").read_text())["snapshots"] == index["snapshots"]
    assert not (exported / "models").exists()  # existing step files stay referenced in the previous root


def test_topr_coordinator_generic_dimensions_slots_task_and_optional_reference(tmp_path, monkeypatch):
    import json

    import yaml

    config = _topr_config(tmp_path)
    dataset = tmp_path / "cheetah.hdf5"
    with h5py.File(dataset, "w") as handle:
        handle["observations"] = np.zeros((8, 17), np.float32)
        handle["actions"] = np.zeros((8, 6), np.float32)
    root = tmp_path / "run"
    args = mod.build_parser().parse_args([
        "rebuttal-pq-is", "--config", str(config), "--task", "halfcheetah-medium-v2",
        "--dataset-path", str(dataset), "--work-dir", str(root), "--smoke",
        "--worker-devices", "cpu,cuda:3,cpu,cuda:3", "--max-workers", "4",
    ])
    observed = {}
    commands = []
    monkeypatch.setattr(mod, "git_output", lambda *args: "c" * 40)
    monkeypatch.setattr(mod, "environment_manifest", dict)
    monkeypatch.setattr(mod, "_plot_rebuttal", lambda *args: None)

    def preflight(**kwargs):
        observed.update(kwargs)
        return {"status": "passed"}

    def prepare(dataset, path, settings, base, device, source_commit):
        assert str(device) == observed.get("expected_preparation_device", "cpu")
        mod._write_rebuttal_json(path / "normalization.json", {})
        mod._write_rebuttal_json(path / "PREPARE_COMPLETE.json", {})

    class FinishedWorker:
        def __init__(self, command, **kwargs):
            worker_args = mod.build_parser().parse_args(command[2:])
            assert worker_args.task == "halfcheetah-medium-v2"
            commands.append(command)
            seed = int(command[command.index("--seed") + 1])
            method = command[command.index("--method") + 1]
            value = float(seed - 200 + (2 if method == "topr" else 0))
            mod._write_rebuttal_json(mod._rebuttal_cell(root, seed, "frozen", method) / "summary.json", {
                "seed": seed, "method": method, "advantage_mode": "frozen", "status": "completed",
                "completed_actor_steps": 4, "late_count": 2, "late_mean": value, "terminal": value,
            })

        def poll(self):
            return 0

    monkeypatch.setattr(mod, "preflight_rollout_environment", preflight)
    monkeypatch.setattr(mod, "prepare_rebuttal_context", prepare)
    monkeypatch.setattr(mod.subprocess, "Popen", FinishedWorker)
    assert mod.run_rebuttal_experiment(args) == 0
    assert observed["expected_observation_dim"] == 17 and observed["expected_action_dim"] == 6
    assert observed["env_id"] == "HalfCheetah-v4"
    assert [c[c.index("--device") + 1] for c in commands] == ["cpu", "cuda:3", "cpu", "cuda:3"]
    assert all(c[c.index("--task") + 1] == "halfcheetah-medium-v2" for c in commands)
    manifest = json.loads((root / "scientific_run_manifest.json").read_text())
    assert manifest["experiment_id"] == mod.TOPR_EXPERIMENT_ID and manifest["q2_calibrations"] == []
    assert manifest["task_id"] == "halfcheetah-medium-v2" and "TOPR negative" in manifest["density_processing"]
    rows = mod._read_rebuttal_csv(root / "summary/METHOD_SUMMARY.csv")
    assert len(rows) == 2 and {r["method"] for r in rows} == {"drpo", "topr"}
    paired = mod._read_rebuttal_csv(root / "summary/PAIRED_COMPARISONS.csv")
    assert len(paired) == 2 and all(r["reference"] == "drpo" and float(r["mean_difference"]) == 2 for r in paired)
    assert len(mod._read_rebuttal_csv(root / "summary/PAIRED_SEED_DIFFERENCES.csv")) == 4
    assert not list((root / "summary").glob("Q2_*"))
    settings, _ = mod.load_rebuttal_config(config, True, "halfcheetah-medium-v2")
    settings["methods_by_advantage_mode"]["frozen"] = ["topr"]
    audit = mod.aggregate_rebuttal(root, settings)
    assert audit["expected_cells"] == 2
    assert mod._read_rebuttal_csv(root / "summary/PAIRED_COMPARISONS.csv") == []
    assert mod._read_rebuttal_csv(root / "summary/PAIRED_SEED_DIFFERENCES.csv") == []
    # The owner-selected single-seed screen composes nine five-job task runs;
    # preparation may use a GPU while all five actor slots use CPUs.
    raw = yaml.safe_load(config.read_text())
    raw.update(seeds=[200], comparison_reference="topr", methods_by_advantage_mode={"frozen": list(mod.TOPR_METHOD_BETAS), "refreshed": []})
    raw["smoke"]["seeds"] = [200]
    families = [("hopper", "Hopper-v4", -20., 3234.3), ("halfcheetah", "HalfCheetah-v4", -280., 12135.),
                ("walker2d", "Walker2d-v4", 1.6, 4592.3)]
    raw["tasks"] = {f"{family}-{quality}-v2": {
        "dataset_basename": f"{family}_{quality}-v2.hdf5", "dataset_sha256": "b" * 64,
        "dataset_id": f"{family}-{quality}-v2", "env_id": env,
        "reference_min_score": low, "reference_max_score": high,
    } for family, env, low, high in families for quality in ("medium", "medium-replay", "medium-expert")}
    config.write_text(yaml.safe_dump(raw))
    all_jobs = []
    for task in raw["tasks"]:
        selected, _ = mod.load_rebuttal_config(config, task=task)
        task_jobs = mod.rebuttal_jobs(selected)
        assert len(task_jobs) == 5 and all(j["seed"] == 200 and j["job_kind"] == "independent" for j in task_jobs)
        all_jobs.extend((task, j["seed"], j["method"]) for j in task_jobs)
    assert len(all_jobs) == len(set(all_jobs)) == 45
    root = tmp_path / "beta_run"
    args.work_dir, args.device = str(root), "cuda:7"
    args.worker_devices, args.max_workers = "cpu,cpu,cpu,cpu,cpu", 5
    observed["expected_preparation_device"] = "cuda:7"
    commands.clear()
    assert mod.run_rebuttal_experiment(args) == 0
    assert len(commands) == 5 and all(c[c.index("--device") + 1] == "cpu" for c in commands)
    manifest = json.loads((root / "scientific_run_manifest.json").read_text())
    assert manifest["preparation_device"] == "cuda:7" and manifest["worker_devices"] == ["cpu"] * 5
    assert {m: rule["beta"] for m, rule in manifest["method_weighting"].items()} == mod.TOPR_METHOD_BETAS
    assert manifest["method_weighting"]["topr"]["identity"] == "canonical_topr_weight_adaptation"
    assert all(rule["identity"] == "beta_topr_logged_frozen_reference_variant" for m, rule in manifest["method_weighting"].items() if m != "topr")
    rows = mod._read_rebuttal_csv(root / "summary/METHOD_SUMMARY.csv")
    assert len(rows) == 5 and all(r["late_mean_n"] == r["terminal_n"] == "1" for r in rows)
    assert all(r["late_mean_seed_sample_sd"] == r["terminal_seed_sample_sd"] == "" for r in rows)
    assert {r["method"]: float(r["topr_beta"]) for r in rows} == mod.TOPR_METHOD_BETAS
    paired = mod._read_rebuttal_csv(root / "summary/PAIRED_COMPARISONS.csv")
    assert len(paired) == 8 and all(r["paired_n"] == "1" and r["difference_sample_sd"] == "" for r in paired)


def test_q2_far_block_cap_and_global_matching_zero_boundaries():
    near, far = torch.tensor([3., 0.], dtype=torch.float64), torch.tensor([0., 4.], dtype=torch.float64)
    factor = mod.q2_cap_factor(far.norm(), 2.)
    torch.testing.assert_close(factor, torch.tensor(.5, dtype=torch.float64))
    torch.testing.assert_close(near + factor * far, torch.tensor([3., 2.], dtype=torch.float64))
    target = (near + factor * far).norm()
    alpha = mod.q2_global_scale(target, target.new_tensor(1.))
    assert alpha > 1  # matching is not clipped to the taper interval
    assert mod.q2_global_scale(target.new_tensor(0.), target).item() == 0
    assert mod.q2_global_scale(target, target.new_tensor(0.)).item() == 0
    assert mod.q2_cap_factor(target.new_tensor(0.), 0.).item() == 1
    assert mod.q2_cap_factor(target, 0.).item() == 0


def test_q2_actual_distinct_actors_match_full_parameter_forces_and_detach_controls():
    _, base, actor, _, context = _rebuttal_toy()
    other = mod._new_rebuttal_policy(base, 3, 2)
    with torch.no_grad():
        other.log_std.fill_(.7)
    obs = torch.from_numpy(context["obs"][:4])
    actions = torch.from_numpy(context["data"].actions[:4])
    lp, _, _ = mod.current_action_log_prob(actor, obs, actions)
    gp, _, _ = mod.current_action_log_prob(other, obs, actions)
    a = torch.tensor([2., -1., -3., 0.], dtype=torch.float64)
    distances = [torch.tensor([8., 1., 3., 9.]), torch.zeros(4)]
    params = list(actor.parameters())

    def force(logp, coef, parameters):
        return torch.cat([g.double().flatten() for g in torch.autograd.grad((logp * coef).sum() / 4, parameters, retain_graph=True)])

    gn = force(lp, torch.tensor([0., -1., 0., 0.]), params)
    gf = force(lp, torch.tensor([0., 0., -3., 0.]), params)
    calibration = {"far_threshold": 2., "far_cap": float(gf.norm()) / 2}
    coefficients, stats = mod.q2_pair_coefficients([actor, other], [lp, gp], distances, a, calibration, "far_cap")
    torch.testing.assert_close(coefficients[0], torch.tensor([2., -1., -1.5, 0.], dtype=torch.float64))
    assert not any(c.requires_grad for c in coefficients)
    assert not any(v.requires_grad for v in stats.values())
    torch.testing.assert_close(stats["target_budget"], (gn + .5 * gf).norm(), rtol=1e-6, atol=1e-7)
    assert float(stats["relative_error_nonzero_target"]) < 1e-6
    graw = force(gp, a.clamp(max=0), list(other.parameters())).norm()
    torch.testing.assert_close(stats["alpha"], stats["target_budget"] / graw)
    assert stats["far_count"] == 1 and stats["near_count"] == 1
    actual = torch.cat([g.double().flatten() for g in torch.autograd.grad(mod.rebuttal_actor_loss(lp, coefficients[0]), params, retain_graph=True)])
    expected = -force(lp, a.clamp_min(0), params) - gn - .5 * gf
    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-7)
    assert actual.numel() == sum(p.numel() for p in params)  # includes learned log_std


def test_q2_masks_use_current_distance_fixed_threshold_and_full_batch_denominator():
    _, base, actor, _, context = _rebuttal_toy()
    other = mod._new_rebuttal_policy(base, 3, 2)
    obs, actions = torch.from_numpy(context["obs"][:4]), torch.from_numpy(context["data"].actions[:4])
    lp = mod.current_action_log_prob(actor, obs, actions)[0]
    gp = mod.current_action_log_prob(other, obs, actions)[0]
    a = torch.tensor([2., -1., -3., 0.], dtype=torch.float64)
    calibration = {"far_threshold": 2., "far_cap": 0.}
    first, _ = mod.q2_pair_coefficients([actor, other], [lp, gp], [torch.tensor([9., 2., 3., 9.]), torch.zeros(4)], a, calibration, "far_zero")
    second, _ = mod.q2_pair_coefficients([actor, other], [lp, gp], [torch.tensor([9., 3., 1., 9.]), torch.zeros(4)], a, calibration, "far_zero")
    torch.testing.assert_close(first[0], torch.tensor([2., -1., 0., 0.], dtype=torch.float64))
    torch.testing.assert_close(second[0], torch.tensor([2., 0., -3., 0.], dtype=torch.float64))
    torch.testing.assert_close(mod.rebuttal_actor_loss(lp, first[0]), -(2 * lp[0] - lp[1]) / 4)
    cap, _ = mod.q2_pair_coefficients([actor, other], [lp, gp], [torch.tensor([9., 2., 3., 9.]), torch.zeros(4)], a, calibration, "far_cap")
    torch.testing.assert_close(cap[0], first[0])


def test_q2_unmatchable_zero_global_force_is_reported_without_epsilon():
    _, _, actor, _, context = _rebuttal_toy()
    other = torch.nn.Linear(1, 1, bias=False)
    lp = mod.current_action_log_prob(actor, torch.from_numpy(context["obs"][:2]), torch.from_numpy(context["data"].actions[:2]))[0]
    zero_lp = other.weight.sum() * torch.zeros(2)
    a = -torch.ones(2)
    coefficients, stats = mod.q2_pair_coefficients([actor, other], [lp, zero_lp], [torch.zeros(2), torch.zeros(2)], a,
                                                  {"far_threshold": 1., "far_cap": 1.}, "far_zero")
    assert stats["target_budget"] > 0 and stats["raw_global_budget"] == 0
    assert stats["alpha"] == 0 and stats["unmatched_zero_raw_gradient"] == 1
    assert stats["matchable"] == 0 and stats["relative_error_nonzero_target"] == 1
    assert torch.equal(coefficients[1], torch.zeros(2, dtype=torch.float64))


def test_q2_calibration_is_initial_only_deterministic_and_preserves_rng_and_gradients():
    _, _, actor, _, context = _rebuttal_toy()
    context["initial_advantage"] = np.tile(np.array([-1., 1.]), 16)
    actor.log_std.grad = torch.full_like(actor.log_std, 7.)
    state = torch.get_rng_state().clone()
    first = mod.calibrate_rebuttal_q2(actor, context, 200, torch.device("cpu"))
    second = mod.calibrate_rebuttal_q2(actor, context, 200, torch.device("cpu"))
    assert first == second and first["status"] == "defined"
    _, d, _ = mod.current_action_log_prob(actor, torch.from_numpy(context["obs"]), torch.from_numpy(context["data"].actions))
    # The float32 mean network may use different GEMM kernels for diagnostic
    # chunks and this full-population reference. Repeated calibration itself
    # is still checked for exact determinism above.
    np.testing.assert_allclose(first["far_threshold"], np.quantile(d.numpy()[context["initial_advantage"] < 0], .75),
                               rtol=1e-6, atol=1e-8)
    assert first["far_cap"] == np.median(first["initial_far_block_norms"])
    assert first["calibration_batches"] == 16 and first["calibration_batch_size"] == 1024
    assert torch.equal(torch.get_rng_state(), state)
    assert torch.equal(actor.log_std.grad, torch.full_like(actor.log_std, 7.))
    context["initial_advantage"] = np.ones(32)
    undefined = mod.calibrate_rebuttal_q2(actor, context, 200, torch.device("cpu"))
    assert undefined["status"] == "undefined_no_initial_negative"
    assert undefined["far_cap"] is undefined["far_threshold"] is None


def test_q2_block_statistics_count_nonzero_targets_and_resume_in_constant_space():
    block = mod._Q2BudgetBlock(torch.device("cpu"))
    block.add({"nonzero_target": torch.tensor(0.), "relative_error_nonzero_target": torch.tensor(0.), "alpha": torch.tensor(0.)})
    resumed = mod._Q2BudgetBlock(torch.device("cpu"), block.state_dict())
    resumed.add({"nonzero_target": torch.tensor(1.), "relative_error_nonzero_target": torch.tensor(.5), "alpha": torch.tensor(2.)})
    row = resumed.finish(2)
    assert row["updates"] == 2 and row["q2_nonzero_target_steps"] == 1
    assert row["q2_relative_error_nonzero_target_mean"] == .5
    assert row["q2_alpha_mean"] == 1 and row["q2_alpha_max"] == 2
    assert resumed.state_dict() is None
    resumed.add({"nonzero_target": torch.tensor(0.), "relative_error_nonzero_target": torch.tensor(0.)})
    row = resumed.finish(3)
    assert row["q2_relative_error_nonzero_target_mean"] is row["q2_relative_error_nonzero_target_max"] is None


def test_rebuttal_behavior_logp_and_transition_ids_follow_loader_limit(tmp_path):
    path = tmp_path / "data.hdf5"
    with h5py.File(path, "w") as handle:
        handle["observations"] = np.zeros((6, 3), np.float32)
        handle["actions"] = np.zeros((6, 2), np.float32)
        handle["rewards"] = np.arange(6, dtype=np.float32)
        handle["terminals"] = np.zeros(6, bool)
        handle["infos/action_log_probs"] = np.arange(6, dtype=np.float32) + .5
    data = mod.load_hopper_hdf5(path, 4)
    np.testing.assert_array_equal(data.behavior_log_prob, [.5, 1.5, 2.5, 3.5])
    np.testing.assert_array_equal(data.transition_ids, np.arange(4))


def test_rebuttal_mass_uses_natural_population_and_explicit_ratio_boundaries():
    values = np.array([2., -1., -3., 0.])
    mass = mod.rebuttal_mass(values)
    assert mass["p"] == .5 and mass["q"] == 1 and mass["rho"] == 2
    assert mass["p"] - mass["q"] == values.mean()
    assert mass["p"] + mass["q"] == np.abs(values).mean()
    assert mod.rebuttal_mass(np.array([-1., 0.]))["ratio_status"] == "p_zero_q_positive"
    both = mod.rebuttal_mass(np.zeros(4))
    assert both["rho"] is None and both["ratio_status"] == "both_zero"


def test_rebuttal_raw_is_score_gradient_including_ratios_above_one():
    settings, _, actor, _, context = _rebuttal_toy()
    obs = torch.from_numpy(context["obs"][:4])
    actions = torch.from_numpy(context["data"].actions[:4])
    lp, d, _ = mod.current_action_log_prob(actor, obs, actions)
    ratios = torch.tensor([1., 2., .5, 7.], dtype=torch.float64)
    behavior = lp.detach() - ratios.log()
    a = torch.tensor([2., -1., -3., .2], dtype=torch.float64)
    c = mod.rebuttal_actor_coefficients(a, lp, d, behavior, "is", settings)
    assert not c.requires_grad
    torch.testing.assert_close(c, ratios * a)
    params = list(actor.parameters())
    expected = [torch.zeros_like(p) for p in params]
    for i in range(len(a)):
        grads = torch.autograd.grad(lp[i], params, retain_graph=True)
        for result, grad in zip(expected, grads):
            result.add_(-ratios[i] * a[i] * grad / len(a))
    actual = torch.autograd.grad(mod.rebuttal_actor_loss(lp, c), params, retain_graph=True)
    for first, second in zip(actual, expected):
        torch.testing.assert_close(first, second)
    one = mod.rebuttal_actor_coefficients(a, lp, d, lp.detach(), "is", settings)
    signed = mod.rebuttal_actor_coefficients(a, lp, d, behavior, "signed", settings)
    torch.testing.assert_close(one, signed)


def test_rebuttal_tanh_action_ratio_matches_native_ratio_and_counts_boundaries():
    _, _, actor, _, context = _rebuttal_toy()
    obs = torch.from_numpy(context["obs"][:3])
    action = torch.tensor([[.1, -.2], [.4, .7], [1., -1.]])
    lp, _, clipped = mod.current_action_log_prob(actor, obs, action)
    safe = action.double().clamp(-1 + 1e-6, 1 - 1e-6)
    u = torch.atanh(safe)
    mean, ls = actor.latent_parameters(obs)
    learner_native = torch.distributions.Normal(mean.double(), ls.double().exp()).log_prob(u).sum(-1)
    behavior_native = torch.distributions.Normal(torch.zeros_like(u), torch.ones_like(u)).log_prob(u).sum(-1)
    behavior_action = behavior_native - torch.log1p(-safe.square()).sum(-1)
    torch.testing.assert_close(lp - behavior_action, learner_native - behavior_native)
    assert clipped.tolist() == [False, False, True]


def test_rebuttal_exp_mapping_stop_gradient_and_no_retention_floor():
    settings, *_ = _rebuttal_toy()
    d = torch.tensor([0., 4., 10000.], dtype=torch.float64, requires_grad=True)
    c = mod.rebuttal_actor_coefficients(-torch.ones(3), torch.zeros(3), d, torch.zeros(3), "drpo", settings)
    assert not c.requires_grad
    torch.testing.assert_close(-c, torch.exp(-.03066675 * d.detach()))
    assert 0 < -c[-1] < np.exp(-40)


def test_rebuttal_positive_only_and_force_keep_full_batch_denominator():
    _, _, actor, _, context = _rebuttal_toy()
    obs, actions = context["obs"][:4], context["data"].actions[:4]
    lp, _, _ = mod.current_action_log_prob(actor, torch.from_numpy(obs), torch.from_numpy(actions))
    c = torch.tensor([2., 0., 0., 0.], dtype=torch.float64)
    torch.testing.assert_close(mod.rebuttal_actor_loss(lp, c), -2 * lp[0] / 4)
    force = mod._rebuttal_force(actor, obs[:1], actions[:1], np.array([2.]), 4, 4, torch.device("cpu"))
    grads = torch.autograd.grad(-mod.rebuttal_actor_loss(lp, c), list(actor.parameters()))
    torch.testing.assert_close(force, torch.cat([g.flatten().double() for g in grads]))


def test_rebuttal_td_restores_value_units_and_uses_frozen_advantage_constants():
    settings, _, _, critic, context = _rebuttal_toy()
    obs, nxt = torch.from_numpy(context["obs"][:4]), torch.from_numpy(context["next_obs"][:4])
    reward = torch.tensor([1., 2., 3., 4.])
    done = torch.tensor([False, True, False, True])
    a, delta = mod.current_rebuttal_advantage(critic, obs, nxt, reward, done, context["target_norm"], .99, .3, 2.)
    value, next_value = critic(obs).detach().double() * 7 + 30, critic(nxt).detach().double() * 7 + 30
    expected = reward.double() + .99 * (~done).double() * next_value - value
    torch.testing.assert_close(delta, expected)
    torch.testing.assert_close(a, (expected - .3) / 2)
    opt = torch.optim.SGD(critic.parameters(), lr=.01)
    before = [p.detach().clone() for p in critic.parameters()]
    normalized_target = (reward.double() + .99 * (~done).double() * next_value - 30) / 7
    loss = .5 * (critic(obs) - normalized_target).square().mean()
    grads = torch.autograd.grad(loss, list(critic.parameters()))
    settings["critic_gradient_clip"] = 1e9
    stats = mod.update_rebuttal_critic(critic, opt, obs, nxt, reward, done, context["target_norm"], .99, settings)
    assert stats["finite"]
    for parameter, previous, grad in zip(critic.parameters(), before, grads):
        torch.testing.assert_close(parameter, previous - .01 * grad)
    changed, _ = mod.current_rebuttal_advantage(critic, obs, nxt, reward, done, context["target_norm"], .99, .3, 2.)
    assert not torch.equal(changed, a)


def test_rebuttal_diagnostics_preserve_gradients_rng_modes_and_optimizer():
    _, _, actor, critic, context = _rebuttal_toy()
    actor.eval()
    optimizer = torch.optim.AdamW(actor.parameters())
    for parameter in actor.parameters():
        parameter.grad = torch.ones_like(parameter)
    optimizer.step()
    before = [p.detach().clone() for p in actor.parameters()]
    grads = [p.grad.clone() for p in actor.parameters()]
    moments = [v["exp_avg"].clone() for v in optimizer.state.values()]
    rng = torch.get_rng_state().clone()
    first, previous = mod.collect_rebuttal_diagnostics(actor, critic, context, "drpo", "frozen", 0, None, torch.device("cpu"))
    rows = mod.collect_rebuttal_gradient_variance(actor, critic, context, "frozen", "drpo", 0, 200, torch.device("cpu"))
    second, _ = mod.collect_rebuttal_diagnostics(actor, critic, context, "drpo", "frozen", 1, previous, torch.device("cpu"))
    assert first["p_raw"] == second["p_raw"] and first["q_raw"] == second["q_raw"]
    assert second["previous_rms_change"] == 0
    assert all(row["status"] == "finite" for row in rows)
    assert not actor.training
    torch.testing.assert_close(torch.get_rng_state(), rng)
    for p, saved, grad in zip(actor.parameters(), before, grads):
        torch.testing.assert_close(p, saved, rtol=0, atol=0)
        torch.testing.assert_close(p.grad, grad, rtol=0, atol=0)
    for state, moment in zip(optimizer.state.values(), moments):
        torch.testing.assert_close(state["exp_avg"], moment, rtol=0, atol=0)


def test_rebuttal_ess_is_stable_when_unscaled_weights_overflow():
    result = mod._rebuttal_weight_statistics(np.array([1000., 1000., -1000.]))
    assert result["ess"] == 2
    assert result["ratio_nonfinite_fraction"] == 2 / 3
    assert result["ratio_zero_fraction"] == 1 / 3


def test_rebuttal_large_finite_mass_and_ratio_means_do_not_overflow_during_summing():
    with np.errstate(over="ignore"):
        mass = mod.rebuttal_mass(np.full(8, 1e308))
        weights = mod._rebuttal_weight_statistics(np.full(8, 709.))
    assert mass["p"] == 1e308 and mass["q"] == 0 and mass["rho"] == 0
    assert weights["ratio_nonfinite_fraction"] == 0 and weights["ess"] == 8
    np.testing.assert_allclose(weights["ratio_mean"], np.exp(709.), rtol=1e-14)


def test_rebuttal_gradient_variance_matches_direct_sample_covariance_trace():
    settings, _, actor, critic, context = _rebuttal_toy()
    settings["variance_batches"] = 4
    rows = mod.collect_rebuttal_gradient_variance(actor, critic, context, "frozen", "signed", 0, 200, torch.device("cpu"))
    rng = np.random.default_rng(np.random.SeedSequence([settings["diagnostic_seed"], 200, 0, 19]))
    gradients = []
    for _ in range(4):
        ids = rng.choice(context["train_ids"], size=8, replace=False)
        logp, _, _ = mod.current_action_log_prob(actor, torch.from_numpy(context["obs"][ids]), torch.from_numpy(context["data"].actions[ids]))
        loss = -(torch.from_numpy(context["initial_advantage"][ids]) * logp).mean()
        gradients.append(torch.cat([g.flatten().double() for g in torch.autograd.grad(loss, list(actor.parameters()))]).numpy())
    expected = np.var(np.stack(gradients), axis=0, ddof=1).sum()
    signed = next(row for row in rows if row["rule"] == "signed")
    np.testing.assert_allclose(signed["variance_trace"], expected, rtol=1e-10)


def test_rebuttal_zero_positive_mass_and_empty_force_groups_are_observations():
    _, _, actor, critic, context = _rebuttal_toy()
    with torch.no_grad():
        for parameter in actor.parameters():
            parameter.zero_()
    context["data"].actions[:] = 0
    context["initial_advantage"][:] = -1
    row, _ = mod.collect_rebuttal_diagnostics(actor, critic, context, "positive_only", "frozen", 0, None, torch.device("cpu"))
    assert row["p_raw"] == 0 and row["q_raw"] == 1 and row["rho_raw"] is None
    assert row["ratio_status_effective"] == "both_zero"
    assert row["positive_count"] == row["far_negative_count"] == 0
    assert row["raw_positive_force_norm"] == row["effective_negative_force_norm"] == 0


def test_rebuttal_gradient_norm_reduction_does_not_overflow_on_finite_is_gradients():
    parameter = torch.nn.Parameter(torch.zeros(2))
    parameter.grad = torch.tensor([3e25, 4e25])
    before = mod._clip_rebuttal_gradients([parameter], 100.)
    assert torch.isfinite(before)
    np.testing.assert_allclose(float(before), 5e25, rtol=1e-6)
    np.testing.assert_allclose(parameter.grad.numpy(), [60., 80.], rtol=1e-6)


def test_episode_ids_and_discounted_returns() -> None:
    terminals = np.array([0, 1, 0, 0], dtype=bool)
    timeouts = np.array([0, 0, 0, 1], dtype=bool)
    ids = mod.build_episode_ids(terminals, timeouts)
    np.testing.assert_array_equal(ids, np.array([0, 0, 1, 1]))
    returns = mod.discounted_returns(
        np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32),
        terminals,
        timeouts,
        0.5,
    )
    np.testing.assert_allclose(returns, np.array([2.0, 2.0, 5.0, 4.0]))


def test_gaussian_components_and_autograd_identity() -> None:
    torch.manual_seed(0)
    policy = mod.SquashedGaussianPolicy(3, 2, (8,), -5.0, 2.0, 1e-6)
    obs = torch.randn(6, 3)
    actions = torch.tanh(torch.randn(6, 2))
    components = policy.score_components(obs, actions)
    torch.testing.assert_close(
        components["corrected_q_xi"], components["radius"].square()
    )
    assert torch.isfinite(components["joint_output_score_norm"]).all()
    error = mod.analytic_output_autograd_relative_error(
        policy,
        obs.numpy(),
        actions.numpy(),
        np.arange(6, dtype=np.int64),
        torch.device("cpu"),
    )
    assert error < 1e-6


def test_squashed_gaussian_is_finite_near_action_boundary() -> None:
    policy = mod.SquashedGaussianPolicy(3, 2, (8,), -5.0, 2.0, 1e-6)
    obs = torch.zeros(4, 3)
    actions = torch.tensor([[0.999999, -0.999999]] * 4)
    log_prob = policy.log_prob(obs, actions)
    components = policy.score_components(obs, actions)
    assert torch.isfinite(log_prob).all()
    assert torch.isfinite(components["radius"]).all()
    assert torch.isfinite(components["raw_log_scale_score_norm"]).all()


def test_advantage_matching_separates_distance_and_matches_magnitude() -> None:
    rng = np.random.default_rng(0)
    count = 400
    advantages = -rng.uniform(0.1, 2.0, size=count).astype(np.float32)
    distances = np.linspace(0.2, 5.0, count).astype(np.float32)
    negative = np.arange(count, dtype=np.int64)
    near, far, summary = mod.match_near_far_indices(
        advantages,
        distances,
        negative,
        0.25,
        0.75,
        20,
        64,
        0.10,
        7,
    )
    assert len(near) > 0
    assert len(near) == len(far)
    assert summary["distance_far_near_ratio"] > 2.0
    assert 0.9 <= summary["advantage_magnitude_far_near_ratio"] <= 1.1


def test_far_cap_and_budget_global_are_supported() -> None:
    torch.manual_seed(1)
    policy = mod.SquashedGaussianPolicy(2, 1, (4,), -5.0, 2.0, 1e-6)
    obs = torch.zeros(4, 2)
    actions = torch.tensor([[0.0], [0.2], [0.95], [-0.95]])
    advantages = torch.tensor([1.0, -1.0, -1.0, -1.0])
    signed_loss, _ = mod.actor_batch_loss(
        policy, obs, actions, advantages, "signed", 1.0, 1.0, 0.5
    )
    capped_loss, diagnostics = mod.actor_batch_loss(
        policy, obs, actions, advantages, "far_cap", 1.0, 1.0, 0.5
    )
    global_loss, _ = mod.actor_batch_loss(
        policy,
        obs,
        actions,
        advantages,
        "budget_matched_global",
        1.0,
        0.5,
        0.5,
    )
    assert torch.isfinite(signed_loss)
    assert torch.isfinite(capped_loss)
    assert torch.isfinite(global_loss)
    assert diagnostics["far_cap_factor_mean"] < 1.0


def test_config_is_bound_to_q2() -> None:
    config = mod.load_config(
        Path(__file__).parents[1] / "configs" / "e7_hopper_q2_medium_replay_v2.yaml"
    )
    assert config.experiment_id == "EXT-H-E7-Q2"
    assert config.dataset_basename == "hopper_medium_replay-v2.hdf5"
    assert config.formal.seeds == tuple(range(100, 110))
    assert config.pilot.canonical_critic_seed == 42
    assert config.formal.canonical_critic_seed == 100
    assert config.pilot_rollout_required is True
    assert config.formal_rollout_required is True
    assert config.env_backend == "gymnasium_mujoco"
    assert config.rollout_dataset_id == "hopper-medium-replay-v2"
    assert config.env_id == "Hopper-v4"
    assert config.normalized_score_reference_min == -20.272305
    assert config.normalized_score_reference_max == 3234.3
    assert config.process_isolated_preflight is True
    assert config.rollout_preflight_timeout_seconds == 120
    assert config.rollout_preflight_max_steps == 2000
    assert config.formal.rollout_episodes == 5
    assert config.formal.final_rollout_episodes == 20
    assert config.activation == "tanh"
    assert config.init_scheme == "default"
    assert config.init_gain == 1.0


def test_network_builder_supports_recovered_relu_orthogonal_profile() -> None:
    torch.manual_seed(0)
    policy = mod.SquashedGaussianPolicy(
        3,
        2,
        (8,),
        -5.0,
        2.0,
        1e-6,
        activation="relu",
        init_scheme="orthogonal",
        init_gain=float(np.sqrt(2)),
    )
    assert any(isinstance(layer, torch.nn.ReLU) for layer in policy.mean_net.modules())
    assert not any(isinstance(layer, torch.nn.Tanh) for layer in policy.mean_net.modules())
    first = next(layer for layer in policy.mean_net.modules() if isinstance(layer, torch.nn.Linear))
    gram = first.weight.detach().T @ first.weight.detach()
    expected = torch.eye(gram.shape[0]) * 2.0
    torch.testing.assert_close(gram, expected, atol=1e-5, rtol=1e-5)


def test_legacy_hdf5_loader(tmp_path: Path) -> None:
    path = tmp_path / "tiny.hdf5"
    with h5py.File(path, "w") as handle:
        handle["observations"] = np.zeros((6, 3), dtype=np.float32)
        handle["actions"] = np.zeros((6, 2), dtype=np.float32)
        handle["rewards"] = np.arange(6, dtype=np.float32)
        handle["next_observations"] = np.ones((6, 3), dtype=np.float32)
        handle["terminals"] = np.array([0, 1, 0, 0, 0, 1], dtype=np.float32)
        handle["timeouts"] = np.zeros(6, dtype=np.float32)
    data = mod.load_hopper_hdf5(path, None)
    assert data.size == 6
    np.testing.assert_array_equal(data.episode_ids, np.array([0, 0, 1, 1, 1, 1]))


def test_registry_records_completed_q2_without_claiming_method_ranking() -> None:
    import yaml

    root = Path(__file__).parents[1]
    registry = yaml.safe_load((root / "experiments" / "registry.yaml").read_text())
    entry = next(item for item in registry["experiments"] if item["id"] == "EXT-H-E7-Q2")
    assert entry["status"] == "long_run_validated"
    assert entry["scientific_status"] == "long_run_validated"
    assert entry["implementation_state"] == "implemented"
    assert entry["execution_gate"]["state"] == "blocked"
    assert entry["execution_gate"]["blocked_by"] == [
        "completed_formal_execution_no_rerun_without_new_registration"
    ]
    assert entry["formal_execution"]["activation_state"] == "blocked"
    assert entry["formal_execution"]["entrypoint"] == "src/drpo/e7_hopper_q2.py"
    assert entry["formal_execution"]["runner_archive_policy"]["mode"] == "forbid"
    assert entry["evidence"]["code_committed"] is True
    assert entry["evidence"]["implementation_tests_passed"] is True
    assert entry["evidence"]["run_started"] is True
    assert entry["evidence"]["raw_complete"] is True
    assert entry["evidence"]["terminal_audited"] is True
    assert entry["evidence"]["formal_seeds_completed"] == 10
    assert entry["closure"]["method_ranking_claim_allowed"] is False
    assert entry["closure"]["finite_terminal_claim_allowed"] is False
    rollout = entry["rollout_evaluation"]
    assert rollout["offline_dataset_id"] == "hopper-medium-replay-v2"
    assert rollout["evaluation_env_id"] == "Hopper-v4"
    assert rollout["backend"] == "gymnasium_mujoco"
    assert rollout["legacy_d4rl_fallback"] == "forbidden"
    assert rollout["process_isolated_preflight"] is True


def test_handoff_preserves_history_under_v43_fixed_budget_route() -> None:
    handoff = (Path(__file__).parents[1] / "docs" / "handoff.md").read_text()
    assert "v42 增量登记：状态机一致性、E7 已实现门禁与 E4--E8 路线锁定" in handoff
    assert "v45（E4-TAPER 结果闭环、环境识别与公平性边界版）" in handoff
    assert "v41 增量登记：`EXT-H-E7-Q2` Hopper 实现" in handoff
    assert "implemented + not_run + blocked" in handoff
    assert "CPU 单元测试、静态检查和缩减工程 pilot" in handoff
    assert "尚未运行（not_run）" in handoff
    assert "不能替代 C-U1" in handoff
    assert "v54 增量登记：`EXT-H-E7-Q2` canonical critic" in handoff
    assert "每个 run 只训练或严格复用一个 canonical critic artifact" in handoff
    assert "task_performance_collapse=null" in handoff
    assert "formal gate 也必须为 false" in handoff
    assert "v58 增量登记：`EXT-H-E7-Q2` Gymnasium `Hopper-v4` rollout" in handoff
    assert "离线训练数据仍是 `hopper-medium-replay-v2`" in handoff
    assert "legacy D4RL/mujoco-py fallback 明确禁止" in handoff


def test_task_performance_unavailable_is_not_reported_as_no_collapse() -> None:
    config = mod.load_config(
        Path(__file__).parents[1] / "configs" / "e7_hopper_q2_medium_replay_v2.yaml"
    )
    rows = []
    for step in range(6):
        rows.append(
            {
                "step": step,
                "loss": 1.0,
                "positive_nll": 1.0,
                "gradient_norm": 1.0,
                "update_norm": 1.0,
                "sigma_mean": 1.0,
                "mean_boundary_fraction": 0.0,
                "log_std_min_fraction": 0.0,
                "log_std_max_fraction": 0.0,
                "mean_abs": 0.1,
                "phantom_distance_mean": 1.0,
                "rollout_status": "unavailable",
                "normalized_return": float("nan"),
            }
        )
    audit = mod.classify_actor_terminal(rows, config, None, False)
    assert audit["task_performance_status"] == "unavailable"
    assert audit["task_performance_collapse"] is None
    assert audit["normalized_return_available"] is False


def test_d4rl_reference_normalization_endpoints() -> None:
    lo = -20.272305
    hi = 3234.3
    assert mod.normalize_d4rl_reference_score(lo, lo, hi, percent=True) == 0.0
    assert mod.normalize_d4rl_reference_score(hi, lo, hi, percent=True) == 100.0
    midpoint = (lo + hi) / 2.0
    assert mod.normalize_d4rl_reference_score(midpoint, lo, hi, percent=True) == 50.0


def test_rollout_preflight_exercises_gymnasium_and_manual_score(
    tmp_path: Path, monkeypatch
) -> None:
    class FakeSpace:
        low = np.array([-1.0, -1.0], dtype=np.float32)
        high = np.array([1.0, 1.0], dtype=np.float32)

        def seed(self, seed: int) -> None:
            self._seed = seed

        def sample(self) -> np.ndarray:
            return np.array([0.25, -0.25], dtype=np.float32)

    class FakeSpec:
        max_episode_steps = 3

    class FakeEnv:
        action_space = FakeSpace()
        spec = FakeSpec()

        def __init__(self) -> None:
            self.steps = 0

        def reset(self, seed: int | None = None):
            self.steps = 0
            return np.zeros(3, dtype=np.float32), {"seed": seed}

        def step(self, action: np.ndarray):
            self.steps += 1
            done = self.steps >= 2
            return np.ones(3, dtype=np.float32), 1.0, done, {"ok": True}

        def close(self) -> None:
            pass

    monkeypatch.setattr(
        mod,
        "_open_gymnasium_mujoco_env",
        lambda env_id: (
            FakeEnv(),
            {
                "backend": "gymnasium_mujoco",
                "evaluation_env_id": env_id,
                "legacy_d4rl_fallback": "forbidden",
            },
        ),
    )
    report = mod.preflight_rollout_environment(
        backend="gymnasium_mujoco",
        dataset_id="hopper-medium-replay-v2",
        env_id="Hopper-v4",
        expected_observation_dim=3,
        expected_action_dim=2,
        seed=7,
        max_steps=5,
        normalized_score_percent=True,
        reference_min_score=0.0,
        reference_max_score=10.0,
        output_dir=tmp_path / "preflight",
        required=True,
        process_isolated=False,
        timeout_seconds=30,
    )
    assert report["status"] == "passed"
    assert report["interaction_verified"] is True
    assert report["normalized_score_verified"] is True
    assert report["random_episode"]["steps"] == 2
    assert report["random_episode"]["normalized_return"] == 20.0
    assert report["legacy_d4rl_fallback"] == "forbidden"
    assert (tmp_path / "preflight" / "rollout_preflight.json").is_file()


def test_rollout_preflight_failure_persists_traceback_before_raising(
    tmp_path: Path, monkeypatch
) -> None:
    def fail_open(env_id: str):
        raise RuntimeError("synthetic gymnasium failure")

    monkeypatch.setattr(mod, "_open_gymnasium_mujoco_env", fail_open)
    output_dir = tmp_path / "preflight_failure"
    import pytest

    with pytest.raises(RuntimeError, match="Gymnasium rollout preflight failed"):
        mod.preflight_rollout_environment(
            backend="gymnasium_mujoco",
            dataset_id="hopper-medium-replay-v2",
            env_id="Hopper-v4",
            expected_observation_dim=3,
            expected_action_dim=2,
            seed=7,
            max_steps=5,
            normalized_score_percent=True,
            reference_min_score=0.0,
            reference_max_score=10.0,
            output_dir=output_dir,
            required=True,
            process_isolated=False,
            timeout_seconds=30,
        )
    report = __import__("json").loads(
        (output_dir / "rollout_preflight.json").read_text()
    )
    assert report["status"] == "failed"
    assert report["error_type"] == "RuntimeError"
    assert "synthetic gymnasium failure" in report["error"]
    assert "Traceback" in report["traceback"]


def test_process_isolated_preflight_captures_sigsegv(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(
        mod.subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(
            args=args[0], returncode=-11, stdout="", stderr="native crash"
        ),
    )
    report = mod.preflight_rollout_environment(
        backend="gymnasium_mujoco",
        dataset_id="hopper-medium-replay-v2",
        env_id="Hopper-v4",
        expected_observation_dim=11,
        expected_action_dim=3,
        seed=7,
        max_steps=5,
        normalized_score_percent=True,
        reference_min_score=-20.272305,
        reference_max_score=3234.3,
        output_dir=tmp_path / "segfault",
        required=False,
        process_isolated=True,
        timeout_seconds=30,
    )
    assert report["status"] == "failed"
    assert report["error_type"] == "NativeProcessSignal"
    assert report["subprocess_isolation"]["signal_name"] == "SIGSEGV"
    assert report["legacy_d4rl_fallback"] == "forbidden"


def test_rollout_evaluation_uses_manual_reference_normalization(monkeypatch) -> None:
    class FakeSpace:
        low = np.array([-1.0, -1.0], dtype=np.float32)
        high = np.array([1.0, 1.0], dtype=np.float32)

    class FakeSpec:
        max_episode_steps = 2

    class FakeEnv:
        action_space = FakeSpace()
        spec = FakeSpec()

        def __init__(self) -> None:
            self.steps = 0

        def reset(self, seed: int | None = None):
            self.steps = 0
            return np.zeros(3, dtype=np.float32), {}

        def step(self, action: np.ndarray):
            self.steps += 1
            return np.zeros(3, dtype=np.float32), 2.0, self.steps >= 2, {}

        def close(self) -> None:
            pass

    monkeypatch.setattr(
        mod,
        "_open_gymnasium_mujoco_env",
        lambda env_id: (FakeEnv(), {"evaluation_env_id": env_id}),
    )
    policy = mod.SquashedGaussianPolicy(3, 2, (4,), -5.0, 2.0, 1e-6)
    obs_norm = mod.Normalizer(
        mean=np.zeros(3, dtype=np.float32),
        std=np.ones(3, dtype=np.float32),
    )
    result = mod.evaluate_d4rl_rollouts(
        policy=policy,
        obs_norm=obs_norm,
        backend="gymnasium_mujoco",
        dataset_id="hopper-medium-replay-v2",
        env_id="Hopper-v4",
        episodes=2,
        seed=10,
        device=torch.device("cpu"),
        normalized_score_percent=True,
        reference_min_score=0.0,
        reference_max_score=8.0,
        required=True,
    )
    assert result["rollout_return_mean"] == 4.0
    assert result["normalized_return"] == 50.0
    assert result["offline_dataset_id"] == "hopper-medium-replay-v2"
    assert result["evaluation_env_id"] == "Hopper-v4"


def test_gymnasium_open_path_never_imports_d4rl(monkeypatch) -> None:
    imported: list[str] = []

    class FakeGymnasium:
        __name__ = "gymnasium"

        @staticmethod
        def make(env_id: str):
            assert env_id == "Hopper-v4"
            return type("FakeEnv", (), {"close": lambda self: None})()

    def fake_import(name: str):
        imported.append(name)
        if name == "gymnasium":
            return FakeGymnasium()
        raise AssertionError(f"unexpected import: {name}")

    monkeypatch.setattr(mod.importlib, "import_module", fake_import)
    env, metadata = mod._open_gymnasium_mujoco_env("Hopper-v4")
    env.close()
    assert imported == ["gymnasium"]
    assert metadata["legacy_d4rl_fallback"] == "forbidden"



def test_canonical_critic_is_trained_once_and_strictly_reused(tmp_path: Path) -> None:
    from dataclasses import replace

    config_path = (
        Path(__file__).parents[1] / "configs" / "e7_hopper_q2_medium_replay_v2.yaml"
    )
    config = mod.load_config(config_path)
    mode = mod.ModeConfig(
        max_transitions=None,
        seeds=(7, 8),
        canonical_critic_seed=7,
        critic_max_steps=4,
        critic_min_steps=2,
        critic_eval_interval=1,
        positive_max_steps=2,
        positive_min_steps=1,
        actor_eval_interval=1,
        branch_max_steps=2,
        branch_min_steps=1,
        matched_pairs=2,
        audit_sample_size=4,
        rollout_episodes=1,
        final_rollout_episodes=2,
        rollout_eval_interval=1,
    )
    config = replace(
        config,
        hidden_sizes=(4,),
        critic_batch_size=8,
        audit_windows=2,
        critic_relative_slope_tolerance=1e9,
        critic_gradient_tolerance=1e9,
        critic_update_tolerance=1e9,
    )
    episode_ids = np.repeat(np.arange(10), 3)
    size = len(episode_ids)
    observations = np.linspace(-1.0, 1.0, size * 3, dtype=np.float32).reshape(size, 3)
    actions = np.zeros((size, 2), dtype=np.float32)
    rewards = np.linspace(0.0, 1.0, size, dtype=np.float32)
    terminals = np.zeros(size, dtype=bool)
    terminals[2::3] = True
    data = mod.OfflineData(
        observations=observations,
        actions=actions,
        rewards=rewards,
        next_observations=np.roll(observations, -1, axis=0),
        terminals=terminals,
        timeouts=np.zeros(size, dtype=bool),
        episode_ids=episode_ids,
    )
    dataset_file = tmp_path / "fake.hdf5"
    dataset_file.write_bytes(b"fake")
    dataset_manifest = {
        "basename": dataset_file.name,
        "sha256": mod.sha256_file(dataset_file),
        "size_bytes": dataset_file.stat().st_size,
    }
    root = tmp_path / "canonical"
    first = mod.prepare_canonical_critic_context(
        data=data,
        config=config,
        mode=mode,
        mode_name="pilot",
        config_path=config_path,
        dataset_manifest=dataset_manifest,
        device=torch.device("cpu"),
        artifact_root=root,
        reuse_root=None,
    )
    second = mod.prepare_canonical_critic_context(
        data=data,
        config=config,
        mode=mode,
        mode_name="pilot",
        config_path=config_path,
        dataset_manifest=dataset_manifest,
        device=torch.device("cpu"),
        artifact_root=root,
        reuse_root=None,
    )
    assert first.reused is False
    assert second.reused is True
    assert first.artifact_manifest["critic_training_count"] == 1
    assert second.artifact_manifest["shared_across_all_actor_seeds"] is True
    assert (
        first.critic_audit["selected_checkpoint_role"]
        == "best_validation_checkpoint"
    )
    assert first.critic_audit["fixed_budget_completed"] is True
    assert first.critic_audit["stopping_rule"] == "fixed_optimizer_steps"
    assert first.critic_audit["terminal_audit_controls_stopping"] is False
    assert first.critic_audit["final_stationarity_reconfirmed"] is True
    assert first.critic_audit["optimization_terminal"] is True
    assert first.critic_audit["critic_accepted_for_frozen_advantage"] is True
    assert first.critic_audit["critic_quality_audit_passed"] is False
    np.testing.assert_array_equal(first.advantages, second.advantages)


def test_pilot_terminal_audit_cannot_report_formal_gate_passed(monkeypatch) -> None:
    from types import SimpleNamespace

    mechanism_counts = {
        "natural_far_field_present": 1,
        "corrected_quadratic_branch_empirically_active": 1,
        "measurable_full_parameter_contribution": 1,
        "targeted_far_control_mitigates_dynamics": 1,
        "terminal_audit_records_complete": 1,
        "terminal_state_classification_complete": 1,
        "rollout_available_for_all_methods": 1,
        "all_mechanism_subchecks_passed": 1,
    }
    monkeypatch.setattr(
        mod,
        "aggregate_seed_summaries",
        lambda summaries: {
            "mechanism_subcheck_counts": mechanism_counts,
            "reporting_separation": {},
        },
    )
    method_audit = {
        "task_performance_status": "available",
        "state": "persistent_or_slow_drift",
        "fixed_budget_completed": True,
        "terminal_audit_complete": True,
    }
    summary = {
        "positive_only_initialization": {
            "state": "persistent_or_slow_drift",
            "fixed_budget_completed": True,
            "terminal_audit_complete": True,
            "task_performance_status": "available",
        },
        "methods": {method: dict(method_audit) for method in mod.METHODS},
    }
    canonical = SimpleNamespace(
        critic_audit={
            "fixed_budget_completed": True,
            "optimization_terminal": False,
        },
        artifact_manifest={"complete": True},
    )
    audit = mod.build_terminal_audit(
        summaries=[summary],
        mode_name="pilot",
        expected_seed_count=1,
        canonical=canonical,
        rollout_preflight={"status": "passed"},
        rollout_required=True,
    )
    assert audit["engineering_pipeline_complete"] is True
    assert audit["critic_fixed_budget_completed"] is True
    assert audit["positive_only_fixed_budget_all_seeds"] is True
    assert audit["all_actor_fixed_budgets_completed"] is True
    assert audit["mechanism_subchecks_passed_for_completed_seeds"] is True
    assert audit["paired_seed_evidence_complete"] is False
    assert audit["formal_evidence_prerequisites_complete"] is False
    assert audit["formal_scientific_gate_passed"] is False
    assert audit["independent_validation_gate_all_seeds"] is False
