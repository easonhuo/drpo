            (abs_advantage[far] * negative_factor[far]).sum().cpu()
        ),
    }


def _self_test() -> int:
    raw = torch.tensor([-2.0, -0.1, 0.1, 0.4, 1.2, 3.0])
    stats = _snapshot_stats(raw, 1.0, provider_update=1, include_quantiles=True)
    if stats["samples"] != 6 or stats["effective_positive_count"] != 2:
        raise RuntimeError("snapshot self-test produced the wrong sign split")
    if not math.isclose(stats["effective_positive_fraction"], 2.0 / 6.0, abs_tol=1e-7):
        raise RuntimeError("snapshot self-test produced the wrong positive fraction")
    if not math.isclose(stats["shifted_negative_abs_mass"], 5.6, abs_tol=1e-6):
        raise RuntimeError("snapshot self-test produced the wrong negative mass")
    mass = _near_far_mass(
        torch.tensor([-1.0, -2.0, 3.0]),
        torch.tensor([1.0, 3.0, 4.0]),
        torch.tensor([0.5, 0.25, 1.0]),
    )
    if mass != {
        "near_negative_samples": 1,
        "far_negative_samples": 1,
        "near_raw_negative_mass": 1.0,
        "far_raw_negative_mass": 2.0,
        "near_effective_negative_mass": 0.5,
        "far_effective_negative_mass": 0.5,
    }:
        raise RuntimeError(f"near/far mass self-test failed: {mass}")

    class Actor(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.mean = torch.nn.Linear(2, 1)
            self.log_std = torch.nn.Parameter(torch.zeros(1))

        def forward(self, states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            mean = self.mean(states)
            return mean, self.log_std.expand_as(mean)

    class Agent:
        def __init__(self) -> None:
            self.actor = Actor()
            self.critic = torch.nn.Linear(2, 1)
            self.a_opt = torch.optim.Adam(self.actor.parameters(), lr=EXPECTED_LR)
            self.c_opt = torch.optim.Adam(self.critic.parameters(), lr=EXPECTED_LR)

        def update(
            self, s: Any, a: Any, r: Any, ns: Any, d: Any, ep_ret: Any = None
        ) -> float:
            del a, r, ns, d, ep_ret
            states = torch.as_tensor(s, dtype=torch.float32)
            mean, log_std = self.actor(states)
            loss = mean.square().mean() + 0.01 * log_std.square().mean()
            self.a_opt.zero_grad(set_to_none=True)
            loss.backward()
            self.a_opt.step()
            return float(loss.detach())

    with tempfile.TemporaryDirectory() as temporary:
        module = SimpleNamespace(Agent=Agent)
        audit = TrainingAudit(Path(temporary), total_steps=2, interval=1)
        audit.install(module, "Agent")
        agent = module.Agent()
        states = torch.tensor([[1.0, -1.0], [0.5, 0.25]])
        agent.update(states, None, None, None, None)
        agent.update(states, None, None, None, None)
        completed = audit.validate_complete()
        if completed["records"] != 2:
            raise RuntimeError("training-audit self-test did not record both updates")

    print(
        json.dumps(
            {
                "status": "PASS",
                "advantage_shift": stats,
                "near_far_mass": mass,
                "training_audit": completed,
            },
            sort_keys=True,
        )
    )
    return 0


def _run_bootstrap(argv: list[str]) -> int:
    from drpo import e7_squared_exp_night as suite
    from drpo import e7_squared_exp_night_bootstrap as bootstrap
    from drpo import e7_w0_geometry_diagnostics as geometry_module

    suite.GAE_EXPERIMENT_ID = EXPERIMENT_ID
    suite.TUNING_PROFILE_ID = PROFILE_ID
    suite.EXPECTED_STEPS = EXPECTED_STEPS

    try:
        branch_index = argv.index("--branch-config") + 1
    except (ValueError, IndexError) as exc:
        raise ValueError("bootstrap wrapper requires --branch-config") from exc
    branch_config_path = Path(argv[branch_index]).expanduser().resolve()
    branch_dir = branch_config_path.parent
    snapshots_path = branch_dir / "ADVANTAGE_SHIFT_SNAPSHOTS.jsonl"
