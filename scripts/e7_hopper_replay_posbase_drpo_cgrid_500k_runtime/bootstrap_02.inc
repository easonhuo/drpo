        shifted_q = torch.quantile(shifted, quantiles)
        payload["raw_quantiles"] = {
            "p10": float(raw_q[0]),
            "p25": float(raw_q[1]),
            "p50": float(raw_q[2]),
            "p75": float(raw_q[3]),
            "p90": float(raw_q[4]),
        }
        payload["shifted_quantiles"] = {
            "p10": float(shifted_q[0]),
            "p25": float(shifted_q[1]),
            "p50": float(shifted_q[2]),
            "p75": float(shifted_q[3]),
            "p90": float(shifted_q[4]),
        }
    return payload


class TrainingAudit:
    def __init__(
        self,
        branch_dir: Path,
        *,
        total_steps: int = EXPECTED_STEPS,
        interval: int = DIAGNOSTIC_INTERVAL,
    ) -> None:
        self.branch_dir = branch_dir
        self.jsonl_path = branch_dir / "ACTOR_TRAINING_DYNAMICS.jsonl"
        self.latest_path = branch_dir / "ACTOR_TRAINING_DYNAMICS_LATEST.json"
        self.jsonl_path.unlink(missing_ok=True)
        self.latest_path.unlink(missing_ok=True)
        self.total_steps = int(total_steps)
        self.interval = int(interval)
        if self.total_steps <= 0 or self.interval <= 0:
            raise ValueError("training-audit total_steps and interval must be positive")
        self.update_count = 0
        self.records = 0
        self.optimizer_control: dict[str, Any] | None = None

    def install(self, module: Any, target_class: str) -> dict[str, Any]:
        target = getattr(module, target_class)
        original_init = target.__init__
        original_update = target.update
        audit = self

        def wrapped_init(agent: Any, *args: Any, **kwargs: Any) -> None:
            original_init(agent, *args, **kwargs)
            if audit.optimizer_control is not None:
                raise RuntimeError("canonical agent was instantiated more than once")
            actor_lrs = _optimizer_lrs(agent.a_opt)
            critic_lrs = _optimizer_lrs(agent.c_opt)
            for label, values in (("actor", actor_lrs), ("critic", critic_lrs)):
                if not all(
                    math.isclose(value, EXPECTED_LR, rel_tol=0.0, abs_tol=1e-12)
                    for value in values
                ):
                    raise RuntimeError(f"canonical {label} learning rate changed: {values}")
            audit.optimizer_control = {
                "applied_count": 1,
                "actor_lr": actor_lrs,
                "critic_lr": critic_lrs,
                "actor_lr_unchanged": True,
                "critic_lr_unchanged": True,
            }

        def wrapped_update(
            agent: Any,
            s: Any,
            a: Any,
            r: Any,
            ns: Any,
            d: Any,
            ep_ret: Any = None,
        ) -> Any:
            next_update = audit.update_count + 1
            sampled = next_update % audit.interval == 0 or next_update == audit.total_steps
            parameters = [p for p in agent.actor.parameters() if p.requires_grad]
            before = [p.detach().clone() for p in parameters] if sampled else []
            result = original_update(agent, s, a, r, ns, d, ep_ret)
            audit.update_count = next_update
            if sampled:
                gradients = [p.grad for p in parameters if p.grad is not None]
                if len(gradients) != len(parameters):
                    raise RuntimeError("actor training audit found a missing gradient")
                raw_gradient_norm = _tensor_norm(gradients)
                update_norm = _tensor_norm(
                    [after.detach() - prior for after, prior in zip(parameters, before, strict=True)]
                )
                device = next(agent.actor.parameters()).device
                states = torch.as_tensor(s, dtype=torch.float32, device=device)
                with torch.no_grad():
                    output = agent.actor(states)
                    if not isinstance(output, (tuple, list)) or len(output) != 2:
