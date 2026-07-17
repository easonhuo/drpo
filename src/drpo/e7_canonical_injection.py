"""Fail-closed adapter for the canonical D4RL signed actor update.

This module intentionally does *not* reimplement the old D4RL networks, critic,
optimizer construction, dataset loop, or rollout code. It loads the user's
canonical source tree at runtime, verifies exact source fingerprints, and only
replaces the signed actor-update method of the configured agent class.

The supported update contract is the historical ``signed_td_v_v1`` skeleton:

* ``actor(state) -> (mean, log_std)``;
* ``critic(state) -> value``;
* actor and critic optimizers are ``a_opt`` and ``c_opt``;
* scalar attributes ``gamma``, ``tau`` and ``alpha`` exist;
* ``update(s, a, r, ns, d, ep_ret=None)`` is the trainer entry point.

A source tree that does not satisfy this contract must fail before training. It
must never be silently treated as canonical IQL/DRPO/SNA2C.
"""

from __future__ import annotations

import dataclasses
import hashlib
import importlib.util
import json
import math
import os
import sys
from pathlib import Path
from types import ModuleType
from typing import Any, Callable, Iterable, Mapping

import torch


SUPPORTED_CONTRACT_VERSION = "e7-canonical-contract-v1"
SUPPORTED_AGENT_FLAVOR = "signed_td_v_v1"
SUPPORTED_METHODS = {
    "positive_only",
    "canonical_signed",
    "global",
    "reciprocal_linear",
    "reciprocal_quadratic",
    "exponential",
}
AdvantageProvider = Callable[[Any, Any, torch.Tensor], torch.Tensor]


class CanonicalContractError(RuntimeError):
    """Raised when the claimed canonical source does not match the contract."""


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    """Return the lowercase SHA-256 digest of one file."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _iter_python_files(root: Path) -> Iterable[Path]:
    ignored_parts = {
        ".git",
        ".hg",
        ".svn",
        "__pycache__",
        ".pytest_cache",
        ".mypy_cache",
        ".ruff_cache",
        "outputs",
        "results",
        "runs",
        "checkpoints",
    }
    for path in sorted(root.rglob("*.py")):
        if any(part in ignored_parts for part in path.relative_to(root).parts):
            continue
        if path.is_file():
            yield path


def python_tree_sha256(root: str | Path) -> str:
    """Hash Python source paths and contents in a deterministic order.

    The digest is deliberately restricted to ``*.py`` files so that generated
    datasets, checkpoints, logs, and virtual environments cannot change the
    identity of an otherwise unchanged source tree.
    """

    root_path = Path(root).expanduser().resolve()
    if not root_path.is_dir():
        raise FileNotFoundError(f"canonical source root does not exist: {root_path}")
    digest = hashlib.sha256()
    count = 0
    for path in _iter_python_files(root_path):
        relative = path.relative_to(root_path).as_posix().encode("utf-8")
        digest.update(relative)
        digest.update(b"\0")
        digest.update(bytes.fromhex(sha256_file(path)))
        digest.update(b"\0")
        count += 1
    if count == 0:
        raise CanonicalContractError(f"no Python files found below {root_path}")
    return digest.hexdigest()


@dataclasses.dataclass(frozen=True)
class CanonicalContract:
    """Immutable identity and interface contract for the old D4RL source."""

    contract_version: str
    canonical_source_root: str
    python_tree_sha256: str
    agents_relpath: str
    agents_sha256: str
    trainer_relpath: str
    trainer_sha256: str
    module_name: str
    target_class: str
    agent_flavor: str
    expected_canonical_alpha: float
    return_mode: str = "zero_float"

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "CanonicalContract":
        required = {
            "contract_version",
            "canonical_source_root",
            "python_tree_sha256",
            "agents_relpath",
            "agents_sha256",
            "trainer_relpath",
            "trainer_sha256",
            "module_name",
            "target_class",
            "agent_flavor",
            "expected_canonical_alpha",
        }
        missing = sorted(required - set(raw))
        if missing:
            raise CanonicalContractError(
                f"canonical contract is missing required fields: {missing}"
            )
        contract = cls(
            contract_version=str(raw["contract_version"]),
            canonical_source_root=str(raw["canonical_source_root"]),
            python_tree_sha256=str(raw["python_tree_sha256"]).lower(),
            agents_relpath=str(raw["agents_relpath"]),
            agents_sha256=str(raw["agents_sha256"]).lower(),
            trainer_relpath=str(raw["trainer_relpath"]),
            trainer_sha256=str(raw["trainer_sha256"]).lower(),
            module_name=str(raw["module_name"]),
            target_class=str(raw["target_class"]),
            agent_flavor=str(raw["agent_flavor"]),
            expected_canonical_alpha=float(raw["expected_canonical_alpha"]),
            return_mode=str(raw.get("return_mode", "zero_float")),
        )
        contract.validate_static()
        return contract

    @classmethod
    def load(cls, path: str | Path) -> "CanonicalContract":
        return cls.from_mapping(json.loads(Path(path).read_text()))

    def validate_static(self) -> None:
        if self.contract_version != SUPPORTED_CONTRACT_VERSION:
            raise CanonicalContractError(
                f"unsupported contract_version={self.contract_version!r}; "
                f"expected {SUPPORTED_CONTRACT_VERSION!r}"
            )
        if self.agent_flavor != SUPPORTED_AGENT_FLAVOR:
            raise CanonicalContractError(
                f"unsupported agent_flavor={self.agent_flavor!r}; "
                f"expected {SUPPORTED_AGENT_FLAVOR!r}"
            )
        if self.return_mode not in {"zero_float", "metrics_dict"}:
            raise CanonicalContractError(
                f"unsupported return_mode={self.return_mode!r}"
            )
        if not math.isfinite(self.expected_canonical_alpha):
            raise CanonicalContractError("expected_canonical_alpha must be finite")
        if not (0.0 < self.expected_canonical_alpha <= 1.0):
            raise CanonicalContractError(
                "expected_canonical_alpha must be in (0, 1]"
            )
        for field_name in ("python_tree_sha256", "agents_sha256", "trainer_sha256"):
            value = getattr(self, field_name)
            if len(value) != 64 or any(ch not in "0123456789abcdef" for ch in value):
                raise CanonicalContractError(
                    f"{field_name} must be a SHA-256 hex digest"
                )
        for relpath in (self.agents_relpath, self.trainer_relpath):
            path = Path(relpath)
            if path.is_absolute() or ".." in path.parts:
                raise CanonicalContractError(
                    f"contract path must stay below canonical_source_root: {relpath}"
                )

    @property
    def source_root(self) -> Path:
        return Path(self.canonical_source_root).expanduser().resolve()

    @property
    def agents_path(self) -> Path:
        return (self.source_root / self.agents_relpath).resolve()

    @property
    def trainer_path(self) -> Path:
        return (self.source_root / self.trainer_relpath).resolve()

    def verify_runtime(self) -> dict[str, Any]:
        """Verify path containment and exact source fingerprints."""

        root = self.source_root
        if not root.is_dir():
            raise CanonicalContractError(f"canonical source root is missing: {root}")
        checks: dict[str, Any] = {
            "contract_version": self.contract_version,
            "canonical_source_root": str(root),
        }
        for label, path, expected in (
            ("agents", self.agents_path, self.agents_sha256),
            ("trainer", self.trainer_path, self.trainer_sha256),
        ):
            try:
                path.relative_to(root)
            except ValueError as exc:
                raise CanonicalContractError(
                    f"{label} path escapes canonical source root: {path}"
                ) from exc
            if not path.is_file():
                raise CanonicalContractError(f"{label} file is missing: {path}")
            actual = sha256_file(path)
            if actual != expected:
                raise CanonicalContractError(
                    f"{label} SHA-256 mismatch: expected {expected}, got {actual}"
                )
            checks[f"{label}_path"] = str(path)
            checks[f"{label}_sha256"] = actual
        tree_actual = python_tree_sha256(root)
        if tree_actual != self.python_tree_sha256:
            raise CanonicalContractError(
                "canonical Python source-tree SHA-256 mismatch: "
                f"expected {self.python_tree_sha256}, got {tree_actual}"
            )
        checks["python_tree_sha256"] = tree_actual
        return checks


@dataclasses.dataclass(frozen=True)
class NegativeControl:
    """One injected negative-gradient branch."""

    method: str
    negative_scale: float
    canonical_alpha: float
    reference_distance: float = 2.0
    reciprocal_linear_coefficient: float = 0.4362580032734791
    reciprocal_quadratic_coefficient: float = 0.5520268617673281
    exponential_coefficient: float = 0.374162511054291

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "NegativeControl":
        control = cls(
            method=str(raw["method"]),
            negative_scale=float(raw["negative_scale"]),
            canonical_alpha=float(raw["canonical_alpha"]),
            reference_distance=float(raw.get("reference_distance", 2.0)),
            reciprocal_linear_coefficient=float(
                raw.get("reciprocal_linear_coefficient", 0.4362580032734791)
            ),
            reciprocal_quadratic_coefficient=float(
                raw.get("reciprocal_quadratic_coefficient", 0.5520268617673281)
            ),
            exponential_coefficient=float(
                raw.get("exponential_coefficient", 0.374162511054291)
            ),
        )
        control.validate()
        return control

    @classmethod
    def load(cls, path: str | Path) -> "NegativeControl":
        return cls.from_mapping(json.loads(Path(path).read_text()))

    def validate(self) -> None:
        if self.method not in SUPPORTED_METHODS:
            raise ValueError(
                f"unsupported method={self.method!r}; expected one of "
                f"{sorted(SUPPORTED_METHODS)}"
            )
        for field in dataclasses.fields(self):
            value = getattr(self, field.name)
            if isinstance(value, float) and not math.isfinite(value):
                raise ValueError(f"{field.name} must be finite")
        if self.negative_scale < 0.0:
            raise ValueError("negative_scale must be non-negative")
        if self.reference_distance <= 0.0:
            raise ValueError("reference_distance must be positive")
        if not (0.0 < self.canonical_alpha <= 1.0):
            raise ValueError("canonical_alpha must be in (0, 1]")
        for name in (
            "reciprocal_linear_coefficient",
            "reciprocal_quadratic_coefficient",
            "exponential_coefficient",
        ):
            if getattr(self, name) < 0.0:
                raise ValueError(f"{name} must be non-negative")
        if self.method == "positive_only" and self.negative_scale != 0.0:
            raise ValueError("positive_only requires negative_scale=0")
        if self.method == "canonical_signed" and self.negative_scale != 1.0:
            raise ValueError("canonical_signed requires negative_scale=1")

    @property
    def effective_alpha(self) -> float:
        if self.method == "positive_only":
            return 0.0
        return self.canonical_alpha * self.negative_scale


def detached_standardized_distance(
    actor: torch.nn.Module,
    states: torch.Tensor,
    actions: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return ``(mean, log_std, RMS(z))`` using detached gate geometry."""

    actor_output = actor(states)
    if not isinstance(actor_output, (tuple, list)) or len(actor_output) != 2:
        raise CanonicalContractError(
            "signed_td_v_v1 requires actor(state) -> (mean, log_std)"
        )
    mean, log_std = actor_output
    if mean.shape != actions.shape:
        raise CanonicalContractError(
            f"actor mean shape {tuple(mean.shape)} does not match actions "
            f"{tuple(actions.shape)}"
        )
    if log_std.shape != mean.shape:
        try:
            log_std = log_std.expand_as(mean)
        except RuntimeError as exc:
            raise CanonicalContractError(
                f"log_std shape {tuple(log_std.shape)} cannot expand to "
                f"mean shape {tuple(mean.shape)}"
            ) from exc
    safe_log_std = torch.clamp(log_std, min=-20.0, max=5.0)
    with torch.no_grad():
        z = (actions - mean.detach()) / safe_log_std.detach().exp().clamp_min(1e-8)
        distance = z.square().mean(dim=-1).sqrt()
    return mean, log_std, distance


def taper_factor(distance: torch.Tensor, control: NegativeControl) -> torch.Tensor:
    """Compute the detached shape factor for negative samples."""

    if control.method in {"positive_only", "canonical_signed", "global"}:
        return torch.ones_like(distance)
    u = distance / control.reference_distance
    if control.method == "reciprocal_linear":
        return 1.0 / (1.0 + control.reciprocal_linear_coefficient * u)
    if control.method == "reciprocal_quadratic":
        return 1.0 / (
            1.0 + control.reciprocal_quadratic_coefficient * u.square()
        )
    if control.method == "exponential":
        exponent = torch.clamp(
            -control.exponential_coefficient * u,
            min=-40.0,
            max=0.0,
        )
        return torch.exp(exponent)
    raise AssertionError(f"unreachable method: {control.method}")


def controlled_advantage(
    advantage: torch.Tensor,
    distance: torch.Tensor,
    control: NegativeControl,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Leave positive advantages untouched and control only negative ones."""

    negative = advantage < 0
    shape = taper_factor(distance, control)
    if control.method == "positive_only":
        factor = torch.where(negative, torch.zeros_like(shape), torch.ones_like(shape))
    else:
        negative_factor = control.effective_alpha * shape
        factor = torch.where(negative, negative_factor, torch.ones_like(shape))
    return advantage * factor, factor


def _as_tensor(
    value: Any,
    *,
    device: torch.device,
    dtype: torch.dtype | None = torch.float32,
) -> torch.Tensor:
    if torch.is_tensor(value):
        tensor = value.to(device=device)
        if dtype is not None:
            tensor = tensor.to(dtype=dtype)
        return tensor
    return torch.as_tensor(value, device=device, dtype=dtype)


def _agent_device(agent: Any) -> torch.device:
    try:
        return next(agent.actor.parameters()).device
    except (AttributeError, StopIteration) as exc:
        raise CanonicalContractError(
            "target agent must expose actor parameters"
        ) from exc


def validate_agent_instance(
    agent: Any,
    *,
    expected_alpha: float,
) -> None:
    required = ("actor", "critic", "a_opt", "c_opt", "gamma", "tau", "alpha")
    missing = [name for name in required if not hasattr(agent, name)]
    if missing:
        raise CanonicalContractError(
            f"target agent is missing signed_td_v_v1 attributes: {missing}"
        )
    actual_alpha = float(agent.alpha)
    if not math.isclose(actual_alpha, expected_alpha, rel_tol=0.0, abs_tol=1e-12):
        raise CanonicalContractError(
            "canonical alpha changed at construction time: "
            f"expected {expected_alpha}, got {actual_alpha}"
        )


def build_injected_agent_class(
    base_class: type,
    *,
    control: NegativeControl,
    return_mode: str,
    advantage_provider: AdvantageProvider | None = None,
) -> type:
    """Build one canonical update class with an optional actor-advantage provider.

    The actor loss, critic target, expectile loss, optimizer order, and optimizer
    steps remain single-sourced here. A provider may replace only the detached
    actor advantage batch; it cannot replace the critic update.
    """

    if return_mode not in {"zero_float", "metrics_dict"}:
        raise ValueError(f"unsupported return_mode={return_mode!r}")

    class CanonicalNegativeControlAgent(base_class):  # type: ignore[misc, valid-type]
        _drpo_negative_control = control
        _drpo_advantage_provider = advantage_provider

        def update(
            self,
            s: Any,
            a: Any,
            r: Any,
            ns: Any,
            d: Any,
            ep_ret: Any = None,
        ) -> Any:
            validate_agent_instance(self, expected_alpha=control.canonical_alpha)
            device = _agent_device(self)
            states = _as_tensor(s, device=device)
            actions = _as_tensor(a, device=device)
            rewards = _as_tensor(r, device=device).reshape(-1)
            next_states = _as_tensor(ns, device=device)
            dones = _as_tensor(d, device=device, dtype=torch.bool).reshape(-1)

            values = self.critic(states).squeeze(-1)
            with torch.no_grad():
                next_values = self.critic(next_states).squeeze(-1)
                targets = rewards + float(self.gamma) * next_values * (~dones).float()
            default_advantages = targets - values.detach()
            if advantage_provider is None:
                advantages = default_advantages
            else:
                provided = advantage_provider(self, ep_ret, default_advantages)
                advantages = _as_tensor(provided, device=device).reshape(-1)
                if advantages.shape != default_advantages.shape:
                    raise CanonicalContractError(
                        "advantage provider returned a misaligned batch"
                    )
                if not bool(torch.isfinite(advantages).all()):
                    raise FloatingPointError(
                        "advantage provider returned non-finite values"
                    )
                advantages = advantages.detach()

            mean, log_std, distance = detached_standardized_distance(
                self.actor,
                states,
                actions,
            )
            distribution = torch.distributions.Normal(mean, log_std.exp())
            log_prob = distribution.log_prob(actions).sum(dim=-1)
            weighted_advantage, factor = controlled_advantage(
                advantages,
                distance,
                control,
            )

            # Full-batch normalization is deliberate. Positive-only therefore
            # differs from signed training only by zeroing negative terms.
            actor_loss = -(log_prob * weighted_advantage).mean()
            self.a_opt.zero_grad(set_to_none=True)
            actor_loss.backward()
            self.a_opt.step()

            value_error = targets - values
            expectile = torch.where(
                value_error > 0,
                torch.full_like(value_error, float(self.tau)),
                torch.full_like(value_error, 1.0 - float(self.tau)),
            )
            critic_loss = (expectile * value_error.square()).mean()
            self.c_opt.zero_grad(set_to_none=True)
            critic_loss.backward()
            self.c_opt.step()

            negative = advantages < 0
            metrics = {
                "actor_loss": float(actor_loss.detach().cpu()),
                "critic_loss": float(critic_loss.detach().cpu()),
                "positive_fraction": float((~negative).float().mean().cpu()),
                "negative_fraction": float(negative.float().mean().cpu()),
                "negative_factor_mean": (
                    float(factor[negative].mean().detach().cpu())
                    if bool(negative.any())
                    else float("nan")
                ),
                "negative_distance_mean": (
                    float(distance[negative].mean().detach().cpu())
                    if bool(negative.any())
                    else float("nan")
                ),
                "canonical_alpha": control.canonical_alpha,
                "negative_scale": control.negative_scale,
                "effective_alpha": control.effective_alpha,
                "method": control.method,
            }
            if advantage_provider is not None:
                metrics["advantage_estimator"] = str(
                    getattr(advantage_provider, "estimator", "custom")
                )
            self._drpo_last_negative_control_metrics = metrics
            if return_mode == "metrics_dict":
                return metrics
            return 0.0

    CanonicalNegativeControlAgent.__name__ = base_class.__name__
    CanonicalNegativeControlAgent.__qualname__ = base_class.__qualname__
    CanonicalNegativeControlAgent.__module__ = base_class.__module__
    CanonicalNegativeControlAgent.__doc__ = (
        f"Runtime-injected {base_class.__name__} for method={control.method}, "
        f"negative_scale={control.negative_scale}."
    )
    return CanonicalNegativeControlAgent


def load_verified_canonical_module(
    contract: CanonicalContract,
) -> tuple[ModuleType, dict[str, Any]]:
    """Load the exact contracted module under its historical import name."""

    checks = contract.verify_runtime()
    root_text = str(contract.source_root)
    if root_text not in sys.path:
        sys.path.insert(0, root_text)
    existing = sys.modules.get(contract.module_name)
    if existing is not None:
        existing_file = Path(getattr(existing, "__file__", "")).resolve()
        if existing_file != contract.agents_path:
            raise CanonicalContractError(
                f"module {contract.module_name!r} is already loaded from "
                f"{existing_file}, expected {contract.agents_path}"
            )
        module = existing
    else:
        spec = importlib.util.spec_from_file_location(
            contract.module_name,
            contract.agents_path,
        )
        if spec is None or spec.loader is None:
            raise CanonicalContractError(
                f"cannot create import spec for {contract.agents_path}"
            )
        module = importlib.util.module_from_spec(spec)
        sys.modules[contract.module_name] = module
        try:
            spec.loader.exec_module(module)
        except Exception:
            sys.modules.pop(contract.module_name, None)
            raise
    if not hasattr(module, contract.target_class):
        raise CanonicalContractError(
            f"canonical module has no target class {contract.target_class!r}"
        )
    target = getattr(module, contract.target_class)
    if not isinstance(target, type):
        raise CanonicalContractError(
            f"canonical target {contract.target_class!r} is not a class"
        )
    checks["module_name"] = contract.module_name
    checks["target_class"] = contract.target_class
    return module, checks


def patch_canonical_module(
    module: ModuleType,
    contract: CanonicalContract,
    control: NegativeControl,
    *,
    advantage_provider: AdvantageProvider | None = None,
) -> type:
    """Replace only the configured class inside the already verified module."""

    original = getattr(module, contract.target_class)
    injected = build_injected_agent_class(
        original,
        control=control,
        return_mode=contract.return_mode,
        advantage_provider=advantage_provider,
    )
    setattr(module, contract.target_class, injected)
    return injected


def write_fingerprint_contract(
    *,
    canonical_root: str | Path,
    agents_relpath: str,
    trainer_relpath: str,
    module_name: str,
    target_class: str,
    expected_canonical_alpha: float,
    output: str | Path,
    agent_flavor: str = SUPPORTED_AGENT_FLAVOR,
    return_mode: str = "zero_float",
) -> CanonicalContract:
    """Create a reviewable contract from a local canonical source tree."""

    root = Path(canonical_root).expanduser().resolve()
    contract = CanonicalContract(
        contract_version=SUPPORTED_CONTRACT_VERSION,
        canonical_source_root=str(root),
        python_tree_sha256=python_tree_sha256(root),
        agents_relpath=agents_relpath,
        agents_sha256=sha256_file(root / agents_relpath),
        trainer_relpath=trainer_relpath,
        trainer_sha256=sha256_file(root / trainer_relpath),
        module_name=module_name,
        target_class=target_class,
        agent_flavor=agent_flavor,
        expected_canonical_alpha=float(expected_canonical_alpha),
        return_mode=return_mode,
    )
    contract.validate_static()
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(dataclasses.asdict(contract), indent=2) + "\n")
    return contract


def canonical_environment_manifest() -> dict[str, Any]:
    """Small deterministic runtime manifest for branch provenance."""

    return {
        "python_executable": sys.executable,
        "python_version": sys.version,
        "torch_version": torch.__version__,
        "cwd": os.getcwd(),
    }
