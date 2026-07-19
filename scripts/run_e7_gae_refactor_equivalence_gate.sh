#!/usr/bin/env bash
set -euo pipefail

OLD_COMMIT="2fe97cdcff0e8361b33193dd2a7be8cf63c44a3b"
ROOT="$(git rev-parse --show-toplevel)"
NEW_COMMIT="$(git -C "$ROOT" rev-parse HEAD)"
OUT_DIR="${1:-$ROOT/artifacts/e7_gae_refactor_equivalence}"
TMP_ROOT="$(mktemp -d "${TMPDIR:-/tmp}/e7-gae-equiv.XXXXXX")"
OLD_TREE="$TMP_ROOT/old"
DRIVER="$TMP_ROOT/driver.py"

cleanup() {
  git -C "$ROOT" worktree remove --force "$OLD_TREE" >/dev/null 2>&1 || true
  rm -rf "$TMP_ROOT"
}
trap cleanup EXIT

mkdir -p "$OUT_DIR"
rm -f "$OUT_DIR/OLD_TRACE.json" "$OUT_DIR/NEW_TRACE.json" \
  "$OUT_DIR/SOURCE_IDENTITY.json" "$OUT_DIR/EQUIVALENCE_AUDIT.json"

if ! git -C "$ROOT" cat-file -e "${OLD_COMMIT}^{commit}" 2>/dev/null; then
  git -C "$ROOT" fetch --no-tags origin "$OLD_COMMIT"
fi
git -C "$ROOT" cat-file -e "${OLD_COMMIT}^{commit}"
git -C "$ROOT" worktree add --detach "$OLD_TREE" "$OLD_COMMIT" >/dev/null

cat >"$DRIVER" <<'PY'
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch


def _digest_value(digest: "hashlib._Hash", value: Any) -> None:
    if torch.is_tensor(value):
        tensor = value.detach().cpu().contiguous()
        digest.update(b"tensor\0")
        digest.update(str(tensor.dtype).encode())
        digest.update(str(tuple(tensor.shape)).encode())
        digest.update(tensor.numpy().tobytes())
    elif isinstance(value, np.ndarray):
        array = np.ascontiguousarray(value)
        digest.update(b"ndarray\0")
        digest.update(str(array.dtype).encode())
        digest.update(str(tuple(array.shape)).encode())
        digest.update(array.tobytes())
    elif isinstance(value, dict):
        digest.update(b"dict\0")
        for key in sorted(value, key=lambda item: repr(item)):
            _digest_value(digest, key)
            _digest_value(digest, value[key])
    elif isinstance(value, (list, tuple)):
        digest.update(type(value).__name__.encode() + b"\0")
        for item in value:
            _digest_value(digest, item)
    elif isinstance(value, bool):
        digest.update(b"bool\0" + (b"1" if value else b"0"))
    elif isinstance(value, int):
        digest.update(b"int\0" + str(value).encode())
    elif isinstance(value, float):
        digest.update(b"float\0" + format(value, ".17g").encode())
    elif value is None:
        digest.update(b"none\0")
    else:
        digest.update(type(value).__name__.encode() + b"\0" + repr(value).encode())


def stable_sha256(value: Any) -> str:
    digest = hashlib.sha256()
    _digest_value(digest, value)
    return digest.hexdigest()


def module_sha256(module: torch.nn.Module) -> str:
    return stable_sha256(dict(module.state_dict()))


def optimizer_sha256(optimizer: torch.optim.Optimizer) -> str:
    return stable_sha256(optimizer.state_dict())


class FixtureActor(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.mean = torch.nn.Linear(2, 1, bias=False)
        self.log_std = torch.nn.Parameter(torch.zeros(1, 1))

    def forward(self, states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mean = self.mean(states)
        return mean, self.log_std.expand_as(mean)


class FixtureCritic(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.value = torch.nn.Linear(2, 1, bias=False)

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        return self.value(states)


class FixtureAgent:
    def __init__(self) -> None:
        self.gamma = 0.9
        self.tau = 0.7
        self.alpha = 0.11
        self.actor = FixtureActor()
        self.critic = FixtureCritic()
        self.a_opt = torch.optim.SGD(self.actor.parameters(), lr=1e-2)
        self.c_opt = torch.optim.SGD(self.critic.parameters(), lr=1e-2)


def replay_arrays() -> dict[str, np.ndarray]:
    observations = np.asarray(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
            [2.0, 1.0],
            [-1.0, 0.5],
            [0.5, -1.0],
            [-0.5, -0.5],
            [1.5, -0.5],
        ],
        dtype=np.float32,
    )
    return {
        "observations": observations,
        "actions": np.asarray(
            [[0.8], [-0.7], [0.2], [1.1], [-0.4], [0.6], [-0.2], [0.9]],
            dtype=np.float32,
        ),
        "rewards": np.asarray(
            [0.2, -0.4, 0.3, -0.8, 0.5, -0.2, 0.1, -0.6],
            dtype=np.float32,
        ),
        "next_observations": observations * np.float32(0.6)
        + np.asarray([0.1, -0.05], dtype=np.float32),
        "terminals": np.asarray(
            [False, True, False, False, False, False, False, False],
            dtype=np.bool_,
        ),
        "timeouts": np.asarray(
            [False, False, True, False, False, True, False, False],
            dtype=np.bool_,
        ),
    }


def set_fixed_state(agent: FixtureAgent) -> None:
    with torch.no_grad():
        agent.actor.mean.weight.copy_(torch.tensor([[0.15, -0.2]], dtype=torch.float32))
        agent.actor.log_std.copy_(torch.tensor([[-0.1]], dtype=torch.float32))
        agent.critic.value.weight.copy_(torch.tensor([[0.25, -0.35]], dtype=torch.float32))


def scalar(value: Any) -> float | str:
    number = float(value)
    if math.isnan(number):
        return "nan"
    if math.isinf(number):
        return "inf" if number > 0 else "-inf"
    return number


def make_control(name: str):
    from drpo.e7_canonical_injection import NegativeControl

    if name == "positive_only":
        return NegativeControl(
            method="positive_only",
            negative_scale=0.0,
            canonical_alpha=0.11,
            reference_distance=2.0,
            exponential_coefficient=0.0,
        )
    if name == "sqexp_c128":
        return NegativeControl(
            method="exponential",
            negative_scale=1.0 / 0.11,
            canonical_alpha=0.11,
            reference_distance=2.0,
            exponential_coefficient=128.0,
        )
    raise ValueError(name)


def build_agent(implementation: str, estimator: str, control_name: str):
    arrays = replay_arrays()
    control = make_control(control_name)
    if implementation == "old":
        from drpo.e7_canonical_gae_injection import (
            OrderedReplay,
            SnapshotEstimatorConfig,
            build_joint_snapshot_agent_class,
        )

        replay = OrderedReplay(
            observations=arrays["observations"],
            actions=arrays["actions"],
            rewards=arrays["rewards"],
            next_observations=arrays["next_observations"],
            terminals=arrays["terminals"],
            timeouts=arrays["timeouts"],
        )
        cls = build_joint_snapshot_agent_class(
            FixtureAgent,
            replay=replay,
            negative_control=control,
            estimator=SnapshotEstimatorConfig(
                estimator=estimator,
                gae_lambda=0.95,
                canonical_batch_size=2,
            ),
            return_mode="metrics_dict",
        )
        agent = cls()

        def table():
            return agent._drpo_advantage_table

        def snapshots():
            return list(agent._drpo_snapshot_hashes)

        def snapshot_count():
            return int(agent._drpo_snapshot_count)

        return arrays, agent, table, snapshots, snapshot_count

    if implementation == "new":
        from drpo.e7_canonical_injection import build_injected_agent_class
        from drpo.e7_squared_exp_night_bootstrap import TrajectorySnapshotAdvantage

        provider = TrajectorySnapshotAdvantage(arrays, estimator, batch_size=2)
        cls = build_injected_agent_class(
            FixtureAgent,
            control=control,
            return_mode="metrics_dict",
            advantage_provider=provider,
        )
        agent = cls()

        def table():
            return provider.table

        def snapshots():
            return list(provider.snapshot_hashes)

        def snapshot_count():
            return len(provider.snapshot_hashes)

        return arrays, agent, table, snapshots, snapshot_count

    raise ValueError(implementation)


def run_case(implementation: str, estimator: str, control_name: str) -> dict[str, Any]:
    arrays, agent, table, snapshots, snapshot_count = build_agent(
        implementation, estimator, control_name
    )
    set_fixed_state(agent)
    sequence = [
        [0, 1],
        [2, 3],
        [4, 5],
        [6, 7],
        [1, 6],
        [0, 7],
        [2, 5],
        [3, 4],
        [0, 4],
    ]
    trace: dict[str, Any] = {
        "case": f"{estimator}__{control_name}",
        "estimator": estimator,
        "control": control_name,
        "refresh_interval": 4,
        "initial": {
            "actor_sha256": module_sha256(agent.actor),
            "critic_sha256": module_sha256(agent.critic),
            "actor_optimizer_sha256": optimizer_sha256(agent.a_opt),
            "critic_optimizer_sha256": optimizer_sha256(agent.c_opt),
        },
        "updates": [],
    }
    for update, indices in enumerate(sequence, start=1):
        ids = np.asarray(indices, dtype=np.int64)
        batch = (
            torch.from_numpy(arrays["observations"][ids]),
            torch.from_numpy(arrays["actions"][ids]),
            torch.from_numpy(arrays["rewards"][ids]),
            torch.from_numpy(arrays["next_observations"][ids]),
            torch.from_numpy(arrays["terminals"][ids]),
            torch.from_numpy(ids.astype(np.float32)),
        )
        metrics = agent.update(*batch)
        active_table = table()
        if active_table is None:
            raise RuntimeError("advantage table was not initialized")
        trace["updates"].append(
            {
                "update": update,
                "transition_ids": indices,
                "actor_sha256": module_sha256(agent.actor),
                "critic_sha256": module_sha256(agent.critic),
                "actor_optimizer_sha256": optimizer_sha256(agent.a_opt),
                "critic_optimizer_sha256": optimizer_sha256(agent.c_opt),
                "advantage_table_sha256": stable_sha256(active_table),
                "snapshot_count": snapshot_count(),
                "snapshot_hashes": snapshots(),
                "metrics": {
                    "actor_loss": scalar(metrics["actor_loss"]),
                    "critic_loss": scalar(metrics["critic_loss"]),
                    "positive_fraction": scalar(metrics["positive_fraction"]),
                    "negative_fraction": scalar(metrics["negative_fraction"]),
                    "negative_factor_mean": scalar(metrics["negative_factor_mean"]),
                },
            }
        )
    return trace


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--implementation", choices=("old", "new"), required=True)
    parser.add_argument("--commit", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(0)
    np.random.seed(0)

    cases = [
        run_case(args.implementation, estimator, control)
        for estimator in ("td", "gae")
        for control in ("positive_only", "sqexp_c128")
    ]
    payload = {
        "gate_id": "EXT-H-E7-SQEXP-GAE-REFACTOR-EQUIVALENCE-01",
        "implementation": args.implementation,
        "commit": args.commit,
        "torch_version": torch.__version__,
        "numpy_version": np.__version__,
        "case_count": len(cases),
        "updates_per_case": 9,
        "cases": cases,
    }
    Path(args.output).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
PY

export PYTHONHASHSEED=0
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

PYTHONPATH="$OLD_TREE/src" python "$DRIVER" \
  --implementation old \
  --commit "$OLD_COMMIT" \
  --output "$OUT_DIR/OLD_TRACE.json"

PYTHONPATH="$ROOT/src" python "$DRIVER" \
  --implementation new \
  --commit "$NEW_COMMIT" \
  --output "$OUT_DIR/NEW_TRACE.json"

OLD_COMMIT="$OLD_COMMIT" NEW_COMMIT="$NEW_COMMIT" ROOT="$ROOT" OLD_TREE="$OLD_TREE" \
OUT_DIR="$OUT_DIR" python - <<'PY'
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any

out = Path(os.environ["OUT_DIR"])
old = json.loads((out / "OLD_TRACE.json").read_text())
new = json.loads((out / "NEW_TRACE.json").read_text())


def file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


identity = {
    "gate_id": "EXT-H-E7-SQEXP-GAE-REFACTOR-EQUIVALENCE-01",
    "old_commit": os.environ["OLD_COMMIT"],
    "new_commit": os.environ["NEW_COMMIT"],
    "old_files": {
        "src/drpo/e7_canonical_gae_injection.py": file_sha(
            Path(os.environ["OLD_TREE"]) / "src/drpo/e7_canonical_gae_injection.py"
        ),
        "src/drpo/e7_canonical_injection.py": file_sha(
            Path(os.environ["OLD_TREE"]) / "src/drpo/e7_canonical_injection.py"
        ),
    },
    "new_files": {
        "src/drpo/e7_squared_exp_night_bootstrap.py": file_sha(
            Path(os.environ["ROOT"]) / "src/drpo/e7_squared_exp_night_bootstrap.py"
        ),
        "src/drpo/e7_canonical_injection.py": file_sha(
            Path(os.environ["ROOT"]) / "src/drpo/e7_canonical_injection.py"
        ),
    },
}
(out / "SOURCE_IDENTITY.json").write_text(
    json.dumps(identity, indent=2, sort_keys=True) + "\n"
)

failures: list[dict[str, Any]] = []
exact_initial = (
    "actor_sha256",
    "critic_sha256",
    "actor_optimizer_sha256",
    "critic_optimizer_sha256",
)
exact_update = (
    "transition_ids",
    "actor_sha256",
    "critic_sha256",
    "actor_optimizer_sha256",
    "critic_optimizer_sha256",
    "advantage_table_sha256",
    "snapshot_count",
    "snapshot_hashes",
)
metric_fields = (
    "actor_loss",
    "critic_loss",
    "positive_fraction",
    "negative_fraction",
    "negative_factor_mean",
)

old_cases = {case["case"]: case for case in old["cases"]}
new_cases = {case["case"]: case for case in new["cases"]}
if set(old_cases) != set(new_cases):
    failures.append(
        {
            "case": None,
            "update": None,
            "field": "case_set",
            "old": sorted(old_cases),
            "new": sorted(new_cases),
        }
    )

for name in sorted(set(old_cases) & set(new_cases)):
    left, right = old_cases[name], new_cases[name]
    for field in exact_initial:
        if left["initial"][field] != right["initial"][field]:
            failures.append(
                {
                    "case": name,
                    "update": 0,
                    "field": field,
                    "old": left["initial"][field],
                    "new": right["initial"][field],
                }
            )
    if len(left["updates"]) != len(right["updates"]):
        failures.append(
            {
                "case": name,
                "update": None,
                "field": "update_count",
                "old": len(left["updates"]),
                "new": len(right["updates"]),
            }
        )
        continue
    for old_step, new_step in zip(left["updates"], right["updates"], strict=True):
        update = old_step["update"]
        if update != new_step["update"]:
            failures.append(
                {
                    "case": name,
                    "update": update,
                    "field": "update_index",
                    "old": update,
                    "new": new_step["update"],
                }
            )
        for field in exact_update:
            if old_step[field] != new_step[field]:
                failures.append(
                    {
                        "case": name,
                        "update": update,
                        "field": field,
                        "old": old_step[field],
                        "new": new_step[field],
                    }
                )
        for field in metric_fields:
            a = old_step["metrics"][field]
            b = new_step["metrics"][field]
            if isinstance(a, str) or isinstance(b, str):
                equal = a == b
            else:
                equal = math.isclose(float(a), float(b), rel_tol=0.0, abs_tol=1e-12)
            if not equal:
                failures.append(
                    {
                        "case": name,
                        "update": update,
                        "field": f"metrics.{field}",
                        "old": a,
                        "new": b,
                    }
                )

refresh_positions = {}
for name, case in new_cases.items():
    previous = 0
    positions = []
    for step in case["updates"]:
        if step["snapshot_count"] > previous:
            positions.append(step["update"])
            previous = step["snapshot_count"]
    refresh_positions[name] = positions
    if positions != [1, 5, 9]:
        failures.append(
            {
                "case": name,
                "update": None,
                "field": "refresh_positions",
                "old": [1, 5, 9],
                "new": positions,
            }
        )

audit = {
    "gate_id": "EXT-H-E7-SQEXP-GAE-REFACTOR-EQUIVALENCE-01",
    "status": "PASS" if not failures else "FAIL",
    "old_commit": old["commit"],
    "new_commit": new["commit"],
    "case_count": len(new_cases),
    "updates_per_case": new["updates_per_case"],
    "refresh_positions": refresh_positions,
    "exact_state_and_optimizer_hash_match": not any(
        failure["field"]
        in {
            "actor_sha256",
            "critic_sha256",
            "actor_optimizer_sha256",
            "critic_optimizer_sha256",
        }
        for failure in failures
    ),
    "advantage_table_hash_match": not any(
        failure["field"] == "advantage_table_sha256" for failure in failures
    ),
    "snapshot_trajectory_match": not any(
        failure["field"] in {"snapshot_count", "snapshot_hashes", "refresh_positions"}
        for failure in failures
    ),
    "scalar_absolute_tolerance": 1e-12,
    "failure_count": len(failures),
    "first_failure": failures[0] if failures else None,
    "failures": failures,
    "scientific_result": False,
    "formal_evidence_allowed": False,
    "held_out_seeds_touched": False,
    "real_data_layer_executed": False,
}
(out / "EQUIVALENCE_AUDIT.json").write_text(
    json.dumps(audit, indent=2, sort_keys=True) + "\n"
)
print(json.dumps(audit, indent=2, sort_keys=True))
if failures:
    raise SystemExit(1)
PY
