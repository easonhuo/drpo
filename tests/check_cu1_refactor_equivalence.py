#!/usr/bin/env python3
"""Compare the refactored C-U1 core against the exact pre-refactor Git file."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-commit", required=True)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    return parser.parse_args()


def git_show(repo: Path, revision_path: str) -> str:
    result = subprocess.run(
        ["git", "show", revision_path],
        cwd=repo,
        check=True,
        text=True,
        capture_output=True,
    )
    return result.stdout


def fingerprint(script_path: Path, helper_path: Path) -> dict[str, str]:
    result = subprocess.run(
        [sys.executable, str(helper_path), str(script_path)],
        check=True,
        text=True,
        capture_output=True,
    )
    return json.loads(result.stdout)


def main() -> int:
    args = parse_args()
    repo = args.repo_root.resolve()
    current = repo / "src" / "drpo" / "drpo_cu1_e1_e4_oneclick.py"
    if not current.is_file():
        raise SystemExit(f"missing current runner: {current}")

    helper_source = r'''
import hashlib
import importlib.util
import json
import pathlib
import sys
import torch

path = pathlib.Path(sys.argv[1]).resolve()
sys.path.insert(0, str(path.parent))
name = "cu1_equivalence_candidate"
spec = importlib.util.spec_from_file_location(name, path)
module = importlib.util.module_from_spec(spec)
sys.modules[name] = module
spec.loader.exec_module(module)
module.seed_all(123)
environment = module.make_environment(10)
actor = module.GaussianActor()
values = {
    "train_s": environment.train.s,
    "a_plus": environment.train.a_plus,
    "a_star": environment.train.a_star,
    "negative_actions": environment.train.negative_actions,
    "negative_advantages": environment.train.negative_advantages,
}
output = {
    key: hashlib.sha256(value.detach().cpu().numpy().tobytes()).hexdigest()
    for key, value in values.items()
}
for key, value in actor.state_dict().items():
    output["actor:" + key] = hashlib.sha256(
        value.detach().cpu().numpy().tobytes()
    ).hexdigest()
with torch.no_grad():
    mu, log_std = actor(environment.train.s[:7])
    log_prob = module.gaussian_log_prob(
        mu,
        log_std,
        environment.train.negative_actions[:7],
    )
for key, value in {"mu": mu, "log_std": log_std, "log_prob": log_prob}.items():
    output[key] = hashlib.sha256(value.detach().cpu().numpy().tobytes()).hexdigest()
print(json.dumps(output, sort_keys=True))
'''

    with tempfile.TemporaryDirectory(prefix="cu1-equivalence-") as raw:
        temporary = Path(raw)
        old_runner = temporary / "drpo_cu1_e1_e4_oneclick.py"
        old_runner.write_text(
            git_show(
                repo,
                f"{args.base_commit}:src/drpo/drpo_cu1_e1_e4_oneclick.py",
            ),
            encoding="utf-8",
        )
        helper = temporary / "fingerprint.py"
        helper.write_text(helper_source, encoding="utf-8")
        old = fingerprint(old_runner, helper)
        new = fingerprint(current, helper)

    if old != new:
        differing = sorted(key for key in set(old) | set(new) if old.get(key) != new.get(key))
        print(json.dumps({"matched": False, "differing_keys": differing}, indent=2))
        return 1
    print(json.dumps({"matched": True, "fingerprint_entries": len(new)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
