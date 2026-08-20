#!/usr/bin/env python3
"""Temporary trusted-PR materializer for the E8 dynamic-slot candidate."""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path

BASE = "4b7c991cd8797412830cb34f76dfe35cc09eb90f"
WORK_BRANCH = "dev/e8-dynamic-slot-repair-work-20260820"
OLD_PATCHER_BLOB = "c009d394717b65e8d9e251885f8f08ada59ea7b6"
TARGETS = (
    "src/drpo/e8_multitask_exp_tuning.py",
    "configs/e8_multitask_exp_coldstart.yaml",
    "tests/test_e8_multitask_p0.py",
)


def run(*args: str, capture: bool = False) -> str:
    proc = subprocess.run(
        args,
        text=True,
        stdout=subprocess.PIPE if capture else None,
        stderr=subprocess.PIPE if capture else None,
        check=False,
    )
    if proc.returncode != 0:
        detail = proc.stderr.strip() if capture and proc.stderr else ""
        raise RuntimeError(f"command failed ({proc.returncode}): {' '.join(args)}\n{detail}")
    return proc.stdout if capture and proc.stdout is not None else ""


def materialize() -> None:
    run("git", "checkout", BASE, "--", *TARGETS)
    run(
        "git",
        "fetch",
        "--no-tags",
        "origin",
        f"{WORK_BRANCH}:refs/remotes/origin/{WORK_BRANCH}",
    )
    workflow = run("git", "show", OLD_PATCHER_BLOB, capture=True)
    start_marker = "          python - <<'PY'\n"
    start = workflow.index(start_marker) + len(start_marker)
    end = workflow.index("          PY\n", start)
    script = "\n".join(
        line[10:] if line.startswith("          ") else line
        for line in workflow[start:end].splitlines()
    ) + "\n"

    bad_dynamic = '''dynamic = "\\n".join(
    line[10:] if line.startswith("          ") else line
    for line in dynamic.splitlines()
)
'''
    bad_audit = '''audit = "\\n".join(
    line[10:] if line.startswith("          ") else line
    for line in audit.splitlines()
)
'''
    bad_scheduler = '''scheduler_test = "\\n".join(
    line[10:] if line.startswith("          ") else line
    for line in scheduler_test.splitlines()
).rstrip() + "\\n"
'''
    for label, old, new in (
        ("dynamic", bad_dynamic, ""),
        ("audit", bad_audit, ""),
        ("scheduler-test", bad_scheduler, 'scheduler_test = scheduler_test.rstrip() + "\\n"\n'),
    ):
        count = script.count(old)
        if count != 1:
            raise RuntimeError(f"{label}: expected one faulty dedent block, found {count}")
        script = script.replace(old, new, 1)

    exec(compile(script, "<corrected-dynamic-slot-patcher>", "exec"), {})


def audit_and_stage_artifact() -> Path:
    import yaml

    changed = set(
        run("git", "diff", "--name-only", BASE, "--", *TARGETS, capture=True).splitlines()
    )
    if changed != set(TARGETS):
        raise RuntimeError(f"unexpected target-file scope: {sorted(changed)}")

    base_cfg = yaml.safe_load(run("git", "show", f"{BASE}:{TARGETS[1]}", capture=True))
    current_cfg = yaml.safe_load(Path(TARGETS[1]).read_text(encoding="utf-8"))
    if base_cfg["execution"]["wave_barriers"] is not True:
        raise RuntimeError("base wave_barriers is not true")
    if current_cfg["execution"]["wave_barriers"] is not False:
        raise RuntimeError("candidate wave_barriers is not false")
    base_cfg["execution"]["wave_barriers"] = False
    if current_cfg != base_cfg:
        raise RuntimeError("config changed outside execution.wave_barriers")

    source = Path(TARGETS[0]).read_text(encoding="utf-8")
    if source.count("def cmd_run_dynamic(") != 1:
        raise RuntimeError("cmd_run_dynamic definition count drifted")
    if source.count("def _audit_engineering_queue(") != 1:
        raise RuntimeError("_audit_engineering_queue definition count drifted")
    start = source.index("def cmd_run_dynamic(")
    end = source.index("\ndef cmd_run_all(", start)
    dynamic = source[start:end]
    required = (
        "pending: queue.Queue[Cell] = queue.Queue()",
        "executor.submit(worker, slot, gpu_ids[slot % len(gpu_ids)])",
        '"wave_barriers": False',
        '"scheduling_barrier": False',
        '"countdown_result_controls_transfer_release": False',
    )
    missing = [needle for needle in required if needle not in dynamic]
    if missing:
        raise RuntimeError(f"dynamic scheduler invariant missing: {missing}")
    if "for wave_index, wave in enumerate(waves, start=1):" in dynamic:
        raise RuntimeError("hard-wave scheduling loop remains")

    run("git", "diff", "--check", BASE, "--", *TARGETS)

    artifact = Path("/tmp/e8-dynamic-slot-materialized")
    shutil.rmtree(artifact, ignore_errors=True)
    artifact.mkdir(parents=True)
    for rel in TARGETS:
        dst = artifact / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(rel, dst)
    (artifact / "BASE_COMMIT.txt").write_text(BASE + "\n", encoding="utf-8")
    patch = run("git", "diff", "--binary", BASE, "--", *TARGETS, capture=True)
    (artifact / "candidate.patch").write_text(patch, encoding="utf-8")
    manifest = {
        "base_commit": BASE,
        "targets": list(TARGETS),
        "sha256": {
            rel: hashlib.sha256(Path(rel).read_bytes()).hexdigest() for rel in TARGETS
        },
    }
    (artifact / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return artifact


def upload_artifact(path: Path) -> None:
    action_root = Path("/tmp/upload-artifact-v4")
    shutil.rmtree(action_root, ignore_errors=True)
    run(
        "git",
        "clone",
        "--quiet",
        "--depth",
        "1",
        "--branch",
        "v4",
        "https://github.com/actions/upload-artifact.git",
        str(action_root),
    )
    env = os.environ.copy()
    env["INPUT_NAME"] = "e8-dynamic-slot-materialized"
    env["INPUT_PATH"] = str(path)
    proc = subprocess.run(
        ["node", str(action_root / "dist" / "index.js")],
        env=env,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"artifact upload failed with exit {proc.returncode}")


def main() -> int:
    materialize()
    artifact = audit_and_stage_artifact()
    upload_artifact(artifact)
    print(
        json.dumps(
            {
                "mode": "temporary_e8_dynamic_slot_materializer",
                "base": BASE,
                "candidate_files": list(TARGETS),
                "artifact": "e8-dynamic-slot-materialized",
                "materialized": True,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
