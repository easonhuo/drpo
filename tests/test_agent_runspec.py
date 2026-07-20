from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
import time
import zipfile
from pathlib import Path
from typing import Any

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
VALIDATE = ROOT / "scripts" / "agent" / "validate_runspec.py"
CLAIM = ROOT / "scripts" / "agent" / "claim_next_runspec.py"
RUN_LANE = ROOT / "scripts" / "agent" / "run_lane.py"
PACKAGE = ROOT / "scripts" / "agent" / "package_runspec_artifacts.py"
sys.path.insert(0, str(ROOT / "scripts" / "agent"))
import run_claimed_runspec as claimed_runner  # noqa: E402


def run(cmd: list[str], *, cwd: Path, check: bool = True, env: dict[str, str] | None = None):
    proc = subprocess.run(
        cmd,
        cwd=cwd,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        env=env,
    )
    if check and proc.returncode != 0:
        raise AssertionError(
            f"command failed: {' '.join(cmd)}\nstdout={proc.stdout}\nstderr={proc.stderr}"
        )
    return proc


def _subprocess_env_with_repo_src() -> dict[str, str]:
    """Ensure a spawned RunSpec entrypoint imports ``drpo`` from this checkout.

    The temp demo repo has no ``src/drpo``; production entrypoints import
    ``drpo.runtime_resource_pool`` / ``drpo.runtime_cpu_capacity`` only when a
    CPU pool is declared. On CI the editable install already resolves those to
    this checkout, and prepending ``ROOT/src`` here is the same path. Locally it
    prevents the subprocess from falling back to an unrelated editable install.
    """

    existing = os.environ.get("PYTHONPATH", "")
    prefix = str(ROOT / "src")
    pythonpath = prefix if not existing else f"{prefix}{os.pathsep}{existing}"
    env = dict(os.environ)
    env["PYTHONPATH"] = pythonpath
    return env


def make_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    run(["git", "init", "-q"], cwd=repo)
    run(["git", "config", "user.name", "RunSpec Test"], cwd=repo)
    run(["git", "config", "user.email", "runspec@example.invalid"], cwd=repo)
    (repo / "scripts" / "demo").mkdir(parents=True)
    runner = repo / "scripts" / "demo" / "run_demo.sh"
    runner.write_text(
        textwrap.dedent(
            """
            #!/usr/bin/env bash
            set -euo pipefail
            mkdir -p outputs/e8/demo/logs outputs/e8/demo/metrics
            printf '{"ok": true}\n' > outputs/e8/demo/summary.json
            printf '{"audit": true}\n' > outputs/e8/demo/audit.json
            printf '{"metric": 1}\n' > outputs/e8/demo/metrics/metric.json
            printf '{"max_workers": "%s"}\n' "${DRPO_RUNTIME_MAX_WORKERS:-}" \
              > outputs/e8/demo/runtime_resource.json
            python3 -c "import json,os;aff=sorted(int(x) for x in os.sched_getaffinity(0)) if hasattr(os,'sched_getaffinity') else [];open('outputs/e8/demo/affinity.json','w').write(json.dumps({'affinity':aff}))"
            echo done > outputs/e8/demo/logs/stdout.txt
            echo should-not-package > outputs/e8/demo/model.pt
            """
        ).lstrip()
    )
    runner.chmod(0o755)
    (repo / "experiments").mkdir()
    (repo / "experiments" / "registry.yaml").write_text(
        "schema_version: 2\nexperiments:\n  - experiment_id: EXT-C-E8-DEMO-01\n"
    )
    (repo / "runspecs" / "ready").mkdir(parents=True)
    (repo / ".agent_lane.yaml").write_text(
        "lane: e8\nforbid_cross_lane: true\nallowed_experiment_prefixes:\n  - EXT-C-E8-\n"
    )
    run(["git", "add", "."], cwd=repo)
    run(["git", "commit", "-q", "-m", "base"], cwd=repo)
    commit = run(["git", "rev-parse", "HEAD"], cwd=repo).stdout.strip()
    spec = {
        "version": 1,
        "run_id": "E8_DEMO_20260711",
        "lane": "e8",
        "priority": 10,
        "created_at": "2026-07-11T00:00:00Z",
        "experiment_id": "EXT-C-E8-DEMO-01",
        "repo_commit": commit,
        "entrypoint": {
            "cwd": "repo_root",
            "command": "bash scripts/demo/run_demo.sh",
        },
        "policy": {
            "formal_evidence_allowed": False,
            "existing_script_required": True,
            "forbid_new_launcher": True,
            "forbid_hparam_change": True,
            "forbid_cross_lane": True,
        },
        "outputs": {
            "run_dir": "outputs/e8/demo",
            "summary_file": "outputs/e8/demo/summary.json",
            "audit_file": "outputs/e8/demo/audit.json",
        },
        "success_criteria": [
            "outputs/e8/demo/summary.json exists",
            "outputs/e8/demo/audit.json exists",
        ],
        "artifacts": {
            "package_policy": "manifest_only",
            "include": [
                "outputs/e8/demo/summary.json",
                "outputs/e8/demo/audit.json",
                "outputs/e8/demo/runtime_resource.json",
                "outputs/e8/demo/affinity.json",
                "outputs/e8/demo/metrics/*.json",
                "outputs/e8/demo/logs/*.txt",
            ],
            "exclude": ["**/*.pt", "**/*.safetensors", "**/*.bin"],
            "max_package_size_mb": 10,
            "fail_if_excluded_matched": True,
            "fail_if_package_too_large": True,
        },
    }
    (repo / "runspecs" / "ready" / "E8_DEMO_20260711.yaml").write_text(
        yaml.safe_dump(spec, sort_keys=False)
    )
    return repo


def test_validate_claim_and_run_lane_packages_only_allowed_artifacts(tmp_path: Path):
    repo = make_repo(tmp_path)
    spec = repo / "runspecs" / "ready" / "E8_DEMO_20260711.yaml"

    validated = run(
        ["python", str(VALIDATE), "--repo-root", str(repo), str(spec), "--json"],
        cwd=repo,
    )
    assert json.loads(validated.stdout)["status"] == "PASS"

    claimed = run(["python", str(CLAIM), "--repo-root", str(repo), "--json"], cwd=repo)
    claimed_payload = json.loads(claimed.stdout)
    assert claimed_payload["status"] == "PASS"
    assert claimed_payload["claimed_path"] == ".runspec_state/claimed/E8_DEMO_20260711.yaml"
    assert (repo / "runspecs" / "ready" / "E8_DEMO_20260711.yaml").is_file()

    executed = run(
        [
            "python",
            str(RUN_LANE),
            "--repo-root",
            str(repo),
            "--run-id",
            "E8_DEMO_20260711",
            "--json",
        ],
        cwd=repo,
        check=False,
    )
    # run_lane claims; this run_id is already claimed, so it should fail instead of double-running.
    assert executed.returncode != 0

    run_claimed = run(
        [
            "python",
            str(ROOT / "scripts" / "agent" / "run_claimed_runspec.py"),
            "--repo-root",
            str(repo),
            "--run-id",
            "E8_DEMO_20260711",
            "--max-workers",
            "7",
            "--json",
        ],
        cwd=repo,
    )
    payload = json.loads(run_claimed.stdout)
    assert payload["status"] == "PASS"
    assert payload["runtime_resources"]["max_workers"] == 7
    resource_output = json.loads(
        (repo / "outputs/e8/demo/runtime_resource.json").read_text()
    )
    assert resource_output == {"max_workers": "7"}
    zip_path = repo / payload["artifact_zip"]
    assert zip_path.is_file()
    with zipfile.ZipFile(zip_path) as archive:
        names = set(archive.namelist())
    assert "outputs/e8/demo/summary.json" in names
    assert "outputs/e8/demo/audit.json" in names
    assert "outputs/e8/demo/runtime_resource.json" in names
    assert "outputs/e8/demo/model.pt" not in names
    manifest = json.loads(
        (repo / "runspec_artifacts" / "E8_DEMO_20260711_manifest.json").read_text()
    )
    assert all(not row["path"].endswith("model.pt") for row in manifest["included"])


def test_runtime_resource_request_is_opt_in_and_fail_closed():
    assert claimed_runner.normalize_runtime_resource_request(None) is None
    assert (
        claimed_runner.normalize_runtime_resource_request(
            {"cpu_pool": None, "max_workers": None}
        )
        is None
    )
    with pytest.raises(claimed_runner.RunSpecError, match="requires cpu_pool"):
        claimed_runner.normalize_runtime_resource_request(
            {
                "cpu_pool": None,
                "minimum_available_cpu_cores": 1,
                "max_workers": None,
            }
        )
    request = claimed_runner.normalize_runtime_resource_request(
        {
            "cpu_pool": "0-3",
            "cpu_fraction": 0.8,
            "minimum_available_cpu_cores": 2,
            "wait_timeout_seconds": 0,
            "poll_seconds": 5,
            "sample_seconds": 0.25,
            "max_workers": 3,
        }
    )
    assert request is not None
    assert request["cpu_pool"] == "0-3"
    assert request["max_workers"] == 3
    assert request["scientific_matrix_changed"] is False
    with pytest.raises(claimed_runner.RunSpecError, match="cpu_fraction"):
        claimed_runner.normalize_runtime_resource_request(
            {"cpu_pool": "0", "cpu_fraction": 1.1}
        )
    with pytest.raises(claimed_runner.RunSpecError, match="max_workers"):
        claimed_runner.normalize_runtime_resource_request(
            {"cpu_pool": "0", "max_workers": 0}
        )


def test_prepare_runtime_resources_binds_pool_wait_and_worker_ceiling(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    from drpo import runtime_resource_pool

    class FakePool:
        def as_dict(self) -> dict[str, Any]:
            return {
                "schema_version": 1,
                "effective_cpu_ids": [2, 3],
                "cpu_count": 2,
                "pool_digest": "fake-digest",
            }

    def fake_activate_resource_pool(**kwargs: Any) -> FakePool:
        assert kwargs == {
            "cpu_pool": "2-3",
            "gpu_pool": None,
            "gpu_enforcement": "none",
        }
        return FakePool()

    def fake_write_pool_identity(path: Path, _pool: FakePool) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('{"pool_digest": "fake-digest"}\n')
        return path

    monkeypatch.setattr(
        runtime_resource_pool,
        "activate_resource_pool",
        fake_activate_resource_pool,
    )
    monkeypatch.setattr(
        runtime_resource_pool,
        "write_pool_identity",
        fake_write_pool_identity,
    )
    monkeypatch.setattr(
        claimed_runner,
        "_wait_for_cpu_pool_capacity",
        lambda **_kwargs: {
            "cpu_capacity": {"available_cpu_cores": 1.5},
            "wait": {"status": "PLANNED", "attempt_count": 2},
        },
    )

    repo = tmp_path / "repo"
    repo.mkdir()
    request = {
        "cpu_pool": "2-3",
        "cpu_fraction": 0.85,
        "minimum_available_cpu_cores": 1,
        "wait_timeout_seconds": -1,
        "poll_seconds": 300,
        "sample_seconds": 1,
        "max_workers": 4,
    }
    report = claimed_runner.prepare_runtime_resources(
        repo,
        run_id="E7_RESOURCE_TEST",
        request=request,
    )
    assert report is not None
    assert report["resource_pool"]["effective_cpu_ids"] == [2, 3]
    assert report["capacity"]["wait"]["attempt_count"] == 2
    assert os.environ["DRPO_RUNTIME_MAX_WORKERS"] == "4"
    assert Path(report["pool_identity_path"]).is_file()
    assert Path(report["path"]).is_file()

    repeated = claimed_runner.prepare_runtime_resources(
        repo,
        run_id="E7_RESOURCE_TEST",
        request=request,
    )
    assert repeated is not None
    with pytest.raises(claimed_runner.RunSpecError, match="identity changed"):
        claimed_runner.prepare_runtime_resources(
            repo,
            run_id="E7_RESOURCE_TEST",
            request={**request, "max_workers": 5},
        )


def test_cross_lane_is_rejected(tmp_path: Path):
    repo = make_repo(tmp_path)
    (repo / ".agent_lane.yaml").write_text("lane: e1\nforbid_cross_lane: true\n")
    spec = repo / "runspecs" / "ready" / "E8_DEMO_20260711.yaml"
    proc = run(
        ["python", str(VALIDATE), "--repo-root", str(repo), str(spec), "--json"],
        cwd=repo,
        check=False,
    )
    assert proc.returncode != 0
    assert "does not match workspace lane" in proc.stdout


def test_artifact_packaging_rejects_model_like_include(tmp_path: Path):
    repo = make_repo(tmp_path)
    run(["bash", "scripts/demo/run_demo.sh"], cwd=repo)
    spec_path = repo / "runspecs" / "ready" / "E8_DEMO_20260711.yaml"
    spec = yaml.safe_load(spec_path.read_text())
    spec["artifacts"]["include"].append("outputs/e8/demo/model.pt")
    spec_path.write_text(yaml.safe_dump(spec, sort_keys=False))
    proc = run(
        [
            "python",
            str(PACKAGE),
            "--repo-root",
            str(repo),
            "--runspec",
            str(spec_path),
            "--json",
        ],
        cwd=repo,
        check=False,
    )
    assert proc.returncode != 0
    assert "excluded/model-like" in proc.stdout


def _linux_sched_affinity() -> bool:
    return hasattr(os, "sched_getaffinity") and hasattr(os, "sched_setaffinity")


def _dynamic_cpu_pool() -> list[int]:
    """Pick at most two real CPU ids from the current affinity.

    First and last are chosen on purpose: the selection is dynamic, never
    hardcoded to 0-1, and intentionally non-contiguous so the tests never
    assume a contiguous CPU block.
    """

    affinity = sorted(int(cpu) for cpu in os.sched_getaffinity(0))
    assert affinity, "current CPU affinity is empty"
    if len(affinity) == 1:
        return [affinity[0]]
    return [affinity[0], affinity[-1]]


_RESOURCE_RUN_LANE_ARGS = [
    "--resource-cpu-fraction",
    "1.0",
    "--resource-wait-timeout-seconds",
    "30",
    "--resource-poll-seconds",
    "0.1",
    "--resource-sample-seconds",
    "0.05",
    "--max-workers",
    "3",
]


@pytest.mark.skipif(
    not _linux_sched_affinity(),
    reason="real CPU affinity binding requires Linux sched_getaffinity/sched_setaffinity",
)
def test_run_lane_real_affinity_pool_subprocess_inherits_and_capacity_admits(
    tmp_path: Path,
):
    repo = make_repo(tmp_path)
    pool = _dynamic_cpu_pool()
    pool_str = ",".join(str(cpu) for cpu in pool)

    proc = run(
        [
            "python",
            str(RUN_LANE),
            "--repo-root",
            str(repo),
            "--once",
            "--cpu-pool",
            pool_str,
            "--minimum-available-cpu-cores",
            "0.1",
            *_RESOURCE_RUN_LANE_ARGS,
            "--json",
        ],
        cwd=repo,
        env=_subprocess_env_with_repo_src(),
    )
    payload = json.loads(proc.stdout)
    assert payload["status"] == "PASS"
    assert payload["run_id"] == "E8_DEMO_20260711"

    log_root = repo / ".runspec_state" / "logs" / "E8_DEMO_20260711"

    # The demo subprocess inherited the bound CPU pool exactly.
    affinity_record = json.loads((repo / "outputs/e8/demo/affinity.json").read_text())
    assert affinity_record["affinity"] == pool

    # The immutable pool identity records the effective pool.
    pool_identity = json.loads((log_root / "RESOURCE_POOL.json").read_text())
    assert pool_identity["effective_cpu_ids"] == pool

    # The capacity observation was scoped to the pool only.
    capacity = json.loads((log_root / "RUNTIME_CPU_CAPACITY_LATEST.json").read_text())
    assert capacity["affinity_cpu_ids"] == pool

    # The foreground capacity wait reached the success terminal state.
    wait = json.loads((log_root / "RUNTIME_PLAN_CAPACITY_WAIT.json").read_text())
    assert wait["status"] == "PLANNED"

    # The readiness marker exists and exports the worker ceiling.
    ready = json.loads((log_root / "RUNTIME_RESOURCE_READY.json").read_text())
    assert ready["max_workers"] == 3
    assert ready["worker_environment"] == {"DRPO_RUNTIME_MAX_WORKERS": "3"}

    # The worker ceiling was inherited by the demo subprocess.
    resource_output = json.loads(
        (repo / "outputs/e8/demo/runtime_resource.json").read_text()
    )
    assert resource_output == {"max_workers": "3"}

    # The RunSpec entered the done state and packaged the demo artifact.
    assert (repo / ".runspec_state" / "done" / "E8_DEMO_20260711.yaml").is_file()
    zip_path = repo / payload["artifact_zip"]
    assert zip_path.is_file()
    with zipfile.ZipFile(zip_path) as archive:
        names = set(archive.namelist())
    assert "outputs/e8/demo/affinity.json" in names
    assert "outputs/e8/demo/runtime_resource.json" in names


@pytest.mark.skipif(
    not _linux_sched_affinity(),
    reason="real CPU affinity binding requires Linux sched_getaffinity/sched_setaffinity",
)
def test_run_lane_impossible_capacity_floor_fails_before_entrypoint(
    tmp_path: Path,
):
    repo = make_repo(tmp_path)
    pool = _dynamic_cpu_pool()
    pool_str = ",".join(str(cpu) for cpu in pool)

    started = time.monotonic()
    proc = run(
        [
            "python",
            str(RUN_LANE),
            "--repo-root",
            str(repo),
            "--once",
            "--cpu-pool",
            pool_str,
            "--minimum-available-cpu-cores",
            "100",
            *_RESOURCE_RUN_LANE_ARGS,
            "--json",
        ],
        cwd=repo,
        check=False,
        env=_subprocess_env_with_repo_src(),
    )
    elapsed = time.monotonic() - started

    # Non-zero exit; the synchronous policy-budget check rejects before any wait.
    assert proc.returncode != 0
    assert elapsed < 15.0
    combined = proc.stdout + proc.stderr
    assert "exceeds the selected pool/cgroup policy budget" in combined

    # The demo entrypoint never ran, so its marker files are absent.
    assert not (repo / "outputs/e8/demo/summary.json").exists()
    assert not (repo / "outputs/e8/demo/affinity.json").exists()

    # The RunSpec entered the failed state.
    assert (repo / ".runspec_state" / "failed" / "E8_DEMO_20260711.yaml").is_file()
