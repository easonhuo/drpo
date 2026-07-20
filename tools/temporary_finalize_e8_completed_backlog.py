from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml

REPO = "easonhuo/drpo"
BRANCH = "dev/e8-completed-backlog-closure-01"
PR_NUMBER = 195
UPDATE_ID = "EXT-C-E8-COMPLETED-BACKLOG-CLOSURE-2026-07-20"
DELTA_REL = Path("docs/handoff_deltas") / UPDATE_ID / "HANDOFF_DELTA.yaml"
INITIAL_BRANCH_BASE = "4b718e7439cf78a04f4affa1987ac15582d702d1"
RECIPROCAL_IDS = {
    "EXT-C-E8-ORACLE-OFFLINE-V2-PAPER-ALIGNED-RECIPROCAL-SHAPE-SCREEN-0.5B-01",
    "EXT-C-E8-ORACLE-OFFLINE-V2-PAPER-ALIGNED-RECIPROCAL-HIGH-LAMBDA-EXTENSION-0.5B-01",
    "EXT-C-E8-ORACLE-OFFLINE-V2-PAPER-ALIGNED-RECIPROCAL-QUADRATIC-DENSE-LAMBDA-CURVE-0.5B-01",
}
SOURCE_ROOT = Path("/tmp/e8-backlog-source")
TARGET_ROOT = Path("/tmp/e8-backlog-target")
TRUSTED_ROOT = Path("/tmp/e8-backlog-trusted")


def run(
    args: list[str],
    *,
    cwd: Path | None = None,
    capture: bool = False,
    check: bool = True,
    env: dict[str, str] | None = None,
    input_bytes: bytes | None = None,
) -> subprocess.CompletedProcess[str] | subprocess.CompletedProcess[bytes]:
    kwargs: dict[str, Any] = {
        "cwd": str(cwd) if cwd else None,
        "check": check,
        "env": env,
        "input": input_bytes,
    }
    if capture:
        kwargs.update({"stdout": subprocess.PIPE, "stderr": subprocess.STDOUT, "text": input_bytes is None})
    return subprocess.run(args, **kwargs)


def output(args: list[str], *, cwd: Path | None = None) -> str:
    return subprocess.check_output(args, cwd=str(cwd) if cwd else None, text=True).strip()


def git_show(commit: str, rel: str) -> bytes:
    return subprocess.check_output(["git", "show", f"{commit}:{rel}"])


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def copy_prefix(trigger: str, prefix: str, destination_root: Path) -> list[str]:
    rows = output(["git", "ls-tree", "-r", "--name-only", trigger, "--", prefix]).splitlines()
    rows = [row for row in rows if row == prefix or row.startswith(prefix + "/")]
    if not rows:
        raise SystemExit(f"No evidence files found at {trigger}:{prefix}")
    for rel in rows:
        path = destination_root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(git_show(trigger, rel))
    return rows


def claim_for(experiment_id: str) -> str:
    if "PAPER-ALIGNED-LINEAR-SCAN" in experiment_id:
        return (
            "Under the frozen paper-aligned linear current-surprisal coordinate, the tested two-seed "
            "coefficient response contains a near-tied useful region, while the descending right branch "
            "remains unlocalized; this is external-validity pilot evidence only."
        )
    if "PAPER-ALIGNED-TAU-CURVE" in experiment_id:
        return (
            "Under the frozen paper-aligned current-surprisal coordinate, the single-seed tau response "
            "surface provides localization evidence only and does not identify a statistically supported optimum."
        )
    return (
        "Across the frozen historical continuous-EXP validation-only screen, uncontrolled Global negative "
        "pressure is harmful and sufficient tapering opens a broad usable coefficient region; evaluator RNG "
        "contamination and seed-by-hyperparameter variation prohibit a sharp coefficient ranking."
    )


def entry_name(experiment_id: str) -> str:
    suffix = experiment_id.removeprefix("EXT-C-E8-ORACLE-OFFLINE-V2-").removesuffix("-0.5B-01")
    return "countdown_e8_" + suffix.lower().replace("-", "_").replace(".", "p")


def build_entry(record: dict[str, Any]) -> dict[str, Any]:
    experiment_id = record["experiment_id"]
    delivered = "run_id" in record
    entry: dict[str, Any] = {
        "id": experiment_id,
        "environment": "EXT-C",
        "name": entry_name(experiment_id),
        "status": "pilot",
        "scientific_status": "pilot",
        "role": "Countdown_external_validity_hyperparameter_localization",
        "execution_class": "pilot",
        "registration_state": "registered_after_completed_backlog_closure",
        "claim": claim_for(experiment_id),
        "model": {
            "identity": "Qwen2.5-0.5B-Instruct",
            "initialization": "pretrained_base_plus_fresh_lora",
            "parameterization": "lora",
        },
        "protocol": {
            "fixed_training_steps": 1200,
            "early_stop": False,
            "validation_only": True,
            "test_data_used": False,
            "expected_cells": record["cells"],
            "completed_cells": record["cells"],
        },
        "execution": {
            "state": "delivered" if delivered else "completed_external_package_audited",
            "run_id": record.get("run_id"),
            "terminal_audit_status": record["terminal_audit"],
            "numerical_failures": record["nan_inf_numerical_failures"],
        },
        "provenance": {
            "run_source_commit": record["source_commit"],
            "run_source_commit_resolved": record.get("source_commit_resolved", record.get("source_commit_state") == "clean_run_commit"),
            "source_artifact_zip_sha256": record["source_artifact_zip_sha256"],
            "closure_document": "docs/experiments/E8_COMPLETED_BACKLOG_CLOSURE.md",
            "provenance_document": "docs/experiments/E8_COMPLETED_BACKLOG_PROVENANCE.json",
        },
        "evidence": {
            "terminal_audited": True,
            "all_expected_cells_present": True,
            "source_artifact_zip_sha256": record["source_artifact_zip_sha256"],
            "nan_inf_numerical_events": record["nan_inf_numerical_failures"],
            "test_split_used": record["test_split_used"],
            "compact_closure": "experiments/results/e8_completed_backlog_closure_20260720/RESULT_CLOSURE.json",
        },
        "reporting_separation": [
            "task_performance",
            "support_or_structure_valid_rate_diagnostic",
            "nan_inf_numerical_failure",
        ],
        "paper_use": {
            "allowed": [
                "Countdown_external_validity_coefficient_localization",
                "broad_usable_taper_region_interpretation",
            ],
            "prohibited": [
                "formal_method_ranking",
                "statistical_significance",
                "convergence_or_steady_state",
                "OOD_generalization",
                "universal_exponential_superiority",
                "cross_task_or_cross_model_generalization",
            ],
        },
        "closure": {
            "state": "closed_pilot_evidence_line",
            "backlog_registration_closed": True,
            "closure_update_id": UPDATE_ID,
        },
    }
    if delivered:
        entry["evidence"]["results_repository"] = record["results_repository"]
        entry["evidence"]["results_commit"] = record["results_commit"]
        entry["evidence"]["result_manifest_sha256"] = record["manifest_sha256"]
        entry["evidence_locator"] = {
            "schema_version": 1,
            "primary_run_id": record["run_id"],
            "records": [
                {
                    "run_id": record["run_id"],
                    "lane": "e8",
                    "source_commit": record["source_commit"],
                    "results_repository": record["results_repository"],
                    "results_branch": record["results_branch"],
                    "results_commit": record["results_commit"],
                    "result_path": record["result_path"],
                    "manifest_sha256": record["manifest_sha256"],
                    "export_profile": record["export_profile"],
                }
            ],
        }
    return entry


def failures(path: Path) -> set[str]:
    rows: set[str] = set()
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        match = re.match(r"(?:FAILED|ERROR)\s+(\S+)", line)
        if match:
            rows.add(match.group(1))
    return rows


def main() -> None:
    trigger = output(["git", "rev-parse", "HEAD"])
    current_main = output(["git", "ls-remote", "origin", "refs/heads/main"]).split()[0]
    merge_base = output(["git", "merge-base", current_main, trigger])
    if merge_base != INITIAL_BRANCH_BASE:
        raise SystemExit(f"Unexpected backlog branch merge base: {merge_base}")

    for path in (SOURCE_ROOT, TARGET_ROOT, TRUSTED_ROOT):
        if path.exists():
            shutil.rmtree(path)
        run(["git", "worktree", "add", "--detach", str(path), current_main])

    main_registry = yaml.safe_load((SOURCE_ROOT / "experiments/registry.yaml").read_text(encoding="utf-8"))
    main_ids = {item["id"] for item in main_registry["experiments"]}
    missing_reciprocal = sorted(RECIPROCAL_IDS - main_ids)
    if missing_reciprocal:
        raise SystemExit(f"PR #174 must be merged before backlog finalization: {missing_reciprocal}")

    evidence_files: list[str] = []
    for prefix in (
        "docs/experiments/E8_COMPLETED_BACKLOG_CLOSURE.md",
        "docs/experiments/E8_COMPLETED_BACKLOG_PROVENANCE.json",
        "experiments/results/e8_completed_backlog_closure_20260720",
        "experiments/results/e8_paper_aligned_linear_scan_round1_pilot",
        "experiments/results/e8_paper_aligned_tau_curve_pilot",
    ):
        evidence_files.extend(copy_prefix(trigger, prefix, SOURCE_ROOT))

    provenance = json.loads(
        (SOURCE_ROOT / "docs/experiments/E8_COMPLETED_BACKLOG_PROVENANCE.json").read_text(encoding="utf-8")
    )
    if provenance["experiment_count"] != 6 or provenance["completed_cells"] != 222:
        raise SystemExit("Unexpected six-pilot evidence summary")
    entries = [build_entry(record) for record in provenance["records"]]
    ids = [entry["id"] for entry in entries]

    registry_path = SOURCE_ROOT / "experiments/registry.yaml"
    registry_text = registry_path.read_text(encoding="utf-8")
    registry_payload = yaml.safe_load(registry_text)
    existing_ids = {entry["id"] for entry in registry_payload["experiments"]}
    overlap = sorted(existing_ids & set(ids))
    if overlap:
        raise SystemExit(f"Current main unexpectedly already contains backlog closure IDs: {overlap}")
    marker = "\ndevelopment_experiment_registrations:\n"
    if marker not in registry_text:
        raise SystemExit("Registry insertion marker missing")
    appended = yaml.safe_dump(entries, sort_keys=False, allow_unicode=True, width=120)
    registry_after = registry_text.replace(
        marker, "\n" + appended + "development_experiment_registrations:\n", 1
    )
    parsed_after = yaml.safe_load(registry_after)
    if not set(ids).issubset({entry["id"] for entry in parsed_after["experiments"]}):
        raise SystemExit("Registry insertion verification failed")
    registry_path.write_text(registry_after, encoding="utf-8")

    formal_test_path = SOURCE_ROOT / "tests/test_formal_execution_channel.py"
    formal_test_text = formal_test_path.read_text(encoding="utf-8")
    expected_pilot_count = sum(
        entry.get("execution_class") == "pilot" for entry in parsed_after["experiments"]
    )
    matches = list(re.finditer(r'(?m)^(\s*)"pilot":\s*(\d+),\s*$', formal_test_text))
    if len(matches) != 1:
        raise SystemExit(f"Expected one pilot-count snapshot, found {len(matches)}")
    match = matches[0]
    replacement = f'{match.group(1)}"pilot": {expected_pilot_count},'
    formal_test_path.write_text(
        formal_test_text[: match.start()] + replacement + formal_test_text[match.end() :],
        encoding="utf-8",
    )

    base_handoff = subprocess.check_output(
        ["git", "-C", str(SOURCE_ROOT), "show", f"{current_main}:docs/handoff.md"], text=True
    )
    base_registry = subprocess.check_output(
        ["git", "-C", str(SOURCE_ROOT), "show", f"{current_main}:experiments/registry.yaml"], text=True
    )
    content = (
        "- **Countdown E8 completed-result backlog closure:** six validation-only external-validity pilots are now "
        "authoritatively registered: the 62-cell continuous alpha-by-c grid, three 32-cell alpha-one coefficient "
        "extensions, the 32-cell paper-aligned Linear scan, and the 32-cell Tau response surface. The combined "
        "record is 222/222 cells, every terminal audit passed, NaN/Inf failures were 0, and the test split was not used.\n\n"
        "  The locked interpretation is deliberately qualitative. Historical continuous-EXP evidence supports that "
        "uncontrolled Global negative pressure is harmful and sufficient tapering opens a broad usable coefficient "
        "region; historical evaluator RNG contamination and seed-by-hyperparameter variation prohibit a sharp ranking. "
        "The Linear scan did not localize its descending right branch, and the Tau surface is single-seed localization "
        "evidence only. No significance, convergence, steady state, formal ranking, OOD claim, cross-task/model "
        "generalization, or universal exponential-superiority claim is authorized.\n\n"
        "  The first four predate automatic results-repository delivery and remain bound by audited source commits and "
        "source-package SHA-256 values. Linear and Tau are additionally bound to immutable `drpo-results` manifests. "
        "Durable records are `docs/experiments/E8_COMPLETED_BACKLOG_CLOSURE.md` and "
        "`docs/experiments/E8_COMPLETED_BACKLOG_PROVENANCE.json`."
    )
    operations = [
        {
            "operation_id": "append-e8-completed-backlog-closure",
            "op": "append_to_section",
            "heading_path": [
                "0. 研究与执行原则（每次新会话首先阅读）",
                "0.1 当前执行门禁",
            ],
            "block_id": "e8-completed-backlog-closure-2026-07-20",
            "content": content,
        }
    ]
    sys.path.insert(0, str((SOURCE_ROOT / "scripts").resolve()))
    import handoff_delta_shadow as shadow

    candidate = shadow.render(base_handoff, operations).text
    changes = []
    for experiment_id in ids:
        record_evidence = [
            "experiments/registry.yaml",
            "docs/experiments/E8_COMPLETED_BACKLOG_CLOSURE.md",
            "docs/experiments/E8_COMPLETED_BACKLOG_PROVENANCE.json",
            "experiments/results/e8_completed_backlog_closure_20260720/RESULT_CLOSURE.json",
            DELTA_REL.as_posix(),
        ]
        if "PAPER-ALIGNED-LINEAR-SCAN" in experiment_id:
            record_evidence.append("experiments/results/e8_paper_aligned_linear_scan_round1_pilot/RESULT_SUMMARY.json")
        if "PAPER-ALIGNED-TAU-CURVE" in experiment_id:
            record_evidence.append("experiments/results/e8_paper_aligned_tau_curve_pilot/RESULT_SUMMARY.json")
        changes.append(
            {
                "change_id": "add-" + experiment_id.lower().replace(".", "-")[:80],
                "kind": "add_entity",
                "entity_id": experiment_id,
                "evidence": record_evidence,
            }
        )
    delta = {
        "schema_version": 3,
        "update_id": UPDATE_ID,
        "mode": "authoritative",
        "base": {
            "commit": current_main,
            "handoff_sha256": sha256_text(base_handoff),
            "registry_sha256": sha256_text(base_registry),
        },
        "renderer_version": 1,
        "operations": operations,
        "registry": {
            "mode": "expected_after",
            "exact_base_after_sha256": sha256_text(registry_after),
            "changes": changes,
        },
        "expected": {"exact_base_candidate_sha256": sha256_text(candidate)},
    }
    delta_path = SOURCE_ROOT / DELTA_REL
    delta_path.parent.mkdir(parents=True, exist_ok=True)
    delta_path.write_text(
        yaml.safe_dump(delta, sort_keys=False, allow_unicode=True, width=120), encoding="utf-8"
    )

    run(["git", "config", "user.name", "github-actions[bot]"], cwd=SOURCE_ROOT)
    run(
        ["git", "config", "user.email", "41898282+github-actions[bot]@users.noreply.github.com"],
        cwd=SOURCE_ROOT,
    )
    run(["git", "add", "-A"], cwd=SOURCE_ROOT)
    run(["git", "commit", "-m", "Prepare E8 completed backlog closure transaction"], cwd=SOURCE_ROOT)
    source_commit = output(["git", "rev-parse", "HEAD"], cwd=SOURCE_ROOT)
    changed = output(["git", "diff", "--name-only", current_main, source_commit], cwd=SOURCE_ROOT).splitlines()
    if "docs/handoff.md" in changed or any(row.startswith("docs/handoff_shadow/") for row in changed):
        raise SystemExit("Source transaction modified protected handoff after-image before normalization")
    if any(row.startswith(".github/workflows/") or row.startswith("tools/temporary_") for row in changed):
        raise SystemExit("Source transaction contains temporary execution files")
    run(["git", "diff", "--check", current_main, source_commit], cwd=SOURCE_ROOT)
    run(
        [
            sys.executable,
            str(TRUSTED_ROOT / "scripts/handoff_authority.py"),
            "validate-delta",
            "--repo-root",
            str(SOURCE_ROOT),
            "--delta",
            str(SOURCE_ROOT / DELTA_REL),
            "--source-patch-commit",
            source_commit,
            "--json",
        ]
    )

    patch = subprocess.check_output(["git", "diff", "--binary", current_main, source_commit], cwd=SOURCE_ROOT)
    run(["git", "apply", "--index"], cwd=TARGET_ROOT, input_bytes=patch)
    run(
        [
            sys.executable,
            str(TRUSTED_ROOT / "scripts/handoff_authority.py"),
            "normalize",
            "--repo-root",
            str(TARGET_ROOT),
            "--trusted-repo-root",
            str(TRUSTED_ROOT),
            "--current-before",
            current_main,
            "--source-base",
            current_main,
            "--source-patch-commit",
            source_commit,
            "--json",
        ]
    )
    run(["git", "config", "user.name", "github-actions[bot]"], cwd=TARGET_ROOT)
    run(
        ["git", "config", "user.email", "41898282+github-actions[bot]@users.noreply.github.com"],
        cwd=TARGET_ROOT,
    )
    run(["git", "add", "-A"], cwd=TARGET_ROOT)
    run(["git", "commit", "-m", "Materialize E8 completed backlog pilot closure"], cwd=TARGET_ROOT)
    target_commit = output(["git", "rev-parse", "HEAD"], cwd=TARGET_ROOT)

    target_env = os.environ.copy()
    target_env["PYTHONPATH"] = f"{TARGET_ROOT / 'src'}:{TARGET_ROOT / 'scripts'}"
    run(
        [
            sys.executable,
            "scripts/validate_evidence_locator.py",
            "--repo-root",
            ".",
            "--base",
            current_main,
            "--head",
            target_commit,
            "--json",
        ],
        cwd=TARGET_ROOT,
        env=target_env,
    )
    run([sys.executable, "-m", "compileall", "-q", "src", "scripts", "tools", "tests"], cwd=TARGET_ROOT, env=target_env)
    for command in (
        [sys.executable, "scripts/handoff_authority.py", "verify", "--repo-root", "."],
        [sys.executable, "scripts/validate_formal_execution_channel.py", "--repo-root", "."],
        [sys.executable, "scripts/validate_governance_rule_inventory.py", "--repo-root", "."],
        [sys.executable, "scripts/validate_governance_pipeline_stage_status.py", "--repo-root", "."],
        [sys.executable, "-m", "pytest", "-q", "tests/test_formal_execution_channel.py::test_current_registry_uses_canonical_channel"],
    ):
        run(command, cwd=TARGET_ROOT, env=target_env)

    baseline_log = Path("/tmp/e8-backlog-baseline-pytest.log")
    target_log = Path("/tmp/e8-backlog-target-pytest.log")
    baseline_env = os.environ.copy()
    baseline_env["PYTHONPATH"] = f"{TRUSTED_ROOT / 'src'}:{TRUSTED_ROOT / 'scripts'}"
    with baseline_log.open("wb") as handle:
        baseline_run = subprocess.run(
            [sys.executable, "-m", "pytest", "-q"], cwd=TRUSTED_ROOT, env=baseline_env, stdout=handle, stderr=subprocess.STDOUT
        )
    shutil.rmtree("/tmp/digest-a", ignore_errors=True)
    with target_log.open("wb") as handle:
        target_run = subprocess.run(
            [sys.executable, "-m", "pytest", "-q"], cwd=TARGET_ROOT, env=target_env, stdout=handle, stderr=subprocess.STDOUT
        )
    baseline_failures = failures(baseline_log)
    target_failures = failures(target_log)
    new_failures = sorted(target_failures - baseline_failures)
    differential = {
        "base_commit": current_main,
        "baseline_exit_code": baseline_run.returncode,
        "target_exit_code": target_run.returncode,
        "baseline_failures": sorted(baseline_failures),
        "target_failures": sorted(target_failures),
        "new_failures": new_failures,
    }
    print(json.dumps(differential, indent=2, sort_keys=True))
    if new_failures:
        raise SystemExit(f"Backlog target introduced new pytest failures: {new_failures}")
    if baseline_run.returncode == 0 and target_run.returncode != 0:
        raise SystemExit("Current main is pytest-clean but backlog target is not")

    run(["ruff", "check", "tests/test_formal_execution_channel.py"], cwd=TARGET_ROOT)
    run(["git", "diff", "--check", current_main, target_commit], cwd=TARGET_ROOT)
    if output(["git", "status", "--porcelain"], cwd=TARGET_ROOT):
        raise SystemExit("Target worktree is dirty after validation")
    temp_paths = output(["git", "ls-files", ".github/workflows/temporary_e8_*", "tools/temporary_*e8*"], cwd=TARGET_ROOT)
    if temp_paths:
        raise SystemExit(f"Temporary paths remain in target tree:\n{temp_paths}")

    latest_main = output(["git", "ls-remote", "origin", "refs/heads/main"]).split()[0]
    if latest_main != current_main:
        raise SystemExit(f"main advanced during finalization: {current_main} -> {latest_main}")
    tree = output(["git", "rev-parse", f"{target_commit}^{{tree}}"], cwd=TARGET_ROOT)
    commit_env = os.environ.copy()
    commit_env.update(
        {
            "GIT_AUTHOR_NAME": "github-actions[bot]",
            "GIT_AUTHOR_EMAIL": "41898282+github-actions[bot]@users.noreply.github.com",
            "GIT_COMMITTER_NAME": "github-actions[bot]",
            "GIT_COMMITTER_EMAIL": "41898282+github-actions[bot]@users.noreply.github.com",
        }
    )
    final = subprocess.check_output(
        ["git", "commit-tree", tree, "-p", trigger, "-p", source_commit],
        cwd=TARGET_ROOT,
        env=commit_env,
        input="Finalize six completed E8 pilot closures\n",
        text=True,
    ).strip()
    print(f"E8_BACKLOG_SOURCE_COMMIT={source_commit}")
    print(f"E8_BACKLOG_TARGET_COMMIT={target_commit}")
    print(f"E8_BACKLOG_FINAL_COMMIT={final}")
    run(["git", "push", "origin", f"{final}:refs/heads/{BRANCH}"], cwd=TARGET_ROOT)


if __name__ == "__main__":
    main()
