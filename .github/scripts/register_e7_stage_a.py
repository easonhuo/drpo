#!/usr/bin/env python3
"""Materialize the Stage A registration onto the frozen dev implementation head."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import yaml

REPO = "easonhuo/drpo"
MAIN_SHA = "94f72925cdf7a74f73e7255bd802a413acda6dbf"
IMPLEMENTATION_SHA = "2d4d295022c75b0c2cde283d2d9c3402779c5764"
TARGET_BRANCH = "dev/e7-squared-exp-kl-tune-stage-a"
SOURCE_BRANCH = "registration-source/e7-squared-exp-kl-tune-stage-a"
EXPERIMENT_ID = "EXT-H-E7-SQUARED-EXP-KL-TUNE-01"
UPDATE_ID = "EXT-H-E7-SQUARED-EXP-KL-TUNE-STAGE-A-REGISTRATION-2026-07-14"
DELTA_DIR = f"docs/handoff_deltas/{UPDATE_ID}"


def run(*args: str, cwd: Path | None = None, capture: bool = False) -> str:
    proc = subprocess.run(
        list(args),
        cwd=None if cwd is None else str(cwd),
        text=True,
        stdout=subprocess.PIPE if capture else None,
        stderr=subprocess.STDOUT if capture else None,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"command failed ({proc.returncode}): {' '.join(args)}\n{proc.stdout or ''}"
        )
    return (proc.stdout or "").strip()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def protocol_document() -> str:
    return f"""# {EXPERIMENT_ID}

## Status

- scientific class: Hopper/Walker external-validity development screening pilot;
- stage: `stage_a_kl_threshold_and_reference_lifecycle_screen`;
- implementation commit: `{IMPLEMENTATION_SHA}`;
- registration status: registered before or in parallel with code-first execution;
- result status: `not_run` at registration time;
- held-out seeds `204--207`: untouched and forbidden;
- predecessor: `EXT-H-E7-SQUARED-EXP-NIGHT-01`;
- GAE: excluded from Stage A;
- Stage B: not authorized by this registration.

## Question

Stage A tests whether the positive finite-horizon signal from analytic-KL-triggered
old-policy refresh is robust to the KL threshold and whether its benefit comes from
adaptive refresh timing rather than merely allowing a longer fixed reference window.
It also probes the squared-EXP region above `c=8` without selecting a different
coefficient per dataset.

## Frozen matrix

All 150 branches preserve the predecessor's canonical actor and critic architecture,
critic target and expectile loss, one-step TD advantage, actor-before-critic order,
optimizer, batch size `256`, learning rate `3e-4`, datasets, and evaluation protocol.

- datasets: `hopper-medium-expert-v2`, `walker2d-medium-v2`,
  `walker2d-medium-replay-v2`;
- development seeds: `200, 201`;
- horizon: `1,000,000` optimizer updates;
- evaluation: every `50,000` updates with ten episodes;
- kernel: `w(d)=w(0)exp[-c(d/2)^2]`;
- controls per lifecycle: Positive-only plus `w(0)=1` at
  `c in {{4,8,16,32}}`;
- PPO clip epsilon: `0.2`;
- 500k: intermediate checkpoint only;
- terminal reporting window: 800k--1M.

Reference lifecycles:

1. fixed K4 PPO;
2. fixed K16 PPO with no KL-triggered refresh;
3. `K_max=16`, analytic `KL(old||new)` threshold `0.003`;
4. `K_max=16`, analytic `KL(old||new)` threshold `0.01`;
5. `K_max=16`, analytic `KL(old||new)` threshold `0.03`.

The branch count is

\[
5\times5\times3\times2=150.
\]

The old/current ratio remains a proximal ablation and is not described as
behavior-policy importance correction.

## Code-first launch policy

The clean implementation checkout at `{IMPLEMENTATION_SHA}` may run before the
materialized registration commit reaches the server. The runner records the initial
registration state but registration does not block liveness or the development pilot.
The scientific matrix must remain byte-equivalent to the registered config.

## Qualification output

Terminal aggregation writes `stage_a_qualification.json`. An adaptive threshold
qualifies only when all branches are terminal-audited, pooled paired late mean and
median differences versus fixed K4 are positive, and it wins more than half of the
15 `(dataset, control)` cells. Positive-only and squared-EXP effects are reported
separately. The tie-break order is mean difference, median difference, then cell wins.

The qualification record does not launch Stage B. Stage B requires a separate frozen
registration and explicit launch decision.

## Diagnostics and terminal audit

Every branch records squared-EXP geometry, effective negative mass, PPO ratio-outside
and true objective-clip fractions. Adaptive branches additionally record analytic KL,
interval maxima, triggered refresh counts, and total reference refresh counts.

Task-performance collapse or degradation, support or variance-boundary events, and
NaN/Inf numerical failure remain separate categories. Missing registered thresholds
must be reported as not adjudicated rather than inferred after the run.

## Interpretation limits

This is two-development-seed external-validity screening. It does not establish
convergence, steady state, a universal PPO ranking, lower seed variance from KL
refresh, controlled causal actor-update identification, a GAE result, OOD
generalization, or a formal D4RL method ranking. Hopper/Walker evidence does not
replace C-U1 or D-U1 controlled mechanism evidence.
"""


def scope_document() -> str:
    return f"""# Scope contract: {EXPERIMENT_ID} Stage A

## Allowed changes

- add the dedicated Stage A config, runner, bootstrap, aggregation, runtime autotune,
  launch scripts, and tests already frozen at `{IMPLEMENTATION_SHA}`;
- run exactly the registered 150-branch matrix;
- autotune only active subprocess count;
- record pre-registration code-first launch state without blocking execution;
- preserve and report failures and partial outputs.

## Frozen scientific variables

- datasets: Hopper medium-expert, Walker2d medium, Walker2d medium-replay;
- development seeds: `200,201`;
- held-out seeds: `204--207` forbidden;
- horizon: 1M;
- evaluation: 50k cadence, ten episodes;
- squared-EXP coefficients: `4,8,16,32` plus Positive-only;
- lifecycle controls: fixed K4, fixed K16, and adaptive K16 at
  `target_kl=0.003,0.01,0.03`;
- PPO epsilon: `0.2`;
- one-step TD advantage and all inherited optimizer/critic controls.

## Forbidden changes

- GAE or any other advantage-estimator change;
- KL penalty, entropy bonus, actor-gradient clipping, or value clipping;
- different coefficients per dataset presented as one common method;
- held-out execution;
- automatic Stage B launch;
- convergence, steady-state, universal ranking, stability, OOD, or causal claims;
- destructive changes to predecessor code, results, registry history, or handoff history.

## Completion gate

All 150 branches must be accounted for and the terminal audit must separately report
task performance, support/variance boundary status, and NaN/Inf status. A fixed 1M
endpoint is not convergence.
"""


def runspec() -> str:
    return f"""version: 1
run_id: E7_SQUARED_EXP_KL_TUNE_STAGE_A_PILOT_20260714_01
lane: e7
priority: 126
created_at: '2026-07-14T00:00:00Z'
experiment_id: {EXPERIMENT_ID}
repo_commit: {IMPLEMENTATION_SHA}

provenance:
  commit_policy: exact
  protected_paths:
    - configs/e7_squared_exp_kl_tune_stage_a_v1.json
    - src/drpo/e7_squared_exp_kl_tune_stage_a.py
    - src/drpo/e7_squared_exp_kl_tune_stage_a_bootstrap.py
    - src/drpo/e7_squared_exp_kl_tune_stage_a_aggregate.py
    - src/drpo/e7_squared_exp_kl_tune_stage_a_runtime_autotune.py
    - scripts/run_e7_squared_exp_kl_tune_stage_a_auto.py
    - scripts/run_e7_squared_exp_kl_tune_stage_a_one_click.sh
    - scripts/run_e7_squared_exp_kl_tune_stage_a_resume_one_click.sh

purpose: >
  Execute the 150-branch, 1M-step Stage A development screen comparing fixed K4,
  fixed K16, and analytic-KL adaptive K16 thresholds 0.003, 0.01, and 0.03 under
  Positive-only and squared EXP c={{4,8,16,32}}. Registration state never blocks the
  code-first pilot. Stage B and GAE are outside this RunSpec.

entrypoint:
  command: bash scripts/run_e7_squared_exp_kl_tune_stage_a_one_click.sh
  cwd: repo_root

policy:
  existing_script_required: true
  forbid_hparam_change: true
  forbid_cross_lane: true
  formal_evidence_allowed: false

outputs:
  run_dir: outputs/e7/squared_exp_kl_tune_stage_a_001
  summary_file: outputs/e7/squared_exp_kl_tune_stage_a_001/RUN_SUMMARY.json
  audit_file: outputs/e7/squared_exp_kl_tune_stage_a_001/aggregate/terminal_audit.json

success_criteria:
  - execution plan records exactly 150 branches
  - all branches use development seeds 200 and 201 only
  - held-out seeds 204 through 207 remain untouched
  - every branch reaches exactly 1000000 updates unless separately failed
  - fixed K4, fixed K16, and three adaptive KL thresholds remain paired
  - terminal aggregation writes stage_a_qualification.json
  - task degradation, support or variance boundary, and NaN/Inf remain separate
  - Stage B is not automatically launched
  - fixed 1M is not described as convergence or steady state

artifacts:
  package_policy: manifest_only
  exclude:
    - '**/*.pt'
    - '**/*.pth'
    - '**/*.ckpt'
    - '**/*.safetensors'
    - '**/*.bin'
    - '**/checkpoint*'
    - '**/checkpoints/**'
    - '**/model*'
    - '**/optimizer*'
    - '**/wandb/**'
  max_package_size_mb: 150
  fail_if_package_too_large: true

publish:
  enabled: false
  auto: false

recovery:
  enabled: true
  max_attempts: 2
  resume_command: bash scripts/run_e7_squared_exp_kl_tune_stage_a_resume_one_click.sh
  retryable_exit_codes: [137, 143]
  backoff_seconds: 60
"""


def registry_entry() -> str:
    return f"""- id: {EXPERIMENT_ID}
  environment: Hopper_and_Walker2d_external_validity
  name: squared_exp_kl_threshold_and_reference_lifecycle_stage_a_screen
  status: not_run
  parent_experiment: EXT-H-E7-SQUARED-EXP-NIGHT-01
  registration_base_commit: {IMPLEMENTATION_SHA}
  claim: >-
    Screen whether analytic-KL-triggered adaptive old-policy refresh provides a
    common finite-horizon mean-performance benefit over fixed K4, separate that
    effect from fixed K16 reuse, and probe squared-EXP coefficients above c=8.
  role: external_validity_development_screening
  execution_class: pilot
  protocol_document: docs/experiments/{EXPERIMENT_ID}.md
  scope_contract: docs/scopes/{EXPERIMENT_ID}.md
  proposal_document: docs/proposals/{EXPERIMENT_ID}.md
  config_entrypoint: configs/e7_squared_exp_kl_tune_stage_a_v1.json
  code_entrypoint: src/drpo/e7_squared_exp_kl_tune_stage_a.py
  launch_entrypoint: scripts/run_e7_squared_exp_kl_tune_stage_a_one_click.sh
  resume_entrypoint: scripts/run_e7_squared_exp_kl_tune_stage_a_resume_one_click.sh
  runspec: runspecs/templates/E7_SQUARED_EXP_KL_TUNE_STAGE_A_PILOT_20260714_01.yaml
  implementation_commit: {IMPLEMENTATION_SHA}
  datasets:
  - hopper-medium-expert-v2
  - walker2d-medium-v2
  - walker2d-medium-replay-v2
  development_seeds: [200, 201]
  held_out_seeds: [204, 205, 206, 207]
  matrix:
    horizon_updates: 1000000
    squared_exp_coefficients: [4.0, 8.0, 16.0, 32.0]
    positive_only_anchor: true
    reference_lifecycles:
    - ppo_clip_k4
    - ppo_clip_k16
    - ppo_clip_kl_k16_t0p003
    - ppo_clip_kl_k16_t0p01
    - ppo_clip_kl_k16_t0p03
    expected_branches: 150
  formal_run_status: not_run
  execution:
    state: registered_ready_for_real_data_liveness
    run_id: null
    last_heartbeat_utc: null
    process_exit_code: null
  evidence:
    code_committed: true
    implementation_commit: {IMPLEMENTATION_SHA}
    implementation_tests_passed: true
    run_started: false
    raw_complete: false
    terminal_audited: false
    package_created: false
    delivered_to_user: false
  launch_policy:
    code_first_pre_registration_allowed: true
    registration_blocks_launch: false
    clean_checkout_required: true
    registration_state_recorded: true
  qualification:
    baseline: ppo_clip_k4
    adaptive_candidates:
    - ppo_clip_kl_k16_t0p003
    - ppo_clip_kl_k16_t0p01
    - ppo_clip_kl_k16_t0p03
    stage_b_auto_launch: forbidden
  interpretation_limits:
  - development_screening_only
  - two_development_seeds_not_confirmation
  - fixed_1m_not_convergence_or_steady_state
  - no_universal_ppo_or_kl_superiority_claim
  - no_lower_seed_variance_claim
  - no_gae_result
  - no_ood_generalization_claim
  - hopper_walker_do_not_replace_cu1_du1
"""


def prepare_source(source: Path) -> None:
    write(source / f"docs/experiments/{EXPERIMENT_ID}.md", protocol_document())
    write(source / f"docs/scopes/{EXPERIMENT_ID}.md", scope_document())
    write(
        source
        / "runspecs/templates/E7_SQUARED_EXP_KL_TUNE_STAGE_A_PILOT_20260714_01.yaml",
        runspec(),
    )

    registry_path = source / "experiments/registry.yaml"
    registry_before = registry_path.read_text(encoding="utf-8")
    parsed = yaml.safe_load(registry_before)
    experiments = parsed.get("experiments") if isinstance(parsed, dict) else None
    if not isinstance(experiments, list):
        raise RuntimeError("registry experiments list is missing")
    if any(item.get("id") == EXPERIMENT_ID for item in experiments if isinstance(item, dict)):
        raise RuntimeError(f"{EXPERIMENT_ID} is already registered")
    registry_after = registry_before.rstrip() + "\n" + registry_entry()
    registry_path.write_text(registry_after, encoding="utf-8")
    parsed_after = yaml.safe_load(registry_after)
    matches = [
        item
        for item in parsed_after["experiments"]
        if isinstance(item, dict) and item.get("id") == EXPERIMENT_ID
    ]
    if len(matches) != 1:
        raise RuntimeError("registry append did not create exactly one entity")

    sys.path.insert(0, str(source / "scripts"))
    import handoff_delta_shadow as shadow  # type: ignore

    handoff_text = (source / "docs/handoff.md").read_text(encoding="utf-8")
    operation_content = (
        f"- **Squared-EXP KL threshold Stage A (`{EXPERIMENT_ID}`):** registered as "
        f"a 150-branch Hopper/Walker development screening pilot pinned to implementation "
        f"commit `{IMPLEMENTATION_SHA}`. It compares Positive-only and `c={{4,8,16,32}}` "
        "under fixed K4, fixed K16, and adaptive `K_max=16` analytic-KL thresholds "
        "`0.003`, `0.01`, and `0.03`, with PPO epsilon `0.2`, one-step TD, 1M updates, "
        "and development seeds `200,201`. Held-out seeds `204--207`, GAE, and Stage B "
        "remain forbidden. Code-first clean-checkout launch is allowed and registration "
        "does not block the pilot. Current result status is `not_run`; no convergence, "
        "steady-state, stability, universal ranking, OOD, or causal claim is authorized."
    )
    operations = [
        {
            "operation_id": "append-e7-squared-exp-kl-stage-a-registration",
            "op": "append_to_section",
            "heading_path": [
                "0. 研究与执行原则（每次新会话首先阅读）",
                "0.1 当前执行门禁",
            ],
            "block_id": "e7-squared-exp-kl-tune-stage-a-registration",
            "content": operation_content,
        }
    ]
    rendered = shadow.render(handoff_text, operations)
    shadow.verify_history_preservation(handoff_text, rendered.text, operations)

    evidence = [
        "experiments/registry.yaml",
        f"docs/handoff_deltas/{UPDATE_ID}/HANDOFF_DELTA.yaml",
        f"docs/experiments/{EXPERIMENT_ID}.md",
        f"docs/scopes/{EXPERIMENT_ID}.md",
        f"docs/proposals/{EXPERIMENT_ID}.md",
        "configs/e7_squared_exp_kl_tune_stage_a_v1.json",
        "src/drpo/e7_squared_exp_kl_tune_stage_a.py",
        "src/drpo/e7_squared_exp_kl_tune_stage_a_bootstrap.py",
        "src/drpo/e7_squared_exp_kl_tune_stage_a_aggregate.py",
        "src/drpo/e7_squared_exp_kl_tune_stage_a_runtime_autotune.py",
        "scripts/run_e7_squared_exp_kl_tune_stage_a_auto.py",
        "scripts/run_e7_squared_exp_kl_tune_stage_a_one_click.sh",
        "scripts/run_e7_squared_exp_kl_tune_stage_a_resume_one_click.sh",
        "tests/test_e7_squared_exp_kl_tune_stage_a.py",
        "runspecs/templates/E7_SQUARED_EXP_KL_TUNE_STAGE_A_PILOT_20260714_01.yaml",
    ]
    for relative in evidence:
        if not (source / relative).is_file():
            raise RuntimeError(f"registration evidence is missing: {relative}")

    delta = {
        "schema_version": 3,
        "update_id": UPDATE_ID,
        "mode": "authoritative",
        "base": {
            "commit": IMPLEMENTATION_SHA,
            "handoff_sha256": sha256_text(handoff_text),
            "registry_sha256": sha256_text(registry_before),
        },
        "renderer_version": 1,
        "operations": operations,
        "registry": {
            "mode": "expected_after",
            "exact_base_after_sha256": sha256_text(registry_after),
            "changes": [
                {
                    "change_id": "add-e7-squared-exp-kl-stage-a",
                    "kind": "add_entity",
                    "entity_id": EXPERIMENT_ID,
                    "evidence": evidence,
                }
            ],
        },
        "expected": {
            "exact_base_candidate_sha256": sha256_text(rendered.text),
        },
    }
    delta_path = source / DELTA_DIR / "HANDOFF_DELTA.yaml"
    delta_path.parent.mkdir(parents=True, exist_ok=True)
    delta_path.write_text(
        yaml.safe_dump(delta, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )


def main() -> int:
    root = Path(os.environ.get("GITHUB_WORKSPACE", ".")).resolve()
    run("git", "config", "user.name", "github-actions[bot]", cwd=root)
    run(
        "git",
        "config",
        "user.email",
        "41898282+github-actions[bot]@users.noreply.github.com",
        cwd=root,
    )
    run("git", "fetch", "origin", "main", TARGET_BRANCH, "--prune", cwd=root)
    remote_target = run(
        "git", "rev-parse", f"origin/{TARGET_BRANCH}", cwd=root, capture=True
    )
    if remote_target != IMPLEMENTATION_SHA:
        raise RuntimeError(
            f"target branch moved: expected {IMPLEMENTATION_SHA}, got {remote_target}"
        )
    if run("git", "rev-parse", "origin/main", cwd=root, capture=True) != MAIN_SHA:
        raise RuntimeError("main moved after the approved proposal/implementation base")

    temp = Path("/tmp/e7-stage-a-registration")
    shutil.rmtree(temp, ignore_errors=True)
    source = temp / "source"
    target = temp / "target"
    trusted = temp / "trusted"
    temp.mkdir(parents=True)

    run("git", "worktree", "add", "--detach", str(source), IMPLEMENTATION_SHA, cwd=root)
    run("git", "worktree", "add", "--detach", str(target), IMPLEMENTATION_SHA, cwd=root)
    run("git", "worktree", "add", "--detach", str(trusted), MAIN_SHA, cwd=root)
    for worktree in (source, target):
        run("git", "config", "user.name", "github-actions[bot]", cwd=worktree)
        run(
            "git",
            "config",
            "user.email",
            "41898282+github-actions[bot]@users.noreply.github.com",
            cwd=worktree,
        )

    prepare_source(source)
    run("git", "add", "-A", cwd=source)
    run(
        "git",
        "commit",
        "-m",
        f"Prepare {EXPERIMENT_ID} Stage A registration intent",
        cwd=source,
    )
    source_sha = run("git", "rev-parse", "HEAD", cwd=source, capture=True)
    run(
        sys.executable,
        "scripts/handoff_authority.py",
        "validate-delta",
        "--repo-root",
        str(source),
        "--delta",
        f"{DELTA_DIR}/HANDOFF_DELTA.yaml",
        "--source-patch-commit",
        source_sha,
        "--json",
        cwd=source,
    )
    run(
        "git",
        "push",
        "--force-with-lease",
        "origin",
        f"HEAD:refs/heads/{SOURCE_BRANCH}",
        cwd=source,
    )

    run("git", "cherry-pick", "-n", source_sha, cwd=target)
    run(
        sys.executable,
        "scripts/handoff_authority.py",
        "normalize",
        "--repo-root",
        str(target),
        "--trusted-repo-root",
        str(trusted),
        "--current-before",
        IMPLEMENTATION_SHA,
        "--source-base",
        IMPLEMENTATION_SHA,
        "--source-patch-commit",
        source_sha,
        "--json",
        cwd=target,
    )
    run("git", "add", "-A", cwd=target)
    run(
        "git",
        "commit",
        "-m",
        f"Register {EXPERIMENT_ID} Stage A",
        cwd=target,
    )
    final_sha = run("git", "rev-parse", "HEAD", cwd=target, capture=True)
    run(
        sys.executable,
        "scripts/handoff_authority.py",
        "verify",
        "--repo-root",
        str(target),
        "--json",
        cwd=target,
    )
    run(
        "python",
        "scripts/validate_formal_execution_channel.py",
        "--repo-root",
        str(target),
        cwd=target,
    )
    run(
        "git",
        "push",
        "origin",
        f"HEAD:refs/heads/{TARGET_BRANCH}",
        cwd=target,
    )
    print(
        json.dumps(
            {
                "status": "PASS",
                "implementation_commit": IMPLEMENTATION_SHA,
                "registration_source_commit": source_sha,
                "registration_commit": final_sha,
                "target_branch": TARGET_BRANCH,
                "source_branch": SOURCE_BRANCH,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
