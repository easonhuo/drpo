# E8 Multitask Lambda Wide Right-Tail Protocol 03

Experiment ID: `EXT-C-E8-MULTITASK-EXP-LAMBDA-WIDE-RIGHTTAIL-03`

RunSpec ID: `E8_MULTITASK_LAMBDA_WIDE_RIGHTTAIL_03_20260904_01`.

Status: `not_run`.

Execution class: `pilot` only.

Scientific role: external-validity, single-seed high-lambda response-shape / curve-boundary follow-up. This experiment does not test convergence, statistical significance, steady state, or universal method ranking.

Runtime config authority: `configs/e8_multitask_exp_lambda_wide_righttail_03.yaml`.

## Claim and scope

For the five transfer tasks whose tested high-lambda response boundary remains unresolved after the predecessor pilots, extend the Exp taper coefficient substantially to determine whether the validation response shows a clear right-tail decline, a broad plateau or saturation, or continued increase over the newly tested range.

The experiment is intentionally limited to the tested finite lambda range. It must not be described as establishing an asymptotic regime.

The five active tasks are:

- Word Sorting;
- Mini Sudoku;
- Maze;
- Knights & Knaves;
- WikiSQL.

Countdown, Spiral Matrix, Word Ladder, and Graph Coloring receive zero new scientific cells in this follow-up. They are merely deprioritized for this specific extension based on the preceding pilot response shapes; this does not create a general claim that those tasks are permanently closed.

## Predecessor evidence and grid provenance

The new grid was chosen from the combined predecessor response-shape evidence, not from the original 208-row cold-start anchor alone.

1. Repository-landed historical cold-start anchor:
   `experiments/results/e8_multitask_exp_coldstart_20260820_02/CURVE_ANCHOR.csv` (`208` rows).
2. `EXT-C-E8-MULTITASK-EXP-LAMBDA-COMPLETION-01`, run ID `E8_MULTITASK_EXP_LAMBDA_COMPLETION_20260827_01`: `199/199` raw scientific cells complete and terminal-audited as pilot response-shape evidence. Guarded artifact SHA-256: `1d9dc48e2016205f7f74615f57a67bc89f885c6c51de0977bd9e1d16af68d467`.
3. `EXT-C-E8-MULTITASK-EXP-LAMBDA-CURVE-COMPLETION-02`, run ID `E8_MULTITASK_EXP_LAMBDA_CURVE_COMPLETION_02`: `140/140` scientific workload complete and terminal-audited as pilot response-shape evidence. Guarded artifact SHA-256: `72a68bd27a241f2a0458ebcd1112ff48b8a913e97836888cf24f197ac2ef2944`. Its outer supervisor returned nonzero only after the scientific workload because the final package exceeded the hardened package-size limit; that packaging event must remain separate from scientific workload completion and from NaN/Inf numerical failure.

The two predecessor pilot source commits were not authoritative-main-resolvable at result audit time. Their scientific status therefore remains `pilot`; this follow-up does not upgrade their provenance or result status.

No presentation-digitized Countdown points are scientific inputs to this follow-up.

The predecessor maximum tested lambdas used to define the five new grids are:

| Task | Predecessor max | New first point (1.15x) | New last point (20x) |
| --- | ---: | ---: | ---: |
| Word Sorting | 120 | 138 | 2400 |
| Mini Sudoku | 220 | 253 | 4400 |
| Maze | 100 | 115 | 2000 |
| Knights & Knaves | 110 | 126.5 | 2200 |
| WikiSQL | 240 | 276 | 4800 |

Each active task has exactly 20 explicit lambda values in the runtime YAML, geometrically spaced with common ratio approximately `1.16219938`. The values are runtime authorities and are not regenerated inside Python.

## Frozen scientific workload

- Active tasks: `5`.
- New Exp cells per active task: `20`.
- Total new scientific cells: `100`.
- New Positive-only cells: `0`.
- New Global/uncontrolled endpoint cells: `0`.
- Transfer tuning seed: `4000` for every new Exp cell.
- Training horizon: `1200` optimizer updates, no early stopping.
- Scheduler capacity: `16` cells (`8` GPUs x `2` slots/GPU), dynamic slot queue, no wave barriers.
- Nominal audit geometry: `7` refill batches for 100 cells.
- Test partition access: forbidden.

All other model, data-bank, negative-consumer, optimizer, LoRA, evaluation, runtime, recovery, and terminal-audit settings are inherited unchanged from the reviewed config-driven E8 cold-start family and are frozen explicitly in `configs/e8_multitask_exp_lambda_wide_righttail_03.yaml`.

The only scientific variable changed across the new cells is the configured task-local Exp lambda.

## Metrics and reporting boundary

Primary response metric: validation late-window Pass@8 mean over updates `[800, 900, 1000, 1100, 1200]`.

Secondary response metric: validation late-window Greedy mean. Terminal Pass@8, terminal Greedy, valid/structure diagnostics, and raw per-cell trajectories remain required evidence.

A fixed `1200`-update horizon is not convergence. Any later statement about steady state, collapse, or method ranking requires the applicable terminal-state audit and evidence beyond this pilot's authorized role.

Task-performance degradation, valid/structure-boundary events, and NaN/Inf numerical failures must be reported separately.

## Implementation contract

No experiment-specific Python module or launcher is introduced. The generic config-driven implementation merged in PR #340 remains the execution path:

- `scripts/bootstrap_e8_multitask_exp_coldstart.sh`;
- `scripts/run_e8_multitask_exp_coldstart.sh`;
- `src/drpo/e8_experiment_config.py`;
- `src/drpo/e8_multitask_exp_tuning.py`;
- the canonical imported Countdown/transfer trainer and taper modules identified by the YAML's `canonical_coldstart` block.

The new config must expand to exactly `100` unique Exp cells, with exactly 20 cells for each active task and zero cells for the four inactive tasks. Empty task grids mean zero scientific cells; they are not placeholders for future automatic expansion.

## Pilot launch identity

This experiment must be launched with `E8_COLDSTART_RUN_CLASS=pilot`. The generic runner's default `formal` value must not be relied upon for this experiment.

The preferred execution contract is `runspecs/ready/E8_MULTITASK_LAMBDA_WIDE_RIGHTTAIL_03_20260904_01.yaml`. The RunSpec fixes the run ID, pilot environment, entrypoint, bounded recovery, artifact inventory, and `drpo-results` delivery policy. A manual launch is a fallback only and must use the same run ID and the same reviewed branch/config identity.

PR #343 is the durable review record for the launch snapshot. After final review and CI, the exact PR-head commit SHA is the frozen pilot runtime source identity. If the branch head changes after that review, the new SHA must be reviewed before launch. The bootstrap/run manifests must record the exact resolved source commit and config identity.

The equivalent manual launch form is:

```bash
set -euo pipefail

BRANCH="dev/e8-multitask-lambda-wide-righttail-03"
SOURCE_ROOT="${HOME}/drpo-e8-lambda-wide-righttail-03-source"
BOOTSTRAP_ROOT="${HOME}/drpo-e8-lambda-wide-righttail-03"

if [[ ! -d "${SOURCE_ROOT}/.git" ]]; then
  git clone --branch "${BRANCH}" --single-branch \
    "https://github.com/easonhuo/drpo.git" "${SOURCE_ROOT}"
else
  test -z "$(git -C "${SOURCE_ROOT}" status --porcelain)" || {
    echo "ERROR: dedicated source checkout is dirty" >&2
    exit 1
  }
  git -C "${SOURCE_ROOT}" fetch origin "${BRANCH}"
  git -C "${SOURCE_ROOT}" switch "${BRANCH}"
  git -C "${SOURCE_ROOT}" merge --ff-only "origin/${BRANCH}"
fi

E8_COLDSTART_EXISTING_REPO="${SOURCE_ROOT}" \
E8_COLDSTART_BOOTSTRAP_ROOT="${BOOTSTRAP_ROOT}" \
E8_COLDSTART_TARGET_REF="refs/heads/${BRANCH}" \
E8_COLDSTART_EXPERIMENT_ID="EXT-C-E8-MULTITASK-EXP-LAMBDA-WIDE-RIGHTTAIL-03" \
E8_COLDSTART_CONFIG="configs/e8_multitask_exp_lambda_wide_righttail_03.yaml" \
E8_COLDSTART_RUN_ID="E8_MULTITASK_LAMBDA_WIDE_RIGHTTAIL_03_20260904_01" \
E8_COLDSTART_RUN_CLASS="pilot" \
E8_COLDSTART_REQUIRE_ORIGIN_MAIN="0" \
bash "${SOURCE_ROOT}/scripts/bootstrap_e8_multitask_exp_coldstart.sh" full
```

Before execution, confirm that the resolved branch head is the exact reviewed PR-head SHA. Do not launch from a later unreviewed branch mutation.

## Artifact budget and failure semantics

The repository hardened artifact policy retains a `25 MiB` main-package hard limit. The preceding 140-cell run exceeded that limit only after its scientific workload completed. This 100-cell run must therefore preserve the normal recovery/partial-output path and treat packaging state separately from scientific workload state.

If packaging exceeds the configured limit, do not reinterpret that as training failure, task-performance collapse, or NaN/Inf numerical collapse, and do not silently change scientific parameters to reduce output volume. Preserve the available recovery/terminal evidence and report the packaging failure explicitly.

## Expected outputs and initial result status

The run must use the existing E8 hardened runtime/package layout and preserve raw per-cell curves, aggregate response tables, logs, run manifest, config/source provenance, `RUN_COMPLETE.json` or failure evidence as applicable, terminal audit, recovery state, checksums, and package/recovery status.

Initial scientific result status is `not_run`. After execution, the strongest status this protocol pre-authorizes is `pilot` response-shape / curve-boundary evidence. It does not pre-authorize significance, convergence, steady-state, or method-ranking claims.
