# E8 Multitask Lambda Curve-Completion Protocol 02

Experiment ID: `EXT-C-E8-MULTITASK-EXP-LAMBDA-CURVE-COMPLETION-02`

Status: `not_run`.

Base branch: `main`.

Base commit for protocol design: `8b0616cf0f887f86ec04e398a7604c3d3940aa5d`.

This experiment is a response-shape / curve-boundary successor to `EXT-C-E8-MULTITASK-EXP-LAMBDA-COMPLETION-01`. It is not a method-ranking, significance, convergence, or steady-state experiment.

## Claim and scope

Complete the remaining lambda-response boundaries for seven transfer tasks after the previous tail-completion pilot. `Countdown` and `spiral_matrix` receive zero new scientific cells. Historical results remain immutable and are concatenated only for plotting/response-shape analysis.

The scientific implementation is unchanged: Qwen2.5-0.5B-Instruct, fresh LoRA initialization, the byte-locked paper Countdown trainer/taper/loss path, the same 16-negative consumer, current-policy sequence-surprisal coordinate, optimizer, data, evaluation, scheduler, recovery behavior, and terminal audit taxonomy. The only scientific variable changed across cells is the configured task-local lambda value.

All new Exp cells use tuning seed offset `4000`. No new Positive-only cells are run. Test partitions remain forbidden.

## Frozen workload

- Total new cells: `140`.
- New Exp cells: `140`.
- New Positive-only cells: `0`.
- Countdown: `0` new cells.
- Spiral Matrix: `0` new cells.
- Seven active transfer tasks: `20` Exp cells each.
- Scheduler capacity remains `16` concurrent cells on 8 GPUs with 2 slots/GPU.
- Nominal audit geometry is `9` refill batches (`8 x 16 + 1 x 12 = 140`); runtime remains the existing dynamic slot queue with no hard wave barriers.

## Frozen lambda grids

All values below are explicit runtime authorities. They are not regenerated inside Python.

### Word Sorting — right-tail completion

`[13.0, 14.613200623048, 16.426587111496, 18.465001000944, 20.756366471659, 23.33207288122, 26.227404766527, 29.48202520578, 33.140519161987, 37.253004251237, 41.87581730266, 47.072286115221, 52.913596983634, 59.479769877612, 66.860754633405, 75.157663174333, 84.484154640461, 94.967992402283, 106.752794287896, 120.0]`

### Mini Sudoku — right-tail completion

`[45.0, 48.920032072626, 53.181545288594, 57.814286693108, 62.850594650729, 68.32562457994, 74.277594351841, 80.74805106599, 87.782161065562, 95.42902521626, 103.742021649752, 112.77917836411, 122.603578282148, 133.283799595142, 144.894394465684, 157.516409431221, 171.237951140894, 186.154801374739, 202.371085638346, 220.0]`

### Maze — right-tail completion

`[12.0, 13.416690176528, 15.000631274412, 16.771568521761, 18.75157821924, 20.965342940706, 23.440459223343, 26.207781582928, 29.301807142689, 32.761105670486, 36.628800385118, 40.953105525419, 45.787927383434, 51.193536294075, 57.237317958205, 63.994613465056, 71.549658485611, 79.99663334482, 89.440837062731, 100.0]`

### Word Ladder — right-tail completion

Same 20 values as Word Sorting.

### Knights & Knaves — right-tail completion

`[12.0, 13.484161746455, 15.151884833714, 17.025872155122, 19.131634501178, 21.497837840663, 24.156693553547, 27.144396927966, 30.501619890557, 34.274064677764, 38.513085985287, 43.276390065063, 48.628820286667, 54.64323985706, 61.401523715247, 68.9956731046, 77.529067995666, 87.117874408792, 97.892625794628, 110.0]`

### WikiSQL — aggressive right-tail completion

`[30.0, 33.469737532846, 37.340777683921, 41.65953427844, 46.477789268, 51.853313596896, 57.85055987655, 64.541435173211, 72.006163174686, 80.334246073468, 89.625537699269, 99.991440767825, 111.556242601009, 124.458605333543, 138.853228473926, 154.912703753689, 172.829584504651, 192.818694376052, 215.119703073085, 240.0]`

### Graph Coloring — left/right boundary completion

Left 10 values:

`[0.005, 0.0068, 0.0093, 0.0126, 0.0171, 0.0233, 0.0317, 0.0432, 0.0588, 0.08]`

Right 10 values:

`[35.0, 42.7, 52.1, 63.6, 77.6, 94.7, 115.6, 141.0, 172.1, 210.0]`

The left branch probes the approach to the uncontrolled-negative regime. The right branch probes return toward Positive-only.

## Frozen controls and metrics

- Exp tuning seed offset: `4000` for every active task.
- Positive-only: historical/previously confirmed baselines only; zero new cells in this experiment.
- Training horizon: `1200` optimizer updates; no early stopping.
- Primary response metric: validation late-window Pass@8 mean over updates `[800, 900, 1000, 1100, 1200]`.
- Secondary response metric: validation late-window Greedy mean.
- Terminal metrics and valid/structure diagnostics remain required.
- Task-performance events, valid/structure-boundary events, and NaN/Inf numerical failures remain separately reported.
- Fixed 1200 updates are not convergence.

## Implementation contract

The existing scientific trainer/taper/loss modules must not be changed. The successor uses the current cold-start orchestration path and adds only the minimum experiment-ID/config plumbing needed to expand this frozen 140-cell matrix.

`configs/e8_multitask_exp_lambda_curve_completion.yaml` is the runtime authority for the new grids, expected cell count, hashes, and scheduler audit geometry. The existing generic `scripts/bootstrap_e8_multitask_exp_coldstart.sh` remains unchanged. The shared `scripts/run_e8_multitask_exp_coldstart.sh` remains the entrypoint, with its setup regression gate scoped to the paper Linear/extension path used by this cold-start family so unrelated Reciprocal/AsymRE experiment tests cannot block this pilot. No experiment-specific launcher is introduced.

Before launch, acceptance must verify:

1. the new YAML expands to exactly `140` unique Exp cells;
2. each active task has exactly `20` cells and seed `4000`;
3. Countdown and Spiral Matrix instantiate zero cells;
4. no Positive-only cells are instantiated;
5. each Exp cell carries configured `lambda_value` directly with no rho-derived transport;
6. historical cold-start and Lambda-Completion-01 configs still validate unchanged;
7. canonical paper trainer/taper/loss Git blob identities remain unchanged;
8. the dynamic scheduler remains 16 slots with nominal `9`-batch audit geometry for 140 cells;
9. no test partition is accessed.

## Expected outputs and result status

Default Run ID: `E8_MULTITASK_EXP_LAMBDA_CURVE_COMPLETION_02`.

The reviewed-branch pilot launch below is fresh-machine safe at the repository level. It uses a dedicated source checkout under `$HOME`, clones the canonical experiment branch when absent, otherwise requires the dedicated checkout to be clean and fast-forwards it only. Exact equality with `origin/dev/e8-multitask-lambda-curve-completion-02` is checked before the full bootstrap, so an unrelated `main` checkout, detached HEAD, or another project working tree cannot silently become the experiment source. The host must already provide `bash`, `git`, Python/pip, NVIDIA drivers, and network access; the bootstrap performs the remaining software, GPU, disk, source-lock, liveness, scheduler, recovery, and packaging gates.

```bash
set -euo pipefail
REPO_URL="https://github.com/easonhuo/drpo.git"
BRANCH="dev/e8-multitask-lambda-curve-completion-02"
SOURCE_ROOT="${HOME}/drpo-e8-lambda-curve-completion-02-source"
BOOTSTRAP_ROOT="${HOME}/drpo-e8-multitask-exp-lambda-curve-completion-full"

command -v git >/dev/null 2>&1 || {
  echo "ERROR: git is required before launch." >&2
  exit 1
}

if [[ -e "${SOURCE_ROOT}" && ! -d "${SOURCE_ROOT}/.git" ]]; then
  echo "ERROR: ${SOURCE_ROOT} exists but is not the dedicated git checkout." >&2
  exit 1
fi

if [[ ! -d "${SOURCE_ROOT}/.git" ]]; then
  git clone --branch "${BRANCH}" --single-branch "${REPO_URL}" "${SOURCE_ROOT}"
else
  test -z "$(git -C "${SOURCE_ROOT}" status --porcelain)" || {
    echo "ERROR: dedicated source checkout is dirty; resolve it before launch." >&2
    exit 1
  }
  git -C "${SOURCE_ROOT}" fetch origin "${BRANCH}"
  if git -C "${SOURCE_ROOT}" show-ref --verify --quiet "refs/heads/${BRANCH}"; then
    git -C "${SOURCE_ROOT}" switch "${BRANCH}"
  else
    git -C "${SOURCE_ROOT}" switch --track -c "${BRANCH}" "origin/${BRANCH}"
  fi
  git -C "${SOURCE_ROOT}" merge --ff-only "origin/${BRANCH}"
fi

test "$(git -C "${SOURCE_ROOT}" rev-parse HEAD)" = "$(git -C "${SOURCE_ROOT}" rev-parse "origin/${BRANCH}")" || {
  echo "ERROR: dedicated source commit does not exactly match origin/${BRANCH}." >&2
  exit 1
}

E8_COLDSTART_EXISTING_REPO="${SOURCE_ROOT}" \
E8_COLDSTART_BOOTSTRAP_ROOT="${BOOTSTRAP_ROOT}" \
E8_COLDSTART_TARGET_REF="refs/heads/${BRANCH}" \
E8_COLDSTART_EXPERIMENT_ID="EXT-C-E8-MULTITASK-EXP-LAMBDA-CURVE-COMPLETION-02" \
E8_COLDSTART_CONFIG="configs/e8_multitask_exp_lambda_curve_completion.yaml" \
E8_COLDSTART_RUN_ID="E8_MULTITASK_EXP_LAMBDA_CURVE_COMPLETION_02" \
E8_COLDSTART_RUN_CLASS="pilot" \
E8_COLDSTART_REQUIRE_ORIGIN_MAIN="0" \
bash "${SOURCE_ROOT}/scripts/bootstrap_e8_multitask_exp_coldstart.sh" full
```

Outputs use the existing E8 cold-start runtime/package layout and must include raw per-cell curves, aggregate response tables, logs, run manifest, `RUN_COMPLETE.json`, terminal audit, source/config provenance, checksums, and failure inventory when applicable.

Initial scientific result status remains `not_run`. A successful finite-horizon execution may support response-shape analysis only; it does not by itself establish significance, convergence, steady state, or method ranking.
