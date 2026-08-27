# E8 Multitask Lambda Curve-Completion Protocol

Experiment ID: `EXT-C-E8-MULTITASK-EXP-LAMBDA-COMPLETION-01`

Status: `not_run`.

Implementation base for this repair: `8bdd07590f155ad26bc8cfbd641d40647eab57d2`, whose repository tree is byte-identical to `1723d0c507b2309a1a352c2459165b86b9625c9d`.

This is a curve-completion / response-shape successor to the closed 208-cell pilot `EXT-C-E8-MULTITASK-EXP-COLDSTART-01`. It is not a method-ranking, convergence, or steady-state experiment.

## Claim and scope

Complete the high-lambda response tails for the eight non-Countdown tasks while preserving the historical 208-cell results for later concatenation. Countdown contributes zero new scientific cells. The successor scientific interface is lambda-only: manuscript scale `c = 1`, no rho-derived transport, and the same cold-start execution implementation for the canonical paper trainer/taper/loss, negative-bank construction and 16-negative consumer, task prompt/verifier interface, current-policy sequence-surprisal computation, optimizer, data, evaluation, recovery and terminal audit.

Exp remains single-seed response-shape localization. The two fresh Positive-only seeds are baseline confirmation only. Test partitions remain forbidden.

## Frozen workload

- Total new cells: `199`.
- New Exp cells: `183`.
- New Positive-only cells: `16`.
- Countdown: `0` new scientific cells.
- Exp seed offset for every non-Countdown task: `4000`.
- Fresh Positive-only seed offsets for every non-Countdown task: `8000, 9000`.
- Spiral Matrix: `8 Exp + 2 Positive-only = 10` cells.
- Each of the other seven non-Countdown tasks: `25 Exp + 2 Positive-only = 27` cells.
- With 16 concurrent slots, nominal audit geometry is `12 x 16 + 1 x 7 = 199` cells. This is not a hard-wave scheduler: the cold-start shared dynamic slot queue remains unchanged and an idle slot immediately pulls the next pending cell.

Countdown retains the six historical paper lambda sentinels in configuration only as an engineering/runtime identity anchor. `countdown_seed_offsets` is empty, so those sentinels create no new Countdown cells.

## Frozen lambda construction

For each task, the historical maximum lambda is retained as the left anchor but is not rerun. New values are geometric tail extensions to exactly five times that historical maximum:

$$
\lambda_i=\lambda_{\max}^{\mathrm{old}}\cdot 5^{i/n},\qquad i=1,\ldots,n.
$$

Here `n=8` for Spiral Matrix and `n=25` for the other seven tasks. Runtime consumes the explicitly frozen values from `configs/e8_multitask_exp_lambda_completion.yaml`; it does not regenerate or round them differently.

- `word_sorting` (25): `[2.452938367079, 2.616046362039, 2.790000213697, 2.975521116668, 3.173378221361, 3.384391822795, 3.609436761463, 3.849446050321, 4.105414742971, 4.378404059047, 4.669545783918, 4.980046960959, 5.311194895823, 5.664362493478, 6.041013950129, 6.442710823622, 6.871118507504, 7.328013135568, 7.815288945519, 8.334966132293, 8.889199223569, 9.480286012219, 10.11067708272, 10.782985971023, 11.5]`
- `spiral_matrix` (8): `[8.447110861078, 10.329503437427, 12.631376930953, 15.446210375777, 18.888314098846, 23.097471859905, 28.244617467034, 34.538776395]`
- `mini_sudoku` (25): `[8.531959537667, 9.099291694048, 9.704348569381, 10.349638666672, 11.03783729169, 11.771797644506, 12.554562648566, 13.389377566334, 14.279703453813, 15.229231509727, 16.241898378844, 17.3219024729, 18.473721376774, 19.702130412096, 21.01222243523, 22.40942895173, 23.899542634798, 25.488741341105, 27.183613723546, 28.991186547108, 30.91895382111, 32.974907868588, 35.167572461636, 37.506038160081, 40.0]`
- `maze` (25): `[2.21771388682, 2.365180643558, 2.522453193764, 2.690183573107, 2.86906717433, 3.059845630279, 3.263309888631, 3.480303491084, 3.711726070591, 3.958537081148, 4.221759775589, 4.502485447884, 4.801877957535, 5.121178554813, 5.461711026854, 5.824887185944, 6.212212722743, 6.625293448728, 7.065841953715, 7.535684706081, 8.036769625112, 8.571174156876, 9.141113887095, 9.748951726738, 10.397207708399]`
- `word_ladder` (25): same frozen values as `word_sorting`.
- `knights_knaves` (25): same frozen values as `maze`.
- `graph_color` (25): `[6.39896965325, 6.824468770536, 7.278261427036, 7.762229000004, 8.278377968767, 8.828848233379, 9.415921986425, 10.04203317475, 10.70977759036, 11.421923632295, 12.181423784133, 12.991426854675, 13.855291032581, 14.776597809072, 15.759166826423, 16.807071713798, 17.924656976098, 19.116556005829, 20.38771029266, 21.743389910331, 23.189215365832, 24.731180901441, 26.375679346227, 28.12952862006, 30.0]`
- `wikisql` (25): `[5.545773699484, 5.914539601131, 6.307826570098, 6.727265133337, 7.174594239598, 7.651668468929, 8.160465721568, 8.703095418117, 9.281807244979, 9.899000481323, 10.557233946249, 11.259236607385, 12.007918894903, 12.806384767862, 13.6579445829, 14.566128818625, 15.534702712619, 16.567681871718, 17.669348920305, 18.84427125562, 20.097319983721, 21.433690114582, 22.858922100063, 24.378924804052, 26.0]`

## Configuration authority

The successor YAML is the runtime authority for the new lambda grid, seed offsets, expected cell count and nominal group count. These successor values are not duplicated as lambda-completion constants in Python.

Successor Exp cells are instantiated with `rho=None` and `lambda_value=<configured lambda>`. The historical optional rho field remains for predecessor provenance only; the successor coefficient path sends the configured `lambda_value` to the unchanged paper trainer.

The negative-bank selector remains exactly the cold-start execution selector: `source_p0_error_class_sequence_then_within_class_reference_rank_spread`. The scheduler remains the cold-start `dynamic_slot_queue` with `wave_barriers: false`.

## Historical concatenation

The historical curve anchor is `experiments/results/e8_multitask_exp_coldstart_20260820_02/CURVE_ANCHOR.csv`. Historical rows are immutable. New outputs remain separately identifiable and may be concatenated with that anchor only for plotting/response-curve analysis; they do not overwrite or relabel old cells.

## Scientific invariants

The following remain frozen from the cold-start execution tree: Qwen2.5-0.5B-Instruct revision, fresh LoRA initialization, LoRA rank/alpha/dropout, 1200 optimizer updates, learning rate, batch/accumulation, source-P0 error-class-preserving negative-bank selection, 16-negative consumer, transfer system prompt, current-policy sequence-surprisal adapter, detached paper-linear exponential taper, task runtime overrides, evaluation cadence, Pass@8/Greedy definitions, no early stopping, test prohibition, shared dynamic-slot recovery semantics and terminal audit taxonomy.

Task-performance collapse, support/structure-boundary events, and NaN/Inf numerical failure remain separately reported. A fixed 1200-update horizon is not convergence.

## Implementation acceptance

Before any scientific launch, the repaired successor implementation must satisfy all of the following:

1. Expanding the successor YAML produces the YAML-declared unique cell count and task/seed geometry.
2. Every successor Exp cell has `rho=None` and carries its configured lambda directly.
3. Historical-wrapper and lambda-only transport are exactly equivalent at the taper/loss/gradient interface for the same lambda.
4. Canonical paper trainer/taper/loss/config source identities remain unchanged.
5. The historical cold-start config still validates and still produces exactly 208 cells; its nominal audit geometry remains 13 x 16 while runtime scheduling remains dynamic refill with no hard wave barrier.
6. The cold-start selector, task prompt/current-policy surprisal interface, task runtime overrides, scheduler and recovery behavior are unchanged from the `1723d0c...` tree.
7. No successor scientific hyperparameter is hard-coded into Python.
