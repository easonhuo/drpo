# EXT-C-E8-ORACLE-OFFLINE-V2-SHARED-SFT-CANONICAL-DPO-BETA-SCAN-0.5B-01

## Status

- implementation state: code-first development scope
- registration state: `dev_code_first_unregistered`
- result status: `not run`
- scientific evidence: none
- environment role: Countdown external-validity baseline only
- implementation dependency: stacked on the unmerged cold-start canonical-DPO implementation in PR #268

This scope authorizes implementation, static validation, and real-model liveness only. It does not authorize an unregistered formal launch, method-ranking claim, convergence claim, or test-split access. After implementation freeze, the normal schema-v3 pilot-registration transaction and reviewed RunSpec are still required.

## Claim under development

The completed cold-start canonical-DPO pilot showed that preference separation can succeed while Countdown task performance and valid-expression support collapse. This successor asks a narrower question: when every DPO cell starts from the same persisted V2 oracle-SFT checkpoint that already has basic task ability, does canonical DPO preserve, improve, or destroy that ability across the same beta response grid?

This experiment does not test the DRPO/TOPR remoteness mechanism itself and cannot establish a universal DPO ranking.

## Frozen two-stage protocol

### Stage 1: one shared V2 oracle-SFT run

- base model: `Qwen2.5-0.5B-Instruct`
- parameterization: LoRA; frozen foundation-model backbone
- training corpus: V2 bank prompts with oracle-positive completions only
- held-out evaluation: structurally disjoint V2 `val.jsonl`
- SFT runs: exactly one
- SFT seed: `2026070700`
- evaluation seed: `2026070790`
- epochs: exactly one
- learning rate: `2e-4`
- micro batch: `2`
- gradient accumulation: `32`
- warmup ratio: `0.05`
- maximum gradient norm: `1.0`
- dtype: BF16
- adaptive metric stopping: forbidden
- test split: forbidden
- persisted checkpoint: `shared_sft/epoch_1_adapter`

The SFT adapter is written once and identity-bound by `SFT_COMPLETE.json` and `SFT_WARMSTART_GATE.json`. A matching completed gate allows reuse; a stale or mismatched non-empty SFT directory fails closed. The adapter is never merged into or duplicated as a full 0.5B model checkpoint.

### Stage 2: shared-checkpoint canonical DPO

All 16 DPO cells load the exact same persisted `epoch_1_adapter` read-only from disk. In each independent cell process:

1. the shared SFT adapter is loaded as trainable policy adapter `default`;
2. it is copied exactly to adapter `reference` before update 1;
3. `reference` is permanently frozen;
4. only the cell-local policy adapter and optimizer state evolve.

The 16 cells therefore share the same foundation-model files and the same SFT checkpoint, but they do not share one mutable in-memory model object or optimizer. This avoids parameter/optimizer cross-cell interference while preventing 16 redundant SFT runs or 16 full-model copies.

## Frozen DPO method and matrix

- chosen completion: oracle completion
- rejected completions: all first-occurrence unique verifier-wrong completions
- pair aggregation: mean over unique rejected completions within prompt, then mean prompts
- sequence score: summed completion-token log probability
- loss: canonical sigmoid DPO, zero label smoothing
- beta grid: `[0.01, 0.03, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0]`
- development seed offsets: `[4000, 5000]`
- matrix: 8 beta points × 2 seeds = 16 cells
- DPO horizon: fixed 1200 optimizer updates
- early stopping: forbidden
- evaluation cadence: Greedy/Pass@8 every 100 updates; Pass@64 every 200 updates
- execution layout: GPU 0–1, two cells per GPU, four full waves
- test split: forbidden

## Persistence and provenance requirements

The shared SFT gate records hashes for:

- V2 training bank;
- V2 held-out validation split;
- base-model config/index metadata when available;
- `adapter_config.json`;
- the single LoRA weight file.

Every DPO `run_identity` must include the same shared-adapter fingerprint. A changed or missing adapter invalidates resume. DPO cells save their own policy/reference LoRA checkpoints, metrics, diagnostics, and terminal audit; they do not overwrite the shared SFT adapter.

## Direct entry point

```bash
E8_WARM_DPO_EXPECTED_COMMIT=<implementation-sha> \
E8_WARM_DPO_FORMAL_RUN_AUTHORIZED=1 \
bash scripts/run_countdown_e8_shared_sft_canonical_dpo_beta_scan.sh full
```

Available modes are `preflight`, `sft`, `liveness`, `run`, and `full`.

## Required gates before the 16-cell matrix

1. config/profile preflight;
2. identity-bound one-epoch SFT completion;
3. shared-adapter file/hash gate;
4. representative beta `0.1` two-step DPO liveness;
5. fresh-process reload of both policy and frozen reference adapters;
6. exact implementation SHA freeze;
7. schema-v3 pilot registration and reviewed RunSpec;
8. explicit launch authorization.

Task-performance degradation, valid-structure/support behavior, and NaN/Inf numerical failure remain separately reported. Fixed 1200-step DPO training is finite-step evidence, not convergence or steady state.
