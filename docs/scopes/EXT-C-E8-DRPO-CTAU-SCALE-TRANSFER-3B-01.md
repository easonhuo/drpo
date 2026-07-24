# Scope: EXT-C-E8-DRPO-CTAU-SCALE-TRANSFER-3B-01

## Identity

- Environment: Countdown external validity (`EXT-C`).
- Target model: `Qwen2.5-3B-Instruct`.
- Source model scale: `Qwen2.5-0.5B-Instruct`.
- Result class: `pilot`, initially `not_run`.
- Scientific role: model-scale transfer of a frozen DRPO coefficient set.

## Claim

Test whether four preselected DRPO `(c, tau)` points from the completed 0.5B
validation response surface remain usable when transferred without retuning to
Qwen2.5-3B-Instruct on the same frozen model-independent E8 V2 bank.

This experiment does not identify a universal optimum, does not reopen the 0.5B
coefficient search, and does not establish convergence, significance, or a
formal method ranking. Countdown remains external-validity evidence and does not
replace C-U1 or D-U1 controlled mechanism identification.

## Frozen scientific matrix

| Label | `c` | `tau` |
|---|---:|---:|
| A | 1.609437912 | 0.125 |
| B | 1.897119985 | 0.25 |
| C | 2.995732274 | 0.125 |
| D | 4.605170186 | 0.75 |

Each point runs with seed offsets `4000` and `5000`, producing exactly eight
cells. The two waves may not use different scientific variables.

The implementation-facing taper is

```text
u = current_sequence_surprisal / 2
weight = exp(-c * max(u - tau, 0))
```

with detached weights and global `alpha=1`. Every first-occurrence unique
negative in each bank row participates; no near/far class selection, hidden
negative scale, weight-sum normalization, gradient-budget matching, TOPR,
AsymRE, Reciprocal, SBRC, or Hybrid is allowed.

## Frozen training and evaluation

- Initialization: pretrained base plus fresh LoRA; no Countdown SFT warm start.
- Bank: frozen model-independent E8 V2 bank.
- Held-out split: `val.jsonl`, structurally disjoint from training and excluded
  from the training loss.
- Horizon: fixed 1200 optimizer steps, no early stopping.
- Evaluation: Greedy and Pass@8 every 100 steps; Pass@64 every 200 steps.
- Primary reporting: mean Pass@8 at steps `800,900,1000,1100,1200`.
- Secondary reporting: terminal Pass@8.
- Best checkpoint: supplementary only.
- Separate `test.jsonl`: not accessed in this pilot.

The 3B base config deliberately preserves the 0.5B optimizer, learning-rate,
batch, accumulation, LoRA, horizon, and evaluation settings. Retuning any of
those fields requires a new experiment ID.

## Runtime contract

The approved execution allocation is exactly:

```text
4 GPUs
1 slot per GPU
4 concurrent cells
8 total cells
2 full waves
```

The allocation changes wall-clock placement only. It does not change the
scientific matrix. A two-step real-model liveness on point A must pass before
the full sweep. Runtime identity must record the exact model config, code SHA,
input hashes, cell `(label,c,tau,seed)`, and one-slot allocation.

## Reporting separation

The terminal audit must report separately:

1. task-performance degradation or collapse;
2. valid-expression or structural-boundary diagnostics;
3. NaN/Inf numerical failure;
4. best-checkpoint versus late-window and terminal metrics.

A fixed 1200-step endpoint is not convergence or steady state.

## Approved new Python path

The repository owner explicitly approved:

`src/drpo/countdown_e8_drpo_ctau_scale_transfer_3b.py`

Its responsibility is limited to the frozen 3B scale-transfer profile, model
identity gate, existing-stack runtime adapter, one-slot resource contract, and
run identity. It must not become a generic replacement for the existing E8
profiles.
