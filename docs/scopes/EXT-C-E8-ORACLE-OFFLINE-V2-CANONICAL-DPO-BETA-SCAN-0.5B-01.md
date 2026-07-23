# EXT-C-E8-ORACLE-OFFLINE-V2-CANONICAL-DPO-BETA-SCAN-0.5B-01

## Status

- implementation state: code-first development scope
- registration state: `dev_code_first_unregistered`
- result status: `not run`
- scientific evidence: none
- environment role: Countdown external-validity baseline only

This scope authorizes implementation and static/liveness validation only. It does not authorize a formal multi-seed launch, method-ranking claim, convergence claim, or test-split access. Formal execution still requires the repository's schema-v3 pilot-registration transaction after the implementation SHA is frozen.

## Claim under development

On the frozen model-independent E8 V2 oracle-offline bank, measure the development response of canonical sigmoid DPO when every oracle completion is paired with every unique verifier-wrong completion from the same prompt. The reference policy is an exact frozen copy of the policy adapter at initialization. The pilot varies only DPO beta and asks whether ordinary preference optimization is a competitive external-validity baseline; it does not test the DRPO/TOPR remoteness mechanism itself.

## Frozen method definition

- model: `Qwen2.5-0.5B-Instruct`
- initialization: pretrained base plus fresh LoRA
- shared backbone: frozen
- policy adapter: `default`, trainable
- reference adapter: `reference`, copied exactly from `default` before update 1 and permanently frozen
- dropout: disabled for both policy and reference likelihood evaluation
- chosen completion: oracle completion
- rejected completions: all first-occurrence unique verifier-wrong bank expressions
- pair aggregation: average DPO loss over unique rejected completions within a prompt, then average prompts
- sequence score: summed completion-token log probability
- loss: canonical sigmoid DPO with zero label smoothing
- beta development grid: `[0.01, 0.03, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0]`
- matrix size: 8 beta points × 2 paired development seeds = 16 cells

For prompt `x`, chosen completion `y+`, and each rejected completion `y-_j`, the pair margin is

```text
(log pi_policy(y+|x) - log pi_policy(y-_j|x))
- (log pi_reference(y+|x) - log pi_reference(y-_j|x))
```

and the pair loss is `-log sigmoid(beta * margin)`.

## Frozen development protocol

- source bank: `EXT-C-E8-ORACLE-OFFLINE-BANK-V2-0.5B-01`
- use all unique negatives: yes
- hard-negative or near/far selection: forbidden
- development seed offsets: `[4000, 5000]`
- fixed horizon: 1200 optimizer updates
- early stopping: forbidden
- evaluation cadence: Greedy/Pass@8 every 100 updates; Pass@64 every 200 updates
- held-out evaluation: structurally disjoint `val.jsonl`
- separate test split: forbidden during development
- paper-facing checkpoint policy: late-window and terminal; best checkpoint supplementary only

## Required diagnostics

- DPO loss and beta
- policy/reference chosen and rejected summed log probabilities
- reference-relative pair-margin mean and quantiles
- pair preference accuracy and logit-saturation fraction
- policy raw gradient norm and optimizer update norm
- prompt unique-negative and raw-bank multiplicity
- held-out Greedy, Pass@8, Pass@64, and valid rate
- task-performance trajectory, support/validity behavior, and NaN/Inf failure reported separately

## Planned code entry point

```text
python3 scripts/run_countdown_e8_oracle_offline_v2_alpha1_highc_scan_auto.py ... \
  --grid_config configs/countdown_e8_oracle_offline_v2_canonical_dpo_beta_scan_0p5b.yaml
```

The implementation extends existing E8 profile/runtime files. It does not create a new Python path.

## Remaining gates

1. static validation and repository CI;
2. two-step liveness on one representative beta cell;
3. terminal checkpoint reload of both adapters;
4. freeze implementation SHA;
5. create and review `DEV_PILOT_REGISTRATION_SPEC.yaml`;
6. complete the normal schema-v3 registration transaction before any formal run.
