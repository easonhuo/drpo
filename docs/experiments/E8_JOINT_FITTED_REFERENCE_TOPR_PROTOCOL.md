# E8 Joint Fitted-Reference TOPR development protocol

## Status and identity

- Candidate experiment ID: `EXT-C-E8-ORACLE-OFFLINE-V2-JOINT-FITTED-REFERENCE-TOPR-0.5B-01`.
- Current state: `dev_code_first_unregistered`.
- Result state: not run. Static checks and a later liveness run are not scientific evidence.
- Environment role: Countdown external validity only; it does not replace C-U1 or D-U1 controlled mechanism identification.

This method is **Joint Fitted-Reference TOPR**. It is not canonical frozen-behavior TOPR, because the frozen E8 V2 bank was not sampled from a logged behavior policy and the reference policy is fitted jointly during training.

## Question

Does behavior-relative negative control based on the full-completion likelihood ratio provide a viable training signal on the frozen model-independent E8 V2 bank when the denominator is a jointly fitted, branch-balanced bank-density model?

The candidate pilot cannot establish convergence, steady state, statistical significance, or a formal ranking against Positive-only, Reciprocal-Quadratic, Exp, AsymRE, or any other method.

## Frozen implementation contract

One frozen Qwen2.5-0.5B-Instruct backbone carries two LoRA adapters:

- policy adapter: `default`;
- fitted reference adapter: `reference`.

The reference adapter is created from the policy adapter configuration and receives an exact parameter copy before the first forward pass. The implementation must verify at the first microbatch that the maximum absolute full-sequence log-ratio is at most `1e-5`.

For each prompt with one positive completion and `K_x` unique negative completions, the fitted reference objective assigns branch mass

\[
q(y^+\mid x)=0.5,
\qquad
q(y^-_j\mid x)=\frac{0.5}{K_x}.
\]

The reference update is therefore

\[
\mathcal L_\mu
=-\frac{1}{2}\bar\ell_\mu(y^+\mid x)
-\frac{1}{2K_x}\sum_j\bar\ell_\mu(y^-_j\mid x),
\]

where `bar-ell` is the existing mean completion-token log-probability used by the E8 trainer.

The TOPR ratio uses the full summed completion log-probability:

\[
\log r_j
=\sum_t\log\pi(y^-_{j,t}\mid x,y^-_{j,<t})
-\sum_t\log\mu(y^-_{j,t}\mid x,y^-_{j,<t}),
\]

\[
w_j=\exp(\min(\log r_j,0)).
\]

Both the reference denominator and the weight are detached from the policy objective. The policy objective remains

\[
\mathcal J_\pi
=\bar\ell_\pi(y^+\mid x)
-\frac{1}{K_x}\sum_j w_j\bar\ell_\pi(y^-_j\mid x).
\]

The two adapters receive one optimizer update per training step, use the same learning rate and scheduler horizon, and share paired dropout RNG streams within each microbatch. The reference receives only its bank-SFT loss. The policy receives only its TOPR objective.

## Data, evaluation, and reporting boundaries

- Use the unchanged 6000-row model-independent oracle-offline V2 bank.
- Use all unique negatives after the existing first-occurrence expression deduplication.
- Do not use near/far class labels or extreme selection.
- Do not normalize the negative loss by the sum of weights.
- Evaluate only the policy adapter on the structurally disjoint held-out `val.jsonl` split.
- Do not access `test.jsonl`.
- Report the fixed late window and terminal Pass@8; best checkpoint metrics remain supplementary.
- Report task performance, valid-expression/structure diagnostics, and NaN/Inf numerical failure separately.
- Record reference loss, reference positive/negative log-probability, log-ratio quantiles, weight quantiles, clipped-at-one fraction, both gradient norms, and both adapter update norms.

## Development sequence and gates

1. Implement within the existing E8 common/trainer/runtime stack; do not create another Python training stack.
2. Pass Python compilation, configuration-contract tests, pure loss/ratio tests, existing E8 regression tests, and repository exact-head CI.
3. Freeze the implementation SHA.
4. Register the pilot through the normal schema-v3 code-first registration transaction.
5. Run a two-step dual-adapter liveness gate. It is not scientific evidence.
6. Only after registration and liveness may a development pilot be launched under the registered RunSpec and terminal-audit requirements.

No experiment is authorized by this development document alone.
