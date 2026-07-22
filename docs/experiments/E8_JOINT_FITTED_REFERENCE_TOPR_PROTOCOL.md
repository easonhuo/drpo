# E8 Joint Fitted-Reference beta-TOPR response-curve development protocol

## Status and identity

- Candidate experiment ID: `EXT-C-E8-ORACLE-OFFLINE-V2-JOINT-FITTED-REFERENCE-TOPR-0.5B-01`.
- Current state: `dev_code_first_unregistered`.
- Result state: not run. Static checks and a later liveness run are not scientific evidence.
- Environment role: Countdown external validity only; it does not replace C-U1 or D-U1 controlled mechanism identification.

This experiment is a **Joint Fitted-Reference beta-TOPR response curve**. It is not canonical frozen-behavior TOPR, because the frozen E8 V2 bank was not sampled from a logged behavior policy and the reference policy is fitted jointly during training.

The terminology by point is frozen:

- `beta = 1`: original TOPR likelihood-ratio rule used with a jointly fitted reference; it is the ratio-rule anchor but not a canonical TOPR reproduction;
- `beta = 0`: no-ratio-taper boundary control, not a TOPR method point;
- all other beta values: tempered Joint Fitted-Reference beta-TOPR variants.

## Question

How does the strength of behavior-relative negative tapering affect task performance, weight distributions, and training stability on the frozen model-independent E8 V2 bank when the denominator is a jointly fitted, branch-balanced bank-density model?

The candidate pilot is designed to resolve a response-curve trend. It cannot establish convergence, steady state, statistical significance, a best beta, or a formal ranking against Positive-only, Reciprocal-Quadratic, Exp, AsymRE, or any other method.

## Frozen beta response curve

The only scanned scientific variable is

\[
\beta
\in
\{0,\ 0.25,\ 0.5,\ 0.75,\ 1,\ 1.5,\ 2,\ 4\}.
\]

Each beta value is paired with seed offsets `4000` and `5000`, giving

\[
8\ \text{beta points}
\times
2\ \text{seeds}
=
16\ \text{cells}.
\]

Learning rates, optimizer settings, the reference update frequency, the reference target distribution, the bank, seeds, horizon, evaluation split, and evaluation cadence are not scanned.

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

The behavior-relative coordinate uses full summed completion log-probability:

\[
\log r_j
=\sum_t\log\pi(y^-_{j,t}\mid x,y^-_{j,<t})
-\sum_t\log\mu(y^-_{j,t}\mid x,y^-_{j,<t}).
\]

The beta response weight is

\[
w_{\beta,j}
=
\exp\!\left(\beta\min(\log r_j,0)\right)
=
\min\!\left[\left(\frac{\pi}{\mu}\right)^\beta,1\right].
\]

Both the reference denominator and the weight are detached from the policy objective. The policy objective remains

\[
\mathcal J_\pi
=\bar\ell_\pi(y^+\mid x)
-\frac{1}{K_x}\sum_j
w_{\beta,j}\bar\ell_\pi(y^-_j\mid x).
\]

The two adapters receive one optimizer update per training step, use the same learning rate and scheduler horizon, and share paired dropout RNG streams within each microbatch. The reference receives only its bank-SFT loss. The policy receives only its beta-TOPR objective.

## Data, execution, evaluation, and reporting boundaries

- Use the unchanged 6000-row model-independent oracle-offline V2 bank.
- Use all unique negatives after the existing first-occurrence expression deduplication.
- Do not use near/far class labels or extreme selection.
- Do not normalize the negative loss by the sum of weights.
- Keep fixed 1200-step training with no early stopping.
- Run on GPU `0,1`, one cell per GPU, for eight full waves after liveness.
- Run the representative two-step liveness at `beta = 1`.
- Evaluate only the policy adapter on the structurally disjoint held-out `val.jsonl` split.
- Do not access `test.jsonl`.
- Report the fixed late window and terminal Pass@8; best checkpoint metrics remain supplementary.
- Plot or tabulate the declared beta response curve without selecting a best beta as a formal result.
- Report task performance, valid-expression/structure diagnostics, and NaN/Inf numerical failure separately.
- Record beta, reference loss, reference positive/negative log-probability, log-ratio quantiles, weight quantiles, clipped-at-one fraction, both gradient norms, and both adapter update norms.

## Development sequence and gates

1. Implement within the existing E8 common/trainer/runtime stack; do not create another Python training stack.
2. Pass Python compilation, configuration-contract tests, pure beta ratio tests, existing E8 regression tests, RunSpec validation, and repository exact-head CI.
3. Freeze the implementation SHA.
4. Bind the deferred-registration RunSpec to that exact implementation SHA.
5. Register the pilot through the normal schema-v3 code-first registration transaction.
6. Run the two-step `beta = 1` dual-adapter liveness gate. It is not scientific evidence.
7. Only after registration and liveness may the 16-cell development pilot be launched under the registered RunSpec and terminal-audit requirements.

No experiment is authorized by this development document alone.
