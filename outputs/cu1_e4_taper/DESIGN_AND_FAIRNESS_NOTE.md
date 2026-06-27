# C-U1-E4-TAPER-01 Design and Fairness Note

## Why equal reward and advantage are used

C-U1 has a continuous two-dimensional action space and a smooth reward function. For every state, the negative candidate set is constructed by selecting eight evenly spaced points from a continuous equal-reward contour centered at the hidden optimum. Their reward and fixed advantage are equal by design. This removes sample-quality magnitude as a confound when studying policy-relative distance and Gaussian score geometry.

The formal dataset is finite, as any offline dataset is finite, but the underlying action/reward space is not discrete or discontinuous. Robustness to continuous angle sampling, random phase, denser contours, thin-annulus noise, and approximate reward-bin matching remains a future registered extension.

## Why equal-quality negatives can have different utility

A negative advantage determines the sign and magnitude of repulsion, but not whether the repulsion direction helps the true task. The useful quantity is the alignment between the negative update direction and the hidden-optimum improvement direction. The registered geometry intentionally includes a local negative whose repulsion points toward the hidden optimum and far-side negatives whose repulsion can point away from it. Thus quality magnitude is distance-matched while directional utility is not.

This supports a controlled informativeness-amplification mismatch: local negatives may retain boundary information, whereas remote negatives can have lower directional relevance while exerting larger Gaussian score influence. It does not support a universal claim that all near negatives are useful or all far negatives are harmful.

## What the current family comparison matches

The three taper families share the same standardized distance, fixed advantages, actor initialization, minibatch stream, negative alpha, and reference-point weight `w(d_ref)=rho`. They do not share the same reference-point slope, average near-bin retention, total negative-gradient norm, or cumulative optimizer update.

Therefore the completed experiment supports an anchor-normalized mechanism-order result. It does not establish the ranking of independently best-tuned families.

## What the analytic quadratic boundary means

For the learnable Gaussian log-scale output branch, the far-field score grows as `Theta(d^2)` before a support boundary. A reciprocal-polynomial taper with tail `Theta(d^-p)` gives weighted order `Theta(d^(2-p))`. Any finite coefficient changes constants but not the tail exponent: `p<2` remains unbounded, `p=2` is bounded, and `p>2` tends to zero. Exponential taper dominates every finite polynomial order.

This is an asymptotic influence-bound result, not a task-reward theorem. Heavy attenuation can approach positive-only and discard useful local negative information.

## Required evidence before stronger claims

Stronger family-ranking or Distance-versus-Global claims require separately registered experiments with matched near-negative retention, matched stepwise or cumulative negative-gradient budgets, equal hyperparameter-search budgets, untouched confirmatory seeds, Pareto reporting, and long-run terminal audits with 2x continuation. These requirements are documented but not authorized for execution by this update.
