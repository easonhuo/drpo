# DRPO paper rewrite v0.9.3: publication-quality outline

Parent: v0.9.2 remains preserved in Git history. This revision strengthens the first four chapters with explicit definition, theorem, citation, appendix, boundary, and word-budget obligations while preserving stable paragraph IDs and manuscript order.

This is one DRPO manuscript being rewritten, not an old paper followed by a sequel.

# Abstract

<!-- MANUSCRIPT:BEGIN ABSTRACT-P01 -->
## [ABSTRACT-P01] Paper Summary

**Claim:** Negative feedback is useful for policy improvement, but historical negative actions can become far-field and exert excessive influence; DRPO controls that transition.

**Reader question:** What is the problem, missing link, theory, method, evidence, and implication in one compact sequence?

**Role:** Summarize the full paper without fixed-advantage scope language, split-manuscript framing, or unverified numerical claims.

**Required evidence:**
- Theorem 1
- source-isolation protocols
- targeted interventions
- external tasks marked TBD until formal closure

**Must include:**
- Negative feedback as a resource
- badness--distance isolation
- stable extrapolation to equilibrium loss
- DRPO attenuation of the far-field component
- controlled and external evidence

**Must avoid:**
- split-manuscript framing
- finished external numbers before formal closure
<!-- MANUSCRIPT:END ABSTRACT-P01 -->

# Introduction

<!-- MANUSCRIPT:BEGIN INTRO-P01 -->
## [INTRO-P01] Negative Feedback as a Policy-Improvement Resource

**Claim:** Off-policy policy optimization needs both attraction toward successful behavior and suppression of known failures, but the latter must remain dynamically controlled.

**Reader question:** Why is negative feedback a necessary policy-improvement signal rather than disposable noise?

**Role:** Establish the broad off-policy setting, the complementary roles of positive and negative feedback, and the paper's governing question.

**Required evidence:**
- offline-RL and off-policy policy-optimization literature
- advantage-weighted and critic-regularized actor fitting
- Theorem 3 stable-extrapolation regime

**Must include:**
- off-policy reuse across offline, replay, recommendation, and post-training settings
- positive attraction toward observed successes
- negative suppression of known failures and competing modes
- Positive-only as a stable but potentially limited reference
- central question of preserving useful negative feedback without excessive repulsion

**Must avoid:**
<!-- MANUSCRIPT:END INTRO-P01 -->

<!-- MANUSCRIPT:BEGIN INTRO-P02 -->
## [INTRO-P02] Historical Reuse Turns Local Feedback into Persistent Repulsion

**Claim:** Repeated reuse makes negative influence learner-relative: as the current policy moves away from a fixed historical action, the same label can become persistent or self-amplifying far-field repulsion.

**Reader question:** How can locally informative negative feedback become dynamically dangerous?

**Role:** Introduce historical reuse as the temporal mechanism and distinguish Gaussian amplification from categorical persistent suppression.

**Required evidence:**
- Gaussian score identity
- repeated-reuse theorem
- categorical log-odds dynamics

**Must include:**
- fixed historical action reused while learner changes
- initially local negative information
- learner-relative remoteness increases after repulsion
- Gaussian score growth with standardized distance
- categorical bounded score with persistent support suppression
- useful-local to destructive-far-field transition

**Must avoid:**
<!-- MANUSCRIPT:END INTRO-P02 -->

<!-- MANUSCRIPT:BEGIN INTRO-P03 -->
## [INTRO-P03] Existing Controls and the Missing Identification Link

**Claim:** Existing controls stabilize off-policy learning, but the causal role of learner-relative remoteness remains unidentified when quality, advantage magnitude, rarity, and distance are entangled.

**Reader question:** What do existing controls solve, and what identification gap remains?

**Role:** Acknowledge established stabilization methods, isolate the missing dynamic claim, and motivate matched quality--distance controls.

**Required evidence:**
- related-work synthesis
- C-U1 E1 source isolation
- C-U1 E3 targeted intervention
- D-U1 common/rare analogue

**Must include:**
- positive-only filtering, global scaling, clipping, support constraints, rarity controls, and quality filtering
- their established stabilization value
- lack of a useful-local to destructive-far-field explanation
- realistic confounding among reward, advantage, rarity, and distance
- matched controls holding quality and coefficient fixed
- source and causal interventions as separate responsibilities

**Must avoid:**
- dropping the existing-method gap
- claiming distance is the only factor
<!-- MANUSCRIPT:END INTRO-P03 -->

<!-- MANUSCRIPT:BEGIN INTRO-P04 -->
## [INTRO-P04] Repulsive Dynamics Explains Stable Extrapolation and Equilibrium Loss

**Claim:** Repulsive Dynamics unifies stable extrapolation, persistent drift, and instability through the aggregate balance of positive and negative update masses and moments.

**Reader question:** What theoretical structure connects beneficial and harmful negative updates?

**Role:** Preview the static score relation, repeated-reuse asymmetry, aggregate equilibrium theorem, and policy-family manifestations.

**Required evidence:**
- Proposition on remoteness and score response
- reuse theorem
- aggregate equilibrium theorem
- policy-family runaway theorem

**Must include:**
- learner-relative remoteness and score response
- positive self-attenuation and negative self-amplification
- positive-dominant stable extrapolation
- critical persistent drift
- negative-dominant instability
- Gaussian unbounded scores versus categorical bounded boundary degeneration

**Must avoid:**
<!-- MANUSCRIPT:END INTRO-P04 -->

<!-- MANUSCRIPT:BEGIN INTRO-P05 -->
## [INTRO-P05] DRPO Controls the Destabilizing Far-Field Term

**Claim:** DRPO preserves informative local negative feedback while attenuating the learner-relative far-field tail that drives the unstable aggregate regime.

**Reader question:** How does the method intervene on the mechanism rather than merely discard negative data?

**Role:** Introduce the method target, local-preservation principle, tail guarantee, and distinction between quality selection and remoteness control.

**Required evidence:**
- DRPO objective
- vanishing weighted far-field proposition
- controlled global and selective controls

**Must include:**
- negative-term reweighting in the empirical actor field
- learner-relative distance or surprisal
- near-field weights remain substantial
- far-field weights decay nonlinearly
- finite-order score-growth tail guarantee
- quality selection and remoteness control as separate axes

**Must avoid:**
- original formulation versus revised formulation
- old paper followed by a new algorithm
- claim that quality hard filtering is mathematically identical to distance tapering
<!-- MANUSCRIPT:END INTRO-P05 -->

<!-- MANUSCRIPT:BEGIN INTRO-P06 -->
## [INTRO-P06] Evidence Chain and Contributions

**Claim:** The evidence chain assigns occurrence, controlled source identification, causal transmission, phase testing, and external validity to distinct experiments and reports terminal outcomes with a separated failure taxonomy.

**Reader question:** How do the experiments jointly support the paper without conflating controlled mechanisms and external validity?

**Role:** State the four research questions, environment responsibilities, reporting discipline, and contributions.

**Required evidence:**
- environment responsibility table
- registered controlled experiments
- terminal-audited result manifests
- external protocols marked TBD until formal closure

**Must include:**
- RQ1 external occurrence
- RQ2 matched source and causal identification
- RQ3 phase transition and control
- RQ4 external task closure
- C-U1/D-U1 controlled responsibilities
- Hopper/Countdown external-validity responsibilities
- task, support-boundary, and NaN/Inf separation
- four concise contributions

**Must avoid:**
<!-- MANUSCRIPT:END INTRO-P06 -->

# Related Work

<!-- MANUSCRIPT:BEGIN RELATED-P01 -->
## [RELATED-P01] Learning from Negative or Suboptimal Behavior

**Claim:** Prior work shows both that negative updates can be filtered for stability and that failures can carry useful learning signal; the missing object is how the same negative action changes as it becomes remote from the learner.

**Reader question:** What is already known about learning from positive, negative, and suboptimal behavior?

**Role:** Synthesize filtering and negative-information literatures, state the established fact, and isolate the repeated-remoteness gap.

**Required evidence:**
- AWR
- IQL
- CRR
- offline policy-fitting literature

**Must include:**
- advantage-weighted regression and critic-regularized regression
- positive-only and filtering approaches
- evidence that suboptimal behavior can suppress bad modes or improve boundaries
- negative feedback as informative rather than merely noisy
- missing learner-relative temporal dynamics

**Must avoid:**
<!-- MANUSCRIPT:END RELATED-P01 -->

<!-- MANUSCRIPT:BEGIN RELATED-P02 -->
## [RELATED-P02] Off-Policy, Stale, and Low-Probability Updates

**Claim:** Importance correction, clipping, trust regions, stale-policy controls, and rarity-aware rules regulate mismatch or update scale, but usually treat remoteness as a static signal rather than an endogenous repeated-reuse process.

**Reader question:** How does this work differ from established off-policy, stale-policy, and low-probability update controls?

**Role:** Credit mismatch controls, state what they establish, and isolate the dynamic feedback-loop distinction.

**Required evidence:**
- PPO
- SAC and replay-based control
- offline-RL review
- D4RL external setting

**Must include:**
- importance weighting and off-policy correction
- PPO-style clipping and trust regions
- replay and stale-policy control
- low-probability or surprisal-aware regulation
- static versus endogenous remoteness
- aggregate equilibrium consequence

**Must avoid:**
<!-- MANUSCRIPT:END RELATED-P02 -->

<!-- MANUSCRIPT:BEGIN RELATED-P03 -->
## [RELATED-P03] Robust Offline Policy Learning

**Claim:** Conservative value learning, in-support dynamic programming, behavior regularization, and data selection address major offline-RL failures, while DRPO focuses on the signed actor field and selectively preserves local negative information.

**Reader question:** How is DRPO related to conservative and behavior-regularized offline reinforcement learning?

**Role:** Differentiate value extrapolation, policy support, data selection, and signed-actor dynamics while explaining compatibility.

**Required evidence:**
- CQL
- IQL
- off-policy without exploration
- CRR and BPPO

**Must include:**
- CQL-style pessimism
- IQL-style in-support value learning
- behavior regularization and proximal objectives
- quality selection
- signed actor field as distinct object
- compatibility with value-side safeguards

**Must avoid:**
<!-- MANUSCRIPT:END RELATED-P03 -->

# Problem Setup

<!-- MANUSCRIPT:BEGIN SETUP-P01 -->
## [SETUP-P01] Signed Actor Update and Influence Factorization

**Claim:** The analysis studies a historical signed actor field whose per-sample negative influence factors into advantage severity and score geometry, while frequency and directional coherence determine aggregation.

**Reader question:** What exact update field is analyzed, what is held fixed during an actor step, and how are positive and negative contributions separated?

**Role:** Define the historical update distribution, fixed actor-step scope, sign decomposition, influence factorization, aggregation variables, and conclusion boundary.

**Required evidence:**
- Equation signed-field
- Equation influence-factorization
- registered fixed-advantage mechanism scope

**Must include:**
- historical update distribution nu
- fixed advantage-like signal during an actor step
- signed empirical actor field
- positive and negative decomposition
- per-sample severity times score factorization
- sample count and directional coherence
- not assumed to equal exact on-policy gradient

**Must avoid:**
<!-- MANUSCRIPT:END SETUP-P01 -->

<!-- MANUSCRIPT:BEGIN SETUP-P02 -->
## [SETUP-P02] Policy-Relative Far Field

**Claim:** Learner-relative remoteness is negative log probability under the current policy; it maps to Mahalanobis distance for Gaussian policies and surprisal for categorical policies without implying identical amplification laws.

**Reader question:** How is far field defined across continuous, categorical, and sequence policies?

**Role:** Define remoteness and score response, give policy-family mappings, state their dynamic status, and prevent false continuous--discrete equivalence.

**Required evidence:**
- Equations remoteness and score-response
- Gaussian and categorical specializations

**Must include:**
- negative log-probability definition
- squared output-coordinate score response
- Gaussian Mahalanobis-distance mapping
- categorical action surprisal
- sequence normalized NLL analogue
- dynamic learner-relative quantity
- shared coordinate but distinct response laws

**Must avoid:**
<!-- MANUSCRIPT:END SETUP-P02 -->

# Repulsive Dynamics

<!-- MANUSCRIPT:BEGIN THEORY-P01 -->
## [THEORY-P01] Learner-Relative Remoteness and Static Score Response

**Claim:** At a fixed policy, Gaussian mean-score magnitude grows without bound with Mahalanobis remoteness, whereas categorical selected-logit response increases with surprisal but saturates.

**Reader question:** How does learner-relative remoteness determine the strength of a single score contribution before any repeated update is considered?

**Role:** Define the static geometry, state the Gaussian and categorical proposition, interpret both cases, and prevent a premature divergence claim.

**Required evidence:**
- Proposition score-remoteness
- Appendix proof score-remoteness

**Must include:**
- fixed-context output coordinates
- Gaussian lower and upper eigenvalue bounds
- isotropic exact relation
- categorical selected-logit response 1-exp(-D)
- full categorical score bounded
- static geometry does not imply aggregate divergence

**Must avoid:**
<!-- MANUSCRIPT:END THEORY-P01 -->

<!-- MANUSCRIPT:BEGIN THEORY-P02 -->
## [THEORY-P02] Self-Attenuation and Self-Amplification under Reuse

**Claim:** For a convex negative log-likelihood along the update path, repeated negative reuse increases remoteness and cannot decrease score response, while sufficiently small positive reuse decreases both.

**Reader question:** How does repeatedly reusing one fixed historical action turn static geometry into a feedback process?

**Role:** State the repeated-reuse theorem, its assumptions, family-specific consequences, and its limit as a single-sample result.

**Required evidence:**
- Theorem reuse
- Appendix proof reuse

**Must include:**
- fixed historical action and coefficient
- negative reuse ascent in negative log likelihood
- positive reuse descent with smoothness condition
- convexity in Gaussian mean and categorical natural coordinates
- Gaussian unbounded versus categorical persistent consequence
- single-sample result does not settle aggregate balance

**Must avoid:**
<!-- MANUSCRIPT:END THEORY-P02 -->

<!-- MANUSCRIPT:BEGIN THEORY-P03 -->
## [THEORY-P03] Aggregate Attraction--Repulsion Equilibria

**Claim:** In a regular exponential family, positive dominance yields a unique stable finite equilibrium when the signed target is feasible, equality yields persistent drift, and negative dominance admits no stable finite equilibrium.

**Reader question:** When do positive attraction and negative repulsion aggregate into stable extrapolation, drift, or instability?

**Role:** Derive the aggregate field, state all three regimes, interpret extrapolation, give stability conditions, and connect feasibility to boundary loss.

**Required evidence:**
- Theorem aggregate equilibrium
- Appendix detailed proof

**Must include:**
- regular minimal exponential family
- positive and negative masses and moments
- signed field derivation
- positive-dominant interior equilibrium
- extrapolation identity beyond positive target
- critical constant field
- negative-dominant repelling stationary point
- discrete step-size condition
- feasible boundary and no finite equilibrium

**Must avoid:**
<!-- MANUSCRIPT:END THEORY-P03 -->

<!-- MANUSCRIPT:BEGIN THEORY-P04 -->
## [THEORY-P04] Policy-Family Manifestations of Negative Dominance

**Claim:** Negative dominance produces unbounded Gaussian mean displacement and far-field scores, but categorical policies approach the simplex boundary while each direct-logit score remains bounded.

**Reader question:** What does aggregate instability look like in Gaussian and categorical policies, and which failure labels are justified?

**Role:** Derive family-specific runaway, distinguish gradient explosion from bounded-gradient boundary degeneration, and preserve reporting boundaries.

**Required evidence:**
- Theorem family runaway
- Gaussian derivation
- categorical derivation

**Must include:**
- negative-dominant aggregate condition
- Gaussian affine repelling field
- unbounded mean and fixed-action score
- categorical gauge-fixed logits
- simplex-boundary subsequence
- bounded full categorical score
- learned covariance as optional amplifier
- task collapse and NaN/Inf not implied

**Must avoid:**
<!-- MANUSCRIPT:END THEORY-P04 -->

<!-- MANUSCRIPT:BEGIN THEORY-P05 -->
## [THEORY-P05] Observable Regimes and Experimental Predictions

**Claim:** The theory predicts a sequence from a Positive-only platform through stable controlled extrapolation to boundary approach or persistent runaway, with targeted far-field attenuation restoring a finite terminal regime when that path is causal.

**Reader question:** Which observable outcomes would support or contradict the theory?

**Role:** Translate the theory into falsifiable predictions, intervention logic, and a separated failure taxonomy.

**Required evidence:**
- C-U1 E2--E4
- D-U1 E5--E6
- Hopper E7
- Countdown E8

**Must include:**
- Positive-only finite platform
- moderate negative stable extrapolation
- boundary approach under stronger or more outward negative moments
- persistent drift or non-vanishing residual beyond feasibility
- near-removal versus far-removal/capping intervention
- separate task, support-boundary, and numerical events
- controlled versus external responsibilities

**Must avoid:**
<!-- MANUSCRIPT:END THEORY-P05 -->

# Distributionally Robust Policy Optimization

<!-- MANUSCRIPT:BEGIN METHOD-P01 -->
## [METHOD-P01] Distributional Reweighting of Signed Actor Updates

**Claim:** DRPO controls the signed actor-update distribution by exponentially attenuating learner-relative remote negative updates while leaving positive updates unchanged.

**Reader question:** What is the current paper's self-contained DRPO update?

**Role:** Define the main method as one current formulation, without old-versus-new chronology or an unregistered combined quality-weight objective.

**Required evidence:**
- method equation
- ablation family

**Must include:**
- raw signed actor update
- negative exponential distance/surprisal envelope
- positive updates unchanged
- quality and remoteness are distinct analysis axes

**Must avoid:**
- optional quality weights in the main update unless separately registered
- claiming quality filtering and remoteness tapering are identical
<!-- MANUSCRIPT:END METHOD-P01 -->

<!-- MANUSCRIPT:BEGIN METHOD-P02 -->
## [METHOD-P02] Direct Bridge to Theorem 1

**Claim:** DRPO replaces the same aggregate negative moment that moves the equilibrium, allowing the method to preserve local displacement while reducing boundary-driving far-field mass.

**Reader question:** Which exact theorem object does DRPO modify?

**Role:** Create an unbroken theory--method equation chain.

**Required evidence:**
- aggregate-moment diagnostics

**Must include:**
- uncontrolled negative moment
- weighted negative moment
- controlled signed target
- local retention and far attenuation

**Must avoid:**
<!-- MANUSCRIPT:END METHOD-P02 -->

<!-- MANUSCRIPT:BEGIN METHOD-P03 -->
## [METHOD-P03] Proposition 2: Vanishing Weighted Far-Field Gradient

**Claim:** An exponential envelope dominates any finite-order score growth, so the weighted negative gradient vanishes in the far field.

**Reader question:** Why use the exponential form rather than an arbitrary taper?

**Role:** Provide the method-level tail guarantee without inventing a utility-decay law.

**Required evidence:**
- analytic proposition
- distance-binned gradient diagnostics

**Must include:**
- finite-order score-growth condition
- exponential times polynomial tends to zero
- no assumption that sample utility decays exponentially

**Must avoid:**
<!-- MANUSCRIPT:END METHOD-P03 -->

<!-- MANUSCRIPT:BEGIN METHOD-P04 -->
## [METHOD-P04] Controls and Ablations

**Claim:** Positive-only, global scaling, hard thresholds, linear tapers, and exponential DRPO isolate which benefits come from retaining local negatives, selecting distance, and controlling total gradient budget.

**Reader question:** How will the method be tested against simpler controls?

**Role:** Define comparisons without predeclaring a winner or conflating quality and distance thresholds.

**Required evidence:**
- C-U1 method matrix
- D-U1 analogues
- terminal audit

**Must include:**
- uncontrolled signed baseline
- Positive-only
- global negative scaling
- linear or reciprocal distance taper
- hard distance threshold
- matched raw negative-gradient budgets

**Must avoid:**
<!-- MANUSCRIPT:END METHOD-P04 -->

# Experiments

<!-- MANUSCRIPT:BEGIN EXP-P01 -->
## [EXP-P01] Environments and Evidence Roles

**Claim:** C-U1 and D-U1 provide controlled identification, while Hopper/D4RL and Countdown provide external validity; no environment substitutes for another's scientific responsibility.

**Reader question:** Why are these environments sufficient and what does each one prove?

**Role:** Introduce environment construction before reporting results so the claims cannot appear manufactured by an opaque simulator.

**Required evidence:**
- Appendix D full specifications

**Must include:**
- C-U1 6D context and 2D action
- D-U1 shared semantic categorical actor
- same-distribution held-out contexts
- Hopper learned critic
- Countdown shared parameters
- environment responsibility table

**Must avoid:**
<!-- MANUSCRIPT:END EXP-P01 -->

<!-- MANUSCRIPT:BEGIN EXP-P02 -->
## [EXP-P02] RQ1: External Occurrence

**Claim:** Hopper and Countdown test whether far-field or rare negative influence becomes disproportionately large before task degradation in realistic policy learning.

**Reader question:** Does the phenomenon occur outside controlled environments?

**Role:** Use external evidence as a reality anchor without treating it as the causal isolation experiment.

**Required evidence:**
- EXT-H-E7-Q2 terminal-audited outputs
- Countdown formal outputs

**Must include:**
- distance/surprisal bins
- positive/negative imbalance
- temporal ordering
- best and terminal checkpoints
- TBD until formal results close

**Must avoid:**
<!-- MANUSCRIPT:END EXP-P02 -->

<!-- MANUSCRIPT:BEGIN EXP-P03 -->
## [EXP-P03] RQ2a: Matched Badness--Distance and Badness--Rarity Isolation

**Claim:** When quality, advantage, semantics, count, coefficient, and policy stage are matched, learner-relative distance or rarity independently changes negative influence.

**Reader question:** Are far or rare gradients larger because the samples are farther, or merely because they are worse?

**Role:** Close the paper's decisive rival explanation with a transparent matched protocol.

**Required evidence:**
- C-U1 E1
- D-U1 matched analogue

**Must include:**
- same context and quality coordinate
- same reward and advantage magnitude
- same count and base weight
- only distance or rarity changes
- score and full-parameter gradients
- direction coherence

**Must avoid:**
<!-- MANUSCRIPT:END EXP-P03 -->

<!-- MANUSCRIPT:BEGIN EXP-P04 -->
## [EXP-P04] RQ2b: Targeted Causal Transmission

**Claim:** Near/far and common/rare interventions test whether the anomalous remote component is the pathway that transmits negative influence into drift, boundary events, and task collapse.

**Reader question:** Do large far-field updates actually cause the observed instability?

**Role:** Move from source identification to causal intervention with equal-budget controls.

**Required evidence:**
- C-U1 E3
- D-U1 causal protocol

**Must include:**
- uncontrolled signed baseline
- near/common removal
- far/rare removal
- far/rare cap
- global equal-budget control
- budget transfer
- separate failure events

**Must avoid:**
<!-- MANUSCRIPT:END EXP-P04 -->

<!-- MANUSCRIPT:BEGIN EXP-P05 -->
## [EXP-P05] RQ3: Phase Transition and DRPO Control

**Claim:** Strength sweeps map the regimes of Theorem 1, while DRPO and matched controls test whether selective far-field attenuation preserves stable extrapolation.

**Reader question:** When does negative feedback become destructive, and can DRPO control that transition?

**Role:** Unify theorem validation, aggregate-term measurement, and method comparison in one result section.

**Required evidence:**
- C-U1 E4
- D-U1 E6
- registered taper experiments

**Must include:**
- Positive-only to stable extrapolation to boundary/drift
- continuous and categorical phase maps
- aggregate negative moment proxy
- distance bins
- paired seeds and raw-budget matching
- best and terminal reporting

**Must avoid:**
<!-- MANUSCRIPT:END EXP-P05 -->

<!-- MANUSCRIPT:BEGIN EXP-P06 -->
## [EXP-P06] RQ4: External Task Closure

**Claim:** External D4RL and Countdown evaluations test whether controlling remote negative influence improves actual task performance without sacrificing the useful negative signal.

**Reader question:** Does the mechanism-targeted control matter on public and sequence tasks?

**Role:** Close the external--controlled--external evidence chain.

**Required evidence:**
- EXT-H-E7-BENCH-01
- EXT-C-E8-SCALE-01

**Must include:**
- D4RL locomotion datasets
- Countdown model scales and common data bank
- same initialization and selection rules
- best and terminal metrics
- mechanism diagnostics beside performance

**Must avoid:**
<!-- MANUSCRIPT:END EXP-P06 -->

# Implications and Conclusion

<!-- MANUSCRIPT:BEGIN DISC-P01 -->
## [DISC-P01] Negative Feedback Is a Resource with a Dynamical Failure Mode

**Claim:** The transferable lesson is to control learner-relative far-field negative influence rather than classify all negative feedback as harmful.

**Reader question:** What should readers remember beyond the specific method?

**Role:** State the positive principle rather than a list of disclaimers.

**Required evidence:**
- theory and controlled evidence

**Must include:**
- negative feedback supports extrapolation
- historical reuse changes its relevance and magnitude
- control object is remote negative influence

**Must avoid:**
<!-- MANUSCRIPT:END DISC-P01 -->

<!-- MANUSCRIPT:BEGIN DISC-P02 -->
## [DISC-P02] Continuous and Categorical Synthesis

**Claim:** Gaussian and categorical policies share aggregate boundary dynamics while differing in local score growth and failure manifestation.

**Reader question:** What is genuinely unified across policy families?

**Role:** Leave one accurate cross-family synthesis.

**Required evidence:**
- family corollaries
- C-U1/D-U1 evidence

**Must include:**
- Gaussian distance-amplified score and support dynamics
- categorical bounded persistent suppression
- shared feasibility-boundary transition

**Must avoid:**
<!-- MANUSCRIPT:END DISC-P02 -->

<!-- MANUSCRIPT:BEGIN DISC-P03 -->
## [DISC-P03] Conclusion

**Claim:** DRPO preserves useful negative feedback by attenuating the remote component of the aggregate term that drives equilibrium loss.

**Reader question:** What is the final three-sentence contribution?

**Role:** Close with theory, identification, and method consequence.

**Required evidence:**
- all main claims

**Must include:**
- stable extrapolation
- badness--distance isolation
- far-field causal pathway
- DRPO recovery

**Must avoid:**
<!-- MANUSCRIPT:END DISC-P03 -->

# Proofs for Repulsive Dynamics

<!-- MANUSCRIPT:BEGIN APP-PROOF-P01 -->
## [APP-PROOF-P01] Proof of Theorem 1

**Claim:** The equilibrium theorem follows from the diffeomorphism between natural and mean parameters in a regular minimal exponential family.

**Reader question:** What are the exact existence, uniqueness, boundary, and stability arguments?

**Role:** Provide the full proof outside the main narrative.

**Required evidence:**
- standard exponential-family properties

**Must include:**
- derive aggregate field
- interior mean-space existence
- boundary divergence
- Jacobian

**Must avoid:**
<!-- MANUSCRIPT:END APP-PROOF-P01 -->

# Gaussian Mean--Variance Derivations

<!-- MANUSCRIPT:BEGIN APP-GAUSS-P01 -->
## [APP-GAUSS-P01] Corrected Gaussian Mean and Variance Dynamics

**Claim:** Far-field negative advantages repel the mean and tend to contract variance when standardized distance exceeds one.

**Reader question:** What corrects the earlier mean-and-variance expansion account?

**Role:** Record the exact score signs and joint equilibrium condition.

**Required evidence:**
- analytic Gaussian score

**Must include:**
- mean score
- log-standard-deviation score
- four sign/location quadrants
- joint equilibrium

**Must avoid:**
<!-- MANUSCRIPT:END APP-GAUSS-P01 -->

# Categorical Boundary Dynamics

<!-- MANUSCRIPT:BEGIN APP-CAT-P01 -->
## [APP-CAT-P01] Categorical Log-Odds and Boundary Behavior

**Claim:** Repeated negative updates produce persistent log-odds decay even though the direct-logit score norm is bounded.

**Reader question:** Why is categorical instability not an unbounded Euclidean gradient explosion?

**Role:** Derive the probability-boundary mechanism accurately.

**Required evidence:**
- categorical algebra

**Must include:**
- softmax score bound
- log-odds update
- probability decay
- shared-parameter caveat

**Must avoid:**
<!-- MANUSCRIPT:END APP-CAT-P01 -->

# Controlled Environments

<!-- MANUSCRIPT:BEGIN APP-ENV-P01 -->
## [APP-ENV-P01] C-U1 and D-U1 Construction

**Claim:** The controlled environments expose ground truth and permit matched isolation without claiming that construction alone establishes external validity.

**Reader question:** How exactly are the controlled environments generated?

**Role:** Provide enough detail for reproducibility and reviewer inspection.

**Required evidence:**
- environment code and manifests

**Must include:**
- C-U1 states/actions/reward/hidden optimum
- train/test same distribution
- D-U1 semantic actions
- matched probes
- dynamic near/far

**Must avoid:**
<!-- MANUSCRIPT:END APP-ENV-P01 -->

# Experimental Protocols and Terminal Audits

<!-- MANUSCRIPT:BEGIN APP-PROT-P01 -->
## [APP-PROT-P01] Stopping, Budget Matching, and Terminal Classification

**Claim:** All dynamic and ranking claims require fixed protocol ownership, paired seeds, explicit budgets, and terminal audits.

**Reader question:** How are false convergence and unfair comparisons prevented?

**Role:** Centralize experiment governance details.

**Required evidence:**
- handoff and registry

**Must include:**
- maximum horizon
- terminal slopes/residuals
- 2x continuation where registered
- raw pre-optimizer negative-gradient budget
- best and terminal

**Must avoid:**
<!-- MANUSCRIPT:END APP-PROT-P01 -->

# Additional Results and Failure Taxonomy

<!-- MANUSCRIPT:BEGIN APP-RES-P01 -->
## [APP-RES-P01] Additional Tables, Curves, and Negative Results

**Claim:** Appendix results preserve uncertainty, failed runs, and non-ranking outcomes instead of presenting only favorable endpoints.

**Reader question:** What evidence is needed beyond the main figures?

**Role:** Provide placeholders for complete result deposition.

**Required evidence:**
- formal artifact packages

**Must include:**
- per-seed tables
- full trajectories
- sensitivity
- negative results
- failure inventory

**Must avoid:**
<!-- MANUSCRIPT:END APP-RES-P01 -->

# Implementation and Reproducibility

<!-- MANUSCRIPT:BEGIN APP-REPRO-P01 -->
## [APP-REPRO-P01] Code, Data, and Artifact Provenance

**Claim:** Every formal claim is bound to code, configuration, seeds, raw outputs, terminal audit, and a source commit.

**Reader question:** How can the study be reproduced and audited?

**Role:** Connect the paper to repository provenance without embedding live status in the outline.

**Required evidence:**
- repository scripts and artifacts

**Must include:**
- repository paths
- run manifests
- checksums
- formal package lifecycle
- Overleaf build

**Must avoid:**
<!-- MANUSCRIPT:END APP-REPRO-P01 -->

# Optimistic Distributional Selection

<!-- MANUSCRIPT:BEGIN APP-DRO-P01 -->
## [APP-DRO-P01] Optimistic-DRO Quality Selection and Its Distinct Role

**Claim:** The Optimistic-DRO subdistribution problem yields a quality-based hard-selection solution; this result is part of the current manuscript but is mathematically distinct from learner-relative remoteness tapering.

**Reader question:** What does the Optimistic-DRO result establish in the current paper, and what does it not establish?

**Role:** Integrate the valid distributional-selection derivation into the current manuscript while keeping quality selection mathematically distinct from learner-relative remoteness tapering.

**Required evidence:**
- closed-form density-ratio-constrained subdistribution solution
- explicit separation of quality and remoteness axes

**Must include:**
- quality-based uncertainty set
- top-quality hard-selection solution
- current-paper integration
- no implication that hard quality filtering equals exponential distance tapering

**Must avoid:**
- split-manuscript chronology
- claim that quality hard filtering derives exponential remoteness tapering
<!-- MANUSCRIPT:END APP-DRO-P01 -->

# Additional Theoretical and Reporting Clarifications

<!-- MANUSCRIPT:BEGIN APP-CORR-P01 -->
## [APP-CORR-P01] Additional Theoretical and Reporting Clarifications

**Claim:** The manuscript consistently uses corrected Gaussian scale dynamics, the Jacobian of the actual signed field, held-out-context terminology for C-U1, and separate reporting of task, boundary, and numerical failures.

**Reader question:** Which technical conventions must remain consistent across the main text and appendix?

**Role:** Collect the technical conventions that prevent contradictory statements across theory, experiments, and reporting.

**Required evidence:**
- corrected Gaussian derivation
- registered reporting protocol
- terminal-audit governance

**Must include:**
- Gaussian mean repulsion with location-dependent scale dynamics
- actual signed-field Jacobian rather than expected Fisher as the stability object
- held-out-context or unseen-state generalization for C-U1
- separate task collapse, policy-family boundary, and NaN/Inf fields
- terminal audit before convergence or long-run language

**Must avoid:**
- historical chronology
- defensive discussion of manuscript versions
<!-- MANUSCRIPT:END APP-CORR-P01 -->
